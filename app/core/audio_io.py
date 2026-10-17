"""音频数据读取层：裸 PCM / WAV -> 统一的 float32 通道矩阵访问接口。

设计要点
--------
* 大文件友好：PCM 用 ``numpy.memmap`` 直接映射；WAV 用 ``soundfile(memory_map=True)``。
* 解码集中在 :meth:`AudioBuffer.read`，返回 ``float32`` 且范围归一到 [-1, 1]。
* 支持 8/16/24/32 位整型（有/无符号、大小端）、32/64 位浮点、交织与平面（planar）排布、
  文件头偏移。
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

try:  # soundfile 在部分环境下可能缺失，只影响 WAV
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


WIDTH_DTYPE = {1: "u1", 2: "i2", 3: "i4", 4: "i4", 8: "f8"}


@dataclass
class PcmFormat:
    """裸 PCM 的显式参数（文件本身不带任何元信息）。"""

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2          # 每样本字节数：1/2/3/4/8
    is_float: bool = False
    signed: bool = True
    big_endian: bool = False
    offset: int = 0                # 文件头需要跳过的字节数
    planar: bool = False           # 平面（分通道）排布而非全交织

    @property
    def block_align(self) -> int:
        return self.sample_width * self.channels

    @property
    def dtype(self) -> np.dtype:
        endian = ">" if self.big_endian else "<"
        if self.is_float:
            code = "f4" if self.sample_width == 4 else "f8"
        elif self.sample_width == 1:
            code = "i1" if self.signed else "u1"
        elif self.sample_width == 3:
            code = "i4"  # 24bit 手工展开成 int32
        else:
            code = {2: "i2", 4: "i4"}[self.sample_width]
        return np.dtype(endian + code)

    @property
    def kind(self) -> str:
        if self.is_float:
            return f"float{self.sample_width * 8}"
        base = {1: "int8", 2: "int16", 3: "int24", 4: "int32"}[self.sample_width]
        if base == "int8" and not self.signed:
            base = "uint8"
        return base

    @property
    def layout(self) -> str:
        return "planar" if self.planar else "interleaved"

    def describe(self) -> str:
        bits = "f32" if self.is_float and self.sample_width == 4 else (
            "f64" if self.is_float else f"{self.sample_width * 8}bit")
        endian = "BE" if self.big_endian else "LE"
        sign = "" if self.is_float or (self.sample_width != 1) else (
            "/u" if not self.signed else "")
        return f"PCM {self.sample_rate}Hz · {self.channels}ch · {bits}{sign} · {endian} · {self.layout}"

    def key(self) -> str:
        return uuid.uuid5(
            uuid.NAMESPACE_URL, "pcm:" + "|".join(f"{k}={v}" for k, v in sorted(asdict(self).items()))
        ).hex

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PcmFormat":
        keep = {f for f in PcmFormat.__dataclass_fields__}
        return PcmFormat(**{k: v for k, v in d.items() if k in keep})


class AudioBuffer:
    """一段音频的原始（未解码）采样，可按帧区间 + 通道号取 float32。"""

    def __init__(
        self,
        raw: np.ndarray,
        frames: int,
        channels: int,
        sample_rate: int,
        fmt: PcmFormat | None,
        is_int24: bool,
        path: str,
    ):
        self.raw = raw
        self.frames = int(frames)
        self.channels = int(channels)
        self.sample_rate = int(sample_rate)
        self.fmt = fmt
        self.is_int24 = is_int24
        self.path = path
        self._gain: float | None = None

    # ---- 元信息 ---------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def gain(self) -> float:
        """整型 -> [-1,1) 的缩放系数；浮点型为 1。"""
        if self._gain is None:
            if self.is_int24:
                self._gain = 1.0 / float(1 << 23)
            else:
                d = self.raw.dtype
                if d.kind == "f":
                    self._gain = 1.0
                elif d.kind == "u":
                    self._gain = 1.0 / 128.0  # uint8: (x-128)/128
                else:
                    self._gain = 1.0 / float(1 << (d.itemsize * 8 - 1))
        return self._gain

    # ---- 解码 -----------------------------------------------------------
    def decode(self, block: np.ndarray) -> np.ndarray:
        """把原始采样块转成 float32，范围 [-1, 1]。"""
        if self.is_int24:
            # block: uint8 (n, 3)（已是小端/大端序，取决于 fmt）
            b = block.astype(np.uint32)
            if self.fmt and self.fmt.big_endian:
                v = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
            else:
                v = (b[:, 2] << 16) | (b[:, 1] << 8) | b[:, 0]
            v = v.astype(np.int64)
            v = np.where(v >= (1 << 23), v - (1 << 24), v)
            return (v * self.gain).astype(np.float32)

        d = self.raw.dtype
        if d.kind == "f":
            return block.astype(np.float32)
        if d.kind == "u":
            return ((block.astype(np.float32) - 128.0) * self.gain).astype(np.float32)
        return (block.astype(np.float32) * self.gain).astype(np.float32)

    def raw_slice(self, start: int, length: int, channels: list[int] | None = None) -> np.ndarray:
        """取原始采样。

        常规位深返回 ``(available, n_channels)``；24bit（3 字节 packed）返回
        ``(available * n_channels, 3)``，由 :meth:`decode` 负责展开后 reshape。
        ``available`` 可能小于 ``length``（读到了文件末尾）。
        """
        start = max(0, min(start, self.frames))
        length = max(0, min(length, self.frames - start))
        chs = list(range(self.channels)) if channels is None else list(channels)
        if length == 0:
            return np.zeros((0, len(chs) * (3 if self.is_int24 else 1)), dtype=np.uint8)
        planar = bool(self.fmt and self.fmt.planar)
        if self.is_int24:
            if planar:
                rows = np.stack([self.raw[c * self.frames + start: c * self.frames + start + length]
                                 for c in chs], axis=1)
            else:
                rows = self.raw[start:start + length][:, chs, :]
            return rows.reshape(length * len(chs), 3)
        if planar:
            out = np.stack([self.raw[c * self.frames + start: c * self.frames + start + length] for c in chs], axis=1)
        else:
            out = self.raw[start:start + length][:, chs]
        return np.ascontiguousarray(out)

    def read(self, start: int, length: int, channels: list[int] | None = None) -> np.ndarray:
        """读并解码为 float32 ``(length, n_channels)``；越界部分补零。"""
        chs = list(range(self.channels)) if channels is None else list(channels)
        out = np.zeros((max(0, int(length)), len(chs)), dtype=np.float32)
        if out.shape[0] == 0 or not chs:
            return out
        block = self.raw_slice(start, out.shape[0], chs)
        dec = np.asarray(self.decode(block), dtype=np.float32).reshape(-1, len(chs))
        take = min(dec.shape[0], out.shape[0])
        if take > 0:
            out[:take] = dec[:take]
        return out


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------


def _resolve_pcm_format(path: str, fmt: PcmFormat | None) -> PcmFormat:
    if fmt is not None:
        return fmt
    return PcmFormat()


def load_pcm(path: str, fmt: PcmFormat | None = None) -> AudioBuffer:
    fmt = _resolve_pcm_format(path, fmt)
    size = os.path.getsize(path)
    if size <= fmt.offset:
        raise ValueError(f"文件过小或偏移过大：{size} 字节 / 偏移 {fmt.offset}")
    usable = size - fmt.offset
    if fmt.sample_width == 3:
        frames = usable // (3 * fmt.channels)
        raw = np.memmap(path, dtype=np.uint8, mode="r", offset=fmt.offset,
                        shape=(frames, fmt.channels, 3))
        return AudioBuffer(raw, frames, fmt.channels, fmt.sample_rate, fmt, True, path)
    bytes_per_frame = fmt.block_align
    frames = usable // bytes_per_frame
    if frames <= 0:
        raise ValueError("按给定参数解析出的帧数为 0，请检查通道数/位深/偏移")
    if fmt.planar:
        shape = (fmt.channels * frames,)
    else:
        shape = (frames, fmt.channels)
    raw = np.memmap(path, dtype=fmt.dtype, mode="r", offset=fmt.offset, shape=shape)
    return AudioBuffer(raw, frames, fmt.channels, fmt.sample_rate, fmt, False, path)


def load_wav(path: str, memory_map: bool = True) -> AudioBuffer:
    if sf is None:
        raise RuntimeError("未安装 soundfile，无法读取 WAV 文件")
    with sf.SoundFile(path) as fh:
        rate = int(fh.samplerate)
        chans = int(fh.channels)
        frames = int(fh.frames)
        subtype = fh.subtype
        endian_hint = "BE" if "XR" in subtype or "VF" in subtype else "LE"
    if frames == 0:
        raise ValueError("WAV 文件没有有效帧")
    if memory_map:
        try:
            data, rate2 = sf.read(path, dtype="float32", always_2d=True, memory_map=True)
            # soundfile 的 memory_map 会写出一个裸 LE float32 临时文件
            fmt = PcmFormat(sample_rate=rate, channels=chans, sample_width=4, is_float=True)
            buf = AudioBuffer(np.asarray(data), frames, chans, rate, fmt, False, path)
            buf.wav_subtype = subtype  # 仅用于展示
            return buf
        except Exception:
            pass  # 退化为整体读取
    data, _ = sf.read(path, dtype="float32", always_2d=True)
    fmt = PcmFormat(sample_rate=rate, channels=chans, sample_width=4, is_float=True)
    buf = AudioBuffer(np.ascontiguousarray(data), frames, chans, rate, fmt, False, path)
    buf.wav_subtype = subtype
    return buf


def wav_subtype_endian(subtype: str) -> str:
    return "BE" if subtype and ("XR" in subtype or "VF" in subtype) else "LE"


def load_audio(path: str, fmt: PcmFormat | None = None) -> AudioBuffer:
    """按扩展名/魔数自动选择 WAV 或 PCM 装载。"""
    with open(path, "rb") as fh:
        head = fh.read(12)
    is_wav = head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if is_wav or path.lower().endswith((".wav", ".wave", ".bnf", ".aif", ".aiff", ".flac", ".ogg")):
        try:
            return load_wav(path)
        except Exception:
            if not path.lower().endswith((".wav", ".wave")):
                raise
    return load_pcm(path, fmt)


def audio_kind(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
        return "wav" if head[:4] == b"RIFF" and head[8:12] == b"WAVE" else "pcm"
    except OSError:
        return "?"


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def export_wav(path: str, data: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> None:
    """``data`` 为 ``(frames, channels)`` 的 float32。"""
    if sf is None:
        raise RuntimeError("未安装 soundfile，无法导出 WAV")
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    if arr.ndim == 1:
        arr = arr[:, None]
    sf.write(path, arr, int(sample_rate), subtype=subtype)


def new_object_id() -> str:
    return uuid.uuid4().hex[:8]
