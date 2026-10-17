"""工程模型：音频源、通道（Lane）、多文件合并通道、单文件解交织。

时间轴约定
----------
所有源在 ``帧 0`` 处对齐，共享一条全局时间轴；工程采样率由用户选择（默认取最长的源），
采样率不同或长度不足的源在读取时按比例换算并补零。这样「多文件纵向堆叠比对」和
「合并成一个新的多声道流（可导出）」可以共用同一套数据通路。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .audio_io import (
    AudioBuffer,
    PcmFormat,
    audio_kind,
    export_wav,
    load_audio,
    new_object_id,
)

CHANNEL_COLORS = [
    "#4fb3ff", "#ff8a4c", "#5ddb7a", "#f2d64b", "#c78bff", "#ff6b8b",
    "#4ce0d8", "#b6e05c", "#ff9ecb", "#9eb6ff", "#e0a45c", "#69f0ae",
    "#f06292", "#aed581", "#80deea", "#ffcc80",
]


def color_for(i: int) -> str:
    return CHANNEL_COLORS[i % len(CHANNEL_COLORS)]


@dataclass
class Lane:
    """一个可显示 / 可播放的通道。"""

    source_id: str
    index: int                              # 源内通道序号
    label: str
    color: str = "#4fb3ff"
    visible: bool = True
    origin: tuple[str, int] | None = None   # 真实数据来源 (source_id, channel)

    @property
    def key(self) -> str:
        return f"{self.source_id}#{self.index}"

    def raw_ref(self) -> tuple[str, int]:
        return self.origin if self.origin else (self.source_id, self.index)


class MixBuffer(AudioBuffer):
    """合并源的虚拟 buffer：长度/采样率由成员推得，读取回落到真实文件。"""

    def __init__(self, project: "Project", parts: list[Lane]):
        self.project = project
        self.parts = parts
        rate = 0
        frames = 0
        for p in parts:
            buf = project.buffer_of(p.raw_ref()[0])
            if buf is None:
                continue
            rate = rate or buf.sample_rate
            if rate:
                frames = max(frames, int(buf.frames * buf.sample_rate / rate))
        rate = rate or 48000
        fmt = PcmFormat(sample_rate=rate, channels=max(1, len(parts)), sample_width=4, is_float=True)
        super().__init__(
            raw=np.zeros((0, max(1, len(parts))), dtype=np.float32),
            frames=frames,
            channels=max(1, len(parts)),
            sample_rate=rate,
            fmt=fmt,
            is_int24=False,
            path="<mix>",
        )
        self.is_mix = True

    def read(self, start, length, channels=None):  # type: ignore[override]
        chs = list(range(self.channels)) if channels is None else [c for c in channels if 0 <= c < len(self.parts)]
        lanes = [self.parts[c] for c in chs]
        return self.project.read(lanes, start, length, self.sample_rate)


@dataclass
class AudioSource:
    sid: str
    label: str
    path: str | None
    buffer: AudioBuffer
    lanes: list[Lane] = field(default_factory=list)
    kind: str = "file"                 # "file" | "mix"
    format_desc: str = ""

    @property
    def channel_count(self) -> int:
        return len(self.lanes)

    @property
    def frames(self) -> int:
        return self.buffer.frames if self.buffer is not None else 0

    @property
    def sample_rate(self) -> int:
        return self.buffer.sample_rate if self.buffer is not None else 0

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


class Project:
    """一次会话中打开的全部音频源 + 全局时间轴。"""

    def __init__(self) -> None:
        self.sources: dict[str, AudioSource] = {}
        self.order: list[str] = []
        self._color_seq = 0
        self.sample_rate: int | None = None

    # ------------------------------------------------------------------
    # 装载
    # ------------------------------------------------------------------
    def open_file(self, path: str, fmt: PcmFormat | None = None) -> AudioSource:
        buffer = load_audio(path, fmt)
        sid = new_object_id()
        lanes = [
            Lane(source_id=sid, index=i, label=f"CH{i}", color=self._next_color(),
                 visible=True, origin=(sid, i))
            for i in range(buffer.channels)
        ]
        kind = audio_kind(path)
        desc = f"WAV · {getattr(buffer, 'wav_subtype', 'PCM_16')}" if kind == "wav" \
            else (buffer.fmt.describe() if buffer.fmt else "PCM")
        src = AudioSource(sid=sid, label=os.path.basename(path), path=path, buffer=buffer,
                          lanes=lanes, kind="file", format_desc=desc)
        self.sources[sid] = src
        self.order.append(sid)
        if self.sample_rate is None or buffer.frames > self.total_frames:
            self.sample_rate = buffer.sample_rate
        return src

    def _next_color(self) -> str:
        c = color_for(self._color_seq)
        self._color_seq += 1
        return c

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def all_lanes(self) -> list[Lane]:
        out: list[Lane] = []
        for sid in self.order:
            src = self.sources.get(sid)
            if src:
                out.extend(src.lanes)
        return out

    def file_sources(self) -> list[AudioSource]:
        return [self.sources[s] for s in self.order if s in self.sources]

    def lane(self, key: str) -> Lane | None:
        for lane in self.all_lanes():
            if lane.key == key:
                return lane
        return None

    def lanes(self, keys: Iterable[str]) -> list[Lane]:
        wanted = set(keys)
        return [l for l in self.all_lanes() if l.key in wanted]

    def visible_lanes(self) -> list[Lane]:
        return [l for l in self.all_lanes() if l.visible]

    def buffer_of(self, sid: str) -> AudioBuffer | None:
        src = self.sources.get(sid)
        return src.buffer if src else None

    @property
    def total_frames(self) -> int:
        vals = [s.frames for s in self.file_sources()]
        return max(vals) if vals else 0

    @property
    def duration(self) -> float:
        rate = self.sample_rate or 48000
        return self.total_frames / rate if rate else 0.0

    def available_rates(self) -> list[int]:
        seen: list[int] = []
        for src in self.file_sources():
            if src.sample_rate and src.sample_rate not in seen:
                seen.append(src.sample_rate)
        return seen

    def set_sample_rate(self, rate: int) -> None:
        self.sample_rate = int(rate)

    def is_empty(self) -> bool:
        return not self.order

    # ------------------------------------------------------------------
    # 数据访问（UI / DSP / 播放共用唯一通路）
    # ------------------------------------------------------------------
    def read(self, lanes: list[Lane], start: int, length: int, rate: int | None = None) -> np.ndarray:
        """读 ``(length, len(lanes))`` 的 float32；跨采样率做线性换算，越界补零。"""
        length = max(0, int(length))
        if length == 0 or not lanes:
            return np.zeros((length, max(0, len(lanes))), dtype=np.float32)
        out = np.zeros((length, len(lanes)), dtype=np.float32)
        rate_out = rate or self.sample_rate

        groups: dict[str, list[tuple[int, int]]] = {}
        for col, lane in enumerate(lanes):
            oid, ch = lane.raw_ref()
            groups.setdefault(oid, []).append((col, ch))

        for oid, items in groups.items():
            buf = self.buffer_of(oid)
            if buf is None:
                continue
            rate_src = buf.sample_rate
            ro = rate_out or rate_src
            scale = rate_src / ro if ro else 1.0
            s0 = int(start * scale)
            # 输出第 i 帧对应源第 i*scale 帧：必须按此线性映射，不能用 linspace 强制端点
            n = int(np.floor((length - 1) * scale)) + 2
            cols = [c for c, _ in items]
            chs = [ch for _, ch in items]
            block = buf.read(s0, n, chs)
            if block.shape[0] == 0:
                continue
            if abs(scale - 1.0) > 1e-9 and block.shape[0] > 1:
                pos = np.minimum(np.arange(length, dtype=np.float64) * scale, block.shape[0] - 1)
                lo = np.floor(pos).astype(np.int64)
                hi = np.minimum(lo + 1, block.shape[0] - 1)
                w = (pos - lo).astype(np.float32)[:, None]
                block = (block[lo] * (1.0 - w) + block[hi] * w).astype(np.float32)
            take = min(block.shape[0], length)
            for j, col in enumerate(cols):
                out[:take, col] = block[:take, j]
        return out

    # ------------------------------------------------------------------
    # 合并通道 / 解交织
    # ------------------------------------------------------------------
    def create_mix(self, name: str, lanes: list[Lane], silence_sources: bool = True) -> AudioSource:
        """把选中的多个通道（可跨文件）串成一个新的多声道源。

        ``silence_sources=True`` 时自动取消被并入的原始通道勾选，避免同一段音频
        既作为独立轨道、又作为合并流被重复播放；原始通道仍保留，可随时勾回。
        """
        if not lanes:
            raise ValueError("没有选中任何通道")
        sid = new_object_id()
        parts: list[Lane] = []
        for i, lane in enumerate(lanes):
            parts.append(Lane(source_id=sid, index=i, label=f"CH{i}", color=self._next_color(),
                              visible=True, origin=lane.raw_ref()))
        src = AudioSource(sid=sid, label=name, path=None, buffer=None, lanes=parts,
                          kind="mix", format_desc=f"合并通道 · {len(parts)} ch")
        src.buffer = MixBuffer(self, parts)
        self.sources[sid] = src
        self.order.append(sid)
        if silence_sources:
            for lane in lanes:
                lane.visible = False
        return src

    def remove_sources(self, sids: Iterable[str]) -> None:
        for sid in set(sids):
            self.sources.pop(sid, None)
        self.order = [s for s in self.order if s in self.sources]
        for src in self.file_sources():
            if src.kind != "mix":
                continue
            keep = [p for p in src.lanes if p.raw_ref()[0] in self.sources]
            for i, p in enumerate(keep):
                p.index = i
            src.lanes = keep
            src.buffer = MixBuffer(self, keep)
        empty_mixes = [s.sid for s in self.file_sources() if s.kind == "mix" and not s.lanes]
        if empty_mixes:
            self.remove_sources(empty_mixes)

    def set_visible(self, keys: Iterable[str], visible: bool) -> None:
        for lane in self.lanes(keys):
            lane.visible = visible

    def toggle_visible(self, keys: Iterable[str]) -> bool:
        lanes = self.lanes(keys)
        target = not all(l.visible for l in lanes) if lanes else False
        for lane in lanes:
            lane.visible = target
        return target

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def render(self, lanes: list[Lane], rate: int | None = None, chunk: int = 1 << 20) -> np.ndarray:
        rate = rate or self.sample_rate or 48000
        total = self.total_frames
        if total == 0:
            return np.zeros((0, max(1, len(lanes))), dtype=np.float32)
        pieces = [self.read(lanes, off, min(chunk, total - off), rate) for off in range(0, total, chunk)]
        return np.concatenate(pieces, axis=0) if pieces else np.zeros((0, len(lanes)), np.float32)

    def export_lanes(self, path: str, lanes: list[Lane], rate: int | None = None,
                     subtype: str = "PCM_16") -> tuple[int, int]:
        """导出选中的通道（合并流 / 单通道 / 多文件堆叠），返回 (帧数, 通道数)。"""
        if not lanes:
            raise ValueError("没有选中任何通道")
        data = self.render(lanes, rate)
        rate = rate or self.sample_rate or 48000
        export_wav(path, data, rate, subtype)
        return int(data.shape[0]), int(data.shape[1])

    def export_deinterleaved(self, outdir: str, sid: str, subtype: str = "PCM_16") -> list[str]:
        """把一个源的所有通道解交织成 N 个单声道 WAV。"""
        src = self.sources.get(sid)
        if src is None:
            raise ValueError("源不存在")
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(src.label)[0]
        written = []
        for lane in src.lanes:
            out = os.path.join(outdir, f"{base}_{lane.label}.wav")
            self.export_lanes(out, [lane], src.sample_rate or self.sample_rate, subtype)
            written.append(out)
        return written

    def export_pcm(self, path: str, lanes: list[Lane], fmt: PcmFormat, rate: int | None = None) -> int:
        """导出裸 PCM（用于回灌到 DSP/声卡调试链路）。"""
        data = self.render(lanes, rate)
        rate = rate or self.sample_rate or 48000
        arr = np.clip(data, -1.0, 1.0)
        if fmt.is_float:
            out = arr.astype(np.float32 if fmt.sample_width == 4 else np.float64)
        elif fmt.sample_width == 1:
            q = np.round(arr * 127.0 + (0 if fmt.signed else 128.0))
            out = q.astype(np.int8 if fmt.signed else np.uint8)
        elif fmt.sample_width == 3:
            q = np.round(arr * ((1 << 23) - 1)).astype(np.int32)
            q = q.ravel()
            b = (q[:, None] >> np.array([0, 8, 16], dtype=np.int32)[None, :]) & 0xFF
            b = b.astype(np.uint8).reshape(-1, 3)
            out = b if not fmt.big_endian else b[:, ::-1]
            out = out.ravel()
        else:
            bits = fmt.sample_width * 8
            q = np.round(arr * ((1 << (bits - 1)) - 1))
            out = q.astype(fmt.dtype)
        if fmt.big_endian and not fmt.is_float and fmt.sample_width != 3:
            out = out.astype(out.dtype.newbyteorder(">"))
        elif fmt.is_float and fmt.big_endian:
            out = out.astype(out.dtype.newbyteorder(">"))
        out = np.ascontiguousarray(out)
        with open(path, "wb") as fh:
            if fmt.offset:
                fh.write(b"\x00" * fmt.offset)
            fh.write(out.tobytes())
        return int(out.nbytes)
