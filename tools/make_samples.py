"""生成验收素材：覆盖 WAV/PCM、多种位深、字节序、交织/平面、单/多通道。

用法：``.venv/bin/python tools/make_samples.py [输出目录]``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

T = 2 * np.pi


def tone(f0: float, n: int, rate: int, amp: float = 0.7, sweep_to: float | None = None) -> np.ndarray:
    x = np.arange(n) / rate
    if sweep_to:
        phase = T * x * (f0 + (sweep_to - f0) * x / max(x[-1], 1e-9)) / 2
    else:
        phase = T * f0 * x
    return (amp * np.sin(phase)).astype(np.float32)


def to_int16(x): return np.round(np.clip(x, -1, 1) * 32767).astype("<i2")
def to_int16_be(x): return np.round(np.clip(x, -1, 1) * 32767).astype(">i2")
def to_uint8(x): return np.round(np.clip(x, -1, 1) * 127 + 128).astype(np.uint8)
def to_int32(x): return np.round(np.clip(x, -1, 1) * 2147483000).astype("<i4")


def to_int24_le(x):
    q = np.round(np.clip(x, -1, 1) * ((1 << 23) - 1)).astype(np.int32)
    q &= 0xFFFFFF
    b = np.stack([(q >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.uint8)
    return b.reshape(-1)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest: dict[str, dict] = {}

    def write(name: str, data: bytes | np.ndarray, meta: dict) -> None:
        path = os.path.join(out_dir, name)
        raw = data.tobytes() if isinstance(data, np.ndarray) else data
        with open(path, "wb") as fh:
            fh.write(raw)
        manifest[name] = meta
        print(f"  {name:38s} {len(raw) / 1e6:8.2f} MB  {meta['desc']}")

    def write_wav(name: str, data: np.ndarray, rate: int, subtype: str, meta: dict) -> None:
        path = os.path.join(out_dir, name)
        sf.write(path, np.ascontiguousarray(data, dtype=np.float32), rate, subtype=subtype)
        size = os.path.getsize(path)
        manifest[name] = {**meta, "kind": "wav", "bytes": size}
        print(f"  {name:38s} {size / 1e6:8.2f} MB  {meta['desc']}")

    print("生成素材 →", out_dir)

    # ---- WAV ------------------------------------------------------------
    n = 48000 * 3
    write_wav("tone_stereo_48k.wav",
              np.stack([tone(440, n, 48000, 0.8), tone(120, n, 48000, 0.5, sweep_to=6000)], axis=1),
              48000, "PCM_16",
              {"desc": "立体声 16bit：左 440Hz 定音，右 120→6000Hz 扫频",
               "rate": 48000, "channels": 2, "frames": n})

    chs = 8
    n8 = 48000 * 4
    data8 = np.stack([tone(100.0 * (i + 1), n8, 48000, 0.9 - 0.08 * i) for i in range(chs)], axis=1)
    data8[:, 3] = 0.0                      # 一个空通道，验证静音通道显示
    data8[400000:404000, 6] = 0.95         # 一处削波/爆音，验证红标
    write_wav("mic8_48k_24bit.wav", data8, 48000, "PCM_24",
              {"desc": "8 通道 24bit：每通道 100–800Hz 不同幅度，ch3 静音，ch6 含一段削波",
               "rate": 48000, "channels": chs, "frames": n8})

    write_wav("mono_16k_int16.wav", tone(1000, 16000 * 2, 16000)[:, None], 16000, "PCM_16",
              {"desc": "单声道 16kHz 16bit 1kHz（采样率与主文件不同，测跨采样率对齐）",
               "rate": 16000, "channels": 1, "frames": 16000 * 2})

    # ---- 裸 PCM ---------------------------------------------------------
    mono = tone(1000, 16000 * 3, 16000, 0.85, sweep_to=None)
    write("pcm_16k_mono_s16le.pcm", to_int16(mono),
          {"desc": "裸 PCM 16kHz 单声道 16bit 小端", "kind": "pcm",
           "rate": 16000, "channels": 1, "sample_width": 2, "is_float": False,
           "signed": True, "big_endian": False, "planar": False, "offset": 0,
           "bytes": mono.size * 2})

    quad = np.stack([tone(1000, 48000 * 4, 48000, 0.8), tone(2500, 48000 * 4, 48000, 0.6),
                     np.random.default_rng(7).uniform(-0.35, 0.35, 48000 * 4).astype(np.float32),
                     tone(150, 48000 * 4, 48000, 0.9)], axis=1)
    write("pcm_48k_4ch_s16le.pcm", to_int16(quad).reshape(-1),
          {"desc": "裸 PCM 48kHz 4 声道 16bit 小端交织（1k/2.5k/白噪/150Hz）", "kind": "pcm",
           "rate": 48000, "channels": 4, "sample_width": 2, "is_float": False,
           "signed": True, "big_endian": False, "planar": False, "offset": 0,
           "bytes": quad.size * 2})

    planar = to_int16(quad).transpose(1, 0).reshape(-1)
    write("pcm_48k_4ch_s16le_planar.pcm", planar,
          {"desc": "同上但平面（planar）排布，验证通道解交织的另一条通路", "kind": "pcm",
           "rate": 48000, "channels": 4, "sample_width": 2, "is_float": False,
           "signed": True, "big_endian": False, "planar": True, "offset": 0,
           "bytes": planar.nbytes})

    write("pcm_48k_stereo_f32.pcm", np.ascontiguousarray(quad[:, :2], dtype="<f4").reshape(-1),
          {"desc": "裸 PCM 48kHz 立体声 float32 小端", "kind": "pcm",
           "rate": 48000, "channels": 2, "sample_width": 4, "is_float": True,
           "signed": True, "big_endian": False, "planar": False, "offset": 0,
           "bytes": quad[:, :2].size * 4})

    write("pcm_48k_stereo_s16be.pcm", to_int16_be(quad[:, :2]).reshape(-1),
          {"desc": "裸 PCM 48kHz 立体声 16bit 大端（验证字节序）", "kind": "pcm",
           "rate": 48000, "channels": 2, "sample_width": 2, "is_float": False,
           "signed": True, "big_endian": True, "planar": False, "offset": 0,
           "bytes": quad[:, :2].size * 2})

    write("pcm_48k_mono_u8.pcm", to_uint8(tone(700, 48000 * 2, 48000, 0.9)),
          {"desc": "裸 PCM 48kHz 单声道 8bit 无符号（验证 uint8 零点在 128）", "kind": "pcm",
           "rate": 48000, "channels": 1, "sample_width": 1, "is_float": False,
           "signed": False, "big_endian": False, "planar": False, "offset": 0,
           "bytes": 48000 * 2})

    write("pcm_48k_mono_s24le.pcm", to_int24_le(tone(500, 48000 * 2, 48000, 0.9)),
          {"desc": "裸 PCM 48kHz 单声道 24bit packed 小端", "kind": "pcm",
           "rate": 48000, "channels": 1, "sample_width": 3, "is_float": False,
           "signed": True, "big_endian": False, "planar": False, "offset": 0,
           "bytes": 48000 * 2 * 3})

    head = b"TSVC" + np.int32(0).tobytes() + b"\x00" * 36      # 44 字节假头
    write("pcm_48k_mono_s16le_with44byteheader.pcm",
          np.frombuffer(head + to_int16(mono).tobytes(), dtype=np.uint8),
          {"desc": "带 44 字节假头的裸 PCM（验证「跳过文件头」）", "kind": "pcm",
           "rate": 16000, "channels": 1, "sample_width": 2, "is_float": False,
           "signed": True, "big_endian": False, "planar": False, "offset": len(head),
           "bytes": len(head) + mono.size * 2})

    # ---- 大文件（性能）--------------------------------------------------
    nbig = 48000 * 60
    big = np.stack([tone(80 + 60 * i, nbig, 48000, 0.5) for i in range(8)], axis=1)
    write("big8ch_60s_48k_s16le.pcm", to_int16(big).reshape(-1),
          {"desc": "8 通道 60 秒 16bit（≈46MB，验证 memmap + 分块抽稀不卡不炸内存）", "kind": "pcm",
           "rate": 48000, "channels": 8, "sample_width": 2, "is_float": False,
           "signed": True, "big_endian": False, "planar": False, "offset": 0,
           "bytes": big.size * 2})

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"\n共 {len(manifest)} 个文件；参数清单见 manifest.json")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    main(target)
