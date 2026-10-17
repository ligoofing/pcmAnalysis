"""DSP 层：波形抽稀包络、STFT 谱图、色标。"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 时域抽稀
# ---------------------------------------------------------------------------


def envelope(data: np.ndarray, columns: int, mode: str = "minmax") -> dict[str, np.ndarray]:
    """把 ``(n, ch)`` 抽稀成 ``(columns, ch)`` 的包络。

    ``mode``：``minmax`` 返回逐列 min/max（放大到样本级时退化为逐样本），
    ``rms`` 额外返回 RMS，``peak`` 返回绝对值峰值。
    """
    n, ch = data.shape
    columns = max(1, min(int(columns), max(n, 1)))
    if n == 0:
        z = np.zeros((columns, ch), dtype=np.float32)
        return {"min": z, "max": z.copy(), "rms": z.copy(), "step": columns}

    if n <= columns:
        # 放大到样本级：min/max 就是样本本身，右侧留白
        out = np.zeros((columns, ch), dtype=np.float32)
        out[:n] = data
        return {"min": out, "max": out.copy(), "rms": np.abs(out), "step": 1}

    # 用补零 + reshape 做无循环分块，避免 Python 层逐列计算
    block = int(np.ceil(n / columns))
    pad = block * columns - n
    padded = np.concatenate([data, np.zeros((pad, ch), dtype=data.dtype)], axis=0) if pad else data
    blocks = padded.reshape(columns, block, ch).astype(np.float32, copy=False)
    lo = blocks.min(axis=1)
    hi = blocks.max(axis=1)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    if mode == "peak":
        lo = -np.abs(blocks).max(axis=1)
        hi = -lo
    return {"min": lo, "max": hi, "rms": rms, "step": block}


def pixel_envelope(block: np.ndarray, cols: int, block_start: int,
                   want_start: int, want_end: int) -> dict[str, np.ndarray]:
    """按像素列精确抽稀，供波形绘制使用。

    ``block`` 覆盖全局帧区间 ``[block_start, block_start + len(block))``；
    返回的 ``min/max/rms`` 形状 ``(cols, ch)``，第 i 列对应
    ``[want_start + i*(want_end-want_start)/cols, ...]``。
    放大到样本级（每列不足 1 个样本）时退化为逐样本绘制。
    """
    cols = max(1, int(cols))
    n, ch = block.shape
    if n <= cols:
        out = np.zeros((cols, ch), dtype=np.float32)
        out[:n] = block
        return {"min": out, "max": out.copy(), "rms": np.abs(out), "per_sample": True}
    edges = np.linspace(want_start, want_end, cols + 1).astype(np.int64)
    edges = np.clip(edges - block_start, 0, n)
    starts = np.maximum.accumulate(edges[:-1])
    starts = np.minimum(starts, max(0, n - 1))
    counts = np.diff(edges)
    b = block.astype(np.float32, copy=False)
    mins = np.minimum.reduceat(b, starts, axis=0)
    maxs = np.maximum.reduceat(b, starts, axis=0)
    cnt = np.maximum(counts, 1).astype(np.float32)[:, None]
    rms = np.sqrt(np.add.reduceat(b * b, starts, axis=0) / cnt)
    return {"min": mins, "max": maxs, "rms": rms, "per_sample": False}


def stft_spectrogram_stream(read, f0: int, f1: int, cols: int, nfft: int = 2048,
                            hop: int | None = None, window: str = "hann",
                            column: int = 0) -> tuple[np.ndarray, int, int]:
    """分块流式谱图：内存与窗口长度无关，适合整首大文件。

    ``read(start, length) -> (length, ch)`` 通常是 ``Project.read``。
    返回 ``(db 矩阵 (cols, nfft//2+1), 实际 nfft, 实际 hop)``。
    """
    f0, f1 = int(f0), int(f1)
    cols = max(1, int(cols))
    nfft = next_pow2(nfft)
    span = max(1, f1 - f0)
    if hop is None:
        hop = max(1, span // cols)
    hop = max(1, int(hop))
    nfft = min(nfft, max(8, span))
    nfft = next_pow2(nfft)
    if hop > nfft:
        # 帧间留空隙会在谱图上打出竖条纹；自动抬到能覆盖 hop 的窗长
        nfft = next_pow2(min(max(8, span), hop))
    win = WINDOWS.get(window, np.hanning)(nfft).astype(np.float32)
    gain = float(win.sum()) or 1.0
    bins = nfft // 2 + 1
    out = np.empty((cols, bins), dtype=np.float32)
    per_chunk = max(1, min(cols, int(2 * (1 << 20) // max(1, nfft))))   # 每次约 2MB 复数中间量
    base_idx = np.arange(nfft)[None, :]
    for a in range(0, cols, per_chunk):
        b = min(cols, a + per_chunk)
        s0 = f0 + a * hop
        need = (b - 1 - a) * hop + nfft
        seg = np.asarray(read(s0, need)[:, column], dtype=np.float32) if need > 0 else np.zeros(0, np.float32)
        if seg.shape[0] < need:
            seg = np.concatenate([seg, np.zeros(need - seg.shape[0], np.float32)])
        idx = np.arange(b - a)[:, None] * hop + base_idx
        frames = seg[idx] * win[None, :]
        mag = np.abs(np.fft.rfft(frames, n=nfft, axis=1)) * (2.0 / gain)
        out[a:b] = mag
    db = 20.0 * np.log10(np.maximum(out, 1e-12))
    return np.clip(db, -140.0, 12.0), nfft, hop


def resample_freq_rows(db: np.ndarray, rows: int, log_scale: bool,
                       fmax_hz: float = 0.0, fmin_hz: float = 20.0) -> np.ndarray:
    """把谱图频率轴重采样到显示行数；``log_scale=True`` 时按对数频率展开。

    ``db`` 形状 ``(时间列, 频率 bin)``，bin 线性覆盖 0..fmax_hz。
    返回 ``(cols, rows)``，行 0 = 底部（低频），行 ``rows-1`` = 顶部（高频）。
    """
    cols, bins = db.shape
    y = np.linspace(0.0, 1.0, rows)
    if not log_scale or not fmax_hz:
        src = y * (bins - 1)
    else:
        # 第 0 行落在 fmin_hz，最后一行落在 fmax_hz：bin = f/fmax*(bins-1)
        lo = float(np.clip(fmin_hz / fmax_hz, 1e-6, 0.5))
        src = lo * (1.0 / lo) ** y * (bins - 1)
    src = np.clip(src, 0, bins - 1)
    lo_i = np.floor(src).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, bins - 1)
    w = (src - lo_i).astype(np.float32)[None, :]
    out = db[:, lo_i] * (1.0 - w) + db[:, hi_i] * w
    return out.astype(np.float32)


def signal_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    """逐通道统计：峰值 / RMS / dBFS / 直流偏置 / 削波样本数。"""
    if data.size == 0:
        ch = data.shape[1] if data.ndim > 1 else 1
        z = np.zeros(ch, dtype=np.float64)
        return {"peak": z, "rms": z, "peak_db": z - 120, "rms_db": z - 120,
                "dc": z, "clip": z.astype(np.int64)}
    d = data.astype(np.float64)
    peak = np.abs(d).max(axis=0)
    rms = np.sqrt(np.mean(d * d, axis=0))
    dc = d.mean(axis=0)
    clip = (np.abs(d) >= 0.999).sum(axis=0)
    return {
        "peak": peak,
        "rms": rms,
        "peak_db": 20 * np.log10(np.maximum(peak, 1e-12)),
        "rms_db": 20 * np.log10(np.maximum(rms, 1e-12)),
        "dc": dc,
        "clip": clip.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# 谱图
# ---------------------------------------------------------------------------

WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "rect": lambda n: np.ones(n),
}


def stft_spectrogram(
    data: np.ndarray,
    nfft: int = 2048,
    hop: int | None = None,
    window: str = "hann",
    channel: int = 0,
    chunk_cols: int = 512,
) -> np.ndarray:
    """单通道谱图，返回 ``(time_cols, freq_bins)`` 的 dB 幅度（floor 截断到 -120dB）。

    频率轴为 0..fs/2 的 ``nfft//2+1`` 根谱线。
    """
    x = data[:, channel] if data.ndim > 1 and data.shape[1] > channel else np.asarray(data).ravel()
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    hop = hop or max(1, nfft // 4)
    win = WINDOWS.get(window, np.hanning)(nfft).astype(np.float32)
    if n < nfft:
        x = np.concatenate([x, np.zeros(nfft - n, np.float32)])
        n = x.shape[0]
    cols = max(1, 1 + (n - nfft) // hop)
    # 从 buffer 头部取 nfft 帧：用补零 + 步长切片构造帧索引（对 memmap 友好）
    padded = np.concatenate([x, np.zeros(nfft, np.float32)])
    gain = float(win.sum()) or 1.0
    out = np.empty((cols, nfft // 2 + 1), dtype=np.float32)
    for a in range(0, cols, chunk_cols):
        b = min(cols, a + chunk_cols)
        idx = np.arange(b - a)[:, None] * hop + np.arange(nfft)[None, :]
        frames = padded[idx] * win[None, :]
        spec = np.fft.rfft(frames, n=nfft, axis=1)
        # 相干增益归一化：满幅正弦的谱峰 = 1.0 → 0 dBFS
        mag = np.abs(spec) * (2.0 / gain)
        out[a:b] = mag
    db = 20.0 * np.log10(np.maximum(out, 1e-12))
    return np.clip(db, -120.0, 12.0)


def next_pow2(x: int) -> int:
    return int(2 ** max(0, int(np.ceil(np.log2(max(1, x))))))


def freq_ticks(fmax: float, count: int = 5) -> list[float]:
    """对数/线性频率刻度（取整齐值）。"""
    if fmax <= 0:
        return []
    targets = [20, 30, 40, 50, 60, 70, 80, 100, 200, 300, 400, 500, 600, 700, 800, 1000,
               2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000, 20000, 24000, 32000, 48000, 64000, 96000]
    step = fmax / max(1, count)
    out = []
    for i in range(1, count + 1):
        v = step * i
        near = min(targets, key=lambda t: abs(t - v))
        if near <= fmax and near not in out:
            out.append(float(near))
    return out


def time_ticks(t0: float, t1: float, count: int = 6) -> list[float]:
    """整齐的时间刻度。"""
    span = t1 - t0
    if span <= 0:
        return [t0]
    raw = span / max(1, count)
    pow10 = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * pow10
        if step >= raw:
            break
    start = np.ceil(t0 / step) * step
    vals = np.arange(start, t1 + step * 0.5, step)
    return [float(v) for v in vals]


# ---------------------------------------------------------------------------
# 色标
# ---------------------------------------------------------------------------

_COLORMAP_STOPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "viridis": [
        (0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)), (1.0, (253, 231, 37)),
    ],
    "grayscale": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "inferno": [
        (0.0, (0, 0, 4)), (0.25, (87, 16, 110)), (0.5, (188, 55, 84)),
        (0.75, (249, 142, 9)), (1.0, (252, 255, 164)),
    ],
    "turbo": [
        (0.0, (48, 18, 59)), (0.2, (60, 157, 252)), (0.4, (23, 229, 192)),
        (0.6, (255, 218, 39)), (0.8, (229, 85, 27)), (1.0, (122, 4, 3)),
    ],
}


def colormap(name: str) -> np.ndarray:
    """返回 256x3 的 uint8 查找表。"""
    stops = _COLORMAP_STOPS.get(name, _COLORMAP_STOPS["viridis"])
    pos = np.array([p for p, _ in stops], dtype=np.float64)
    rgb = np.array([c for _, c in stops], dtype=np.float64)
    x = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(x, pos, rgb[:, ch]), 0, 255).astype(np.uint8)
    return lut


def colorize(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """``values`` 归一化到 0..1 后查表，返回 ``(H, W, 3)`` uint8。"""
    idx = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    return lut[idx]


def db_to_unit(db: np.ndarray, floor_db: float, top_db: float) -> np.ndarray:
    span = max(1e-6, top_db - floor_db)
    return np.clip((db - floor_db) / span, 0.0, 1.0)


def format_time(seconds: float, with_ms: bool = True) -> str:
    if seconds is None or not np.isfinite(seconds):
        seconds = 0.0
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if with_ms:
        return f"{sign}{h:d}:{m:02d}:{int(s):02d}.{int((s - int(s)) * 1000):03d}"
    return f"{sign}{h:d}:{m:02d}:{int(s):02d}"


def format_tick_time(seconds: float, step: float) -> str:
    """按刻度间隔决定时间标签的精度与单位。"""
    if step >= 60:
        return format_time(seconds, with_ms=False)
    if step >= 1:
        return f"{seconds:.1f}s"
    if step >= 0.01:
        return f"{seconds:.3f}s"
    if step >= 1e-4:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds * 1e6:.1f}µs"


def format_freq(hz: float) -> str:
    if hz >= 1000:
        v = hz / 1000.0
        return f"{v:.1f}k".replace(".0k", "k")
    return f"{hz:.0f}Hz"
