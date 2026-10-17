"""无头自检：不启动界面，逐层验证解码 / DSP / 模型 / 视图数学 / 播放位置。

运行：``.venv/bin/python -m tests.selftest``   （退出码非 0 表示有失败项）
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import dsp
from app.core.audio_io import PcmFormat, load_audio, load_pcm, load_wav
from app.core.model import Project
from app.core.player import Player
from app.ui.view_state import ViewState

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def runner(*a, **kw):
            try:
                msg = fn(*a, **kw)
                RESULTS.append((name, True, msg or ""))
                print(f"  ✅ {name}" + (f"  — {msg}" if msg else ""))
            except AssertionError as exc:
                RESULTS.append((name, False, str(exc)))
                print(f"  ❌ {name}  — {exc}")
            except Exception as exc:  # noqa: BLE001
                import traceback
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"  ❌ {name}  — {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return runner
        return runner
    return deco


def rel_err(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    denom = max(1e-9, float(np.abs(b).max()))
    return float(np.abs(a - b).max() / denom)


# ---------------------------------------------------------------------------
# 1. 解码
# ---------------------------------------------------------------------------


@check("裸 PCM 各位深/字节序/排布解码与真值一致")
def t_decode() -> str:
    ref = np.sin(2 * np.pi * 500 * np.arange(4096) / 48000).astype(np.float32) * 0.8
    tmp = tempfile.mkdtemp()
    worst = 0.0
    cases = []
    try:
        i16 = np.round(ref * 32767).astype("<i2")
        cases.append(("s16le", PcmFormat(channels=1, sample_width=2), i16.tobytes(), i16 / 32768.0))
        i16be = np.round(ref * 32767).astype(">i2")
        cases.append(("s16be", PcmFormat(channels=1, sample_width=2, big_endian=True), i16be.tobytes(),
                      i16be.astype("<i2") / 32768.0))
        i32 = np.round(ref * (2 ** 31 - 1)).astype("<i4")
        cases.append(("s32le", PcmFormat(channels=1, sample_width=4), i32.tobytes(), i32 / 2 ** 31))
        u8 = np.round(ref * 127 + 128).astype(np.uint8)
        cases.append(("u8", PcmFormat(channels=1, sample_width=1, signed=False), u8.tobytes(),
                      (u8.astype(np.int32) - 128) / 128.0))
        f32 = ref.astype("<f4")
        cases.append(("f32le", PcmFormat(channels=1, sample_width=4, is_float=True), f32.tobytes(), f32))
        i24 = np.round(ref * ((1 << 23) - 1)).astype(np.int32)
        packed = np.stack([(i24 >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.uint8).reshape(-1)
        cases.append(("s24le", PcmFormat(channels=1, sample_width=3), packed.tobytes(), i24 / float(1 << 23)))

        for tag, fmt, blob, expect in cases:
            path = os.path.join(tmp, f"{tag}.pcm")
            with open(path, "wb") as fh:
                fh.write(blob)
            buf = load_pcm(path, fmt)
            got = buf.read(0, len(expect))[:, 0]
            err = rel_err(got, expect)
            worst = max(worst, err)
            assert err < 2e-3, f"{tag} 解码误差 {err:.2e}（应 <2e-3，即 1 LSB 量化）"

        # 交织 vs 平面
        quad = np.stack([ref, ref * 0.5, -ref, ref * 0.25], axis=1)
        inter = np.round(quad * 32767).astype("<i2").reshape(-1)
        p = os.path.join(tmp, "inter.pcm")
        inter.tofile(p)
        buf = load_pcm(p, PcmFormat(channels=4, sample_width=2))
        assert buf.frames == 4096, f"交织帧数 {buf.frames} != 4096"
        got = buf.read(0, 4096)
        assert rel_err(got, np.round(quad * 32767) / 32768.0) < 2e-3, "交织 4 通道读错"

        pl = os.path.join(tmp, "planar.pcm")
        np.round(quad * 32767).astype("<i2").T.reshape(-1).tofile(pl)
        gotp = load_pcm(pl, PcmFormat(channels=4, sample_width=2, planar=True)).read(0, 4096)
        assert rel_err(gotp, np.round(quad * 32767) / 32768.0) < 2e-3, "平面（planar）排布解交织错"

        # 文件头偏移
        hp = os.path.join(tmp, "head.pcm")
        with open(hp, "wb") as fh:
            fh.write(b"\x00" * 44 + i16.tobytes())
        got_h = load_pcm(hp, PcmFormat(sample_width=2, offset=44)).read(0, 4096)[:, 0]
        assert rel_err(got_h, i16 / 32768.0) < 2e-3, "跳过 44 字节头后数据不对"
        assert load_pcm(hp, PcmFormat(sample_width=2, offset=44)).frames == 4096, "偏移后帧数算错"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"7 种格式全部通过，最大相对误差 {worst:.2e}"


@check("WAV 多通道装载与 soundfile 结果一致")
def t_wav() -> str:
    path = os.path.join(SAMPLES, "mic8_48k_24bit.wav")
    assert os.path.exists(path), "缺少 samples/mic8_48k_24bit.wav，先跑 tools/make_samples.py"
    buf = load_wav(path)
    assert (buf.frames, buf.channels, buf.sample_rate) == (192000, 8, 48000), \
        f"元信息错 {(buf.frames, buf.channels, buf.sample_rate)}"
    import soundfile as sf
    ref, _ = sf.read(path, dtype="float32", always_2d=True)
    got = buf.read(0, 1000)
    assert rel_err(got, ref[:1000]) < 1e-4, "WAV 采样值与 soundfile 不符"
    # 每通道频率不同：ch0=100Hz，ch7=800Hz
    peaks = []
    for c in (0, 7):
        one = buf.read(0, 16384, [c])[:, 0]
        spec = np.abs(np.fft.rfft(one * np.hanning(16384)))
        peaks.append(int(spec.argmax()) * 48000 / 16384)
    assert abs(peaks[0] - 100) < 8 and abs(peaks[1] - 800) < 12, f"通道内容错位：{peaks}"
    assert buf.read(0, 100, [3]).max() == 0.0, "ch3 应为静音"
    return f"8ch/24bit 正常；ch0={peaks[0]:.0f}Hz ch7={peaks[1]:.0f}Hz，ch3 静音"


@check("越界读取补零 / 尾部半帧不崩")
def t_bounds() -> str:
    path = os.path.join(SAMPLES, "pcm_16k_mono_s16le.pcm")
    buf = load_pcm(path, PcmFormat(sample_rate=16000, sample_width=2, channels=1))
    tail = buf.read(buf.frames - 10, 100)
    assert tail.shape == (100, 1) and np.all(tail[10:] == 0), "越界未补零"
    assert np.any(tail[:10] != 0), "有效区间读成 0"
    assert buf.read(buf.frames, 5).max() == 0, "起点越界应全 0"
    return "补零行为正确"


# ---------------------------------------------------------------------------
# 2. DSP
# ---------------------------------------------------------------------------


@check("按像素抽稀：min/max 精确到列，放大到样本级自动退化")
def t_envelope() -> str:
    n = 1000
    x = np.linspace(-1, 1, n, dtype=np.float32)[:, None] * np.array([1.0, 0.5], np.float32)
    env = dsp.pixel_envelope(x, 10, 0, 0, n)
    assert env["min"].shape == (10, 2) and not env["per_sample"], "形状错"
    for i in range(10):
        seg = x[i * 100:(i + 1) * 100]
        assert abs(env["min"][i, 0] - seg[:, 0].min()) < 1e-6, f"第 {i} 列 min 错"
        assert abs(env["max"][i, 0] - seg[:, 0].max()) < 1e-6, f"第 {i} 列 max 错"
    rms = env["rms"][:, 0]
    assert np.allclose(rms, np.sqrt((x[:, 0] ** 2).reshape(10, 100).mean(axis=1)), atol=1e-5), "RMS 错"
    zoom = dsp.pixel_envelope(x[:5], 20, 0, 0, 5)
    assert zoom["per_sample"] and zoom["min"].shape == (20, 2) and zoom["max"][3, 0] == x[3, 0], \
        "样本级未退化或值错"
    return "10 列包络与逐段 min/max 全等"


@check("谱图幅度校准：满幅正弦 ≈ 0 dBFS、频率落在正确谱线")
def t_spectro_calib() -> str:
    rate, nfft, dur = 48000, 4096, 1.0
    freq = 85 * rate / nfft                    # 第 85 条谱线，整周期无泄漏
    t = np.arange(int(rate * dur)) / rate
    amp = 10 ** (-3 / 20)                      # -3 dBFS
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[:, None]
    db, used_nfft, hop = dsp.stft_spectrogram_stream(lambda s, n: x[s:s + n], 0, x.shape[0], 300,
                                                     nfft=nfft, window="blackman")
    assert db.shape[1] == used_nfft // 2 + 1, "谱线数与 nfft 不符"
    col = db[150]
    peak_db = col.max()
    peak_hz = int(col.argmax()) * rate / used_nfft
    assert abs(peak_hz - freq) < 1, f"峰频 {peak_hz:.1f}Hz 偏离 {freq:.1f}Hz"
    assert abs(peak_db - (-3.0)) < 0.05, f"整周期峰值 {peak_db:.2f}dB 偏离 -3dBFS（归一化错）"
    assert np.all(db[:, :5] < peak_db - 20), "直流没被压住"
    # 半格偏离（非整周期）时允许窗函数扇贝损失
    x2 = (amp * np.sin(2 * np.pi * (freq + 0.5 * rate / nfft) * t)).astype(np.float32)[:, None]
    db2 = dsp.stft_spectrogram_stream(lambda s_, n_: x2[s_:s_ + n_], 0, x2.shape[0], 300,
                                      nfft=nfft, window="blackman")[0]
    loss = -3.0 - db2.max()
    # Blackman 窗最大扇贝损失约 1.1dB，是窗函数固有特性
    assert 0.8 <= loss <= 1.3, f"半格扇贝损失 {loss:.2f}dB 偏离 Blackman 标称 1.1dB"
    return f"整周期峰 {peak_db:+.2f}dB（期望 -3.00），半格损失 {loss:.2f}dB（窗函数特性）"


@check("跨采样率窗口：16k 源在 48k 工程里时间轴对齐")
def t_resample_align() -> str:
    proj = Project()
    a = proj.open_file(os.path.join(SAMPLES, "tone_stereo_48k.wav"))
    b = proj.open_file(os.path.join(SAMPLES, "mono_16k_int16.wav"))
    proj.set_sample_rate(48000)
    lane_b = b.lanes[0]
    # 16k 源第 1 秒 → 48k 工程第 48000 帧，应读到与源第 16000 帧同一时刻
    at_1s = proj.read([lane_b], 48000, 9, 48000)[:, 0]      # 9 个 48k 帧 = 3 个 16k 样本
    src_1s = b.buffer.read(16000, 3, [0])[:, 0]
    # 48k 每 3 帧对应 16k 的 1 个样本
    for k in range(3):
        assert abs(at_1s[k * 3] - src_1s[k]) < 1e-5, f"第 {k} 个源样本未对齐：{at_1s[k*3]} vs {src_1s[k]}"
    assert abs(at_1s[0] - src_1s[0]) < 1e-6
    assert a.channel_count == 2 and b.channel_count == 1
    return "16k→48k 帧位置换算一致"


@check("统计量：峰值 / RMS / 直流 / 削波计数")
def t_stats() -> str:
    x = np.ones((1000, 1), np.float32) * 0.5
    s = dsp.signal_stats(x)
    assert abs(s["peak"][0] - 0.5) < 1e-6 and abs(s["rms"][0] - 0.5) < 1e-6
    assert abs(s["dc"][0] - 0.5) < 1e-6 and s["clip"][0] == 0
    y = np.full((10, 1), 0.9995, np.float32)
    assert dsp.signal_stats(y)["clip"][0] == 10, "削波未计数"
    assert abs(dsp.signal_stats(x)["rms_db"][0] - (-6.0206)) < 0.01
    return "0.5 直流 → -6.02dBFS、削波计数正常"


@check("对数频率轴重采样：谱峰行号与 log(f) 映射吻合")
def t_logfreq() -> str:
    rows, bins, fmax = 400, 1025, 24000.0
    bin_hz = fmax / (bins - 1)

    def make(hz: float) -> np.ndarray:
        db = np.full((6, bins), -120.0, np.float32)
        c = int(round(hz / bin_hz))
        # 真实谱峰有窗主瓣，按 ±2 bin 铺开，避免「单 bin 尖峰被行数抽稀错过」这种非 bug 情况
        for k in range(max(0, c - 2), min(bins, c + 3)):
            db[:, k] = -6.0 - 3.0 * abs(k - c)
        return db

    for hz in (300.0, 1000.0, 3000.0, 12000.0):
        db = make(hz)
        r_lin = int(np.argmax(dsp.resample_freq_rows(db, rows, False, fmax)[0]))
        r_log = int(np.argmax(dsp.resample_freq_rows(db, rows, True, fmax, fmin_hz=20.0)[0]))
        f_lin = r_lin / (rows - 1) * fmax
        f_log = 20.0 * (fmax / 20.0) ** (r_log / (rows - 1))
        assert abs(f_lin - hz) < bin_hz * 4, f"线性轴 {hz}Hz → {f_lin:.0f}Hz"
        assert abs(f_log - hz) < max(30.0, hz * 0.06), f"对数轴 {hz}Hz → {f_log:.0f}Hz"
        assert 0 <= r_log < rows and 0 <= r_lin < rows, "行号越界"
    # 对数映射不退化：频率单调 → 行号单调，且低频被明显拉开
    seq = [100.0, 300.0, 1000.0, 3000.0, 12000.0]
    r_logs = [int(np.argmax(dsp.resample_freq_rows(make(h), rows, True, fmax)[0])) for h in seq]
    assert all(b > a for a, b in zip(r_logs, r_logs[1:])), f"对数轴行号非单调：{dict(zip(seq, r_logs))}"
    assert abs(r_logs[1] - r_logs[0] - (r_logs[3] - r_logs[2])) <= max(3, 0.2 * (r_logs[3] - r_logs[2])), \
        f"对数轴每十倍频程行数不等：100→300 占 {r_logs[1] - r_logs[0]} 行，1k→3k 占 {r_logs[3] - r_logs[2]} 行"
    r_lin = int(np.argmax(dsp.resample_freq_rows(make(380.0), rows, False, fmax)[0]))
    r_log = int(np.argmax(dsp.resample_freq_rows(make(380.0), rows, True, fmax)[0]))
    assert r_log > r_lin * 2, f"对数轴未拉开低频：380Hz {r_lin} → {r_log}"
    return "300/1k/3k/12kHz 两轴均落在正确行；对数轴单调且低频展开"


@check("长窗口下自动抬升 nfft：帧间不留未分析的空隙")
def t_spectro_hop() -> str:
    """hop > nfft 意味着大部分音频根本没被分析，短促瞬态会被整段跳过。"""
    rate, secs, cols = 48000, 60, 1000
    x = (0.5 * np.sin(2 * np.pi * 380 * np.arange(rate * secs) / rate)).astype(np.float32)[:, None]
    db, nfft, hop = dsp.stft_spectrogram_stream(lambda s, n: x[s:s + n], 0, x.shape[0], cols,
                                                nfft=512, window="hann")
    assert nfft >= hop, f"hop={hop} 仍大于 nfft={nfft}，存在未分析的帧间空隙"
    peak_bin = int(np.argmax(db.mean(axis=0)))
    peak = peak_bin * rate / nfft
    assert abs(peak - 380) < 25, f"抬升 nfft 后峰频漂到 {peak:.0f}Hz"
    band = db[:, max(0, peak_bin - 2):peak_bin + 3].max(axis=1)
    assert float(band.max() - band.min()) < 2.0, \
        f"等幅正弦的列间波动 {band.max() - band.min():.2f}dB 过大"
    # 瞬态可检出：在某一帧内插入 2ms 爆音，远离基频的高频段底噪必须被抬起
    x2 = x.copy()
    click_at = 30 * rate
    x2[click_at:click_at + int(0.002 * rate), 0] = 0.99
    db2 = dsp.stft_spectrogram_stream(lambda s, n: x2[s:s + n], 0, x2.shape[0], cols, nfft=512)[0]
    far = int(db.shape[1] * 0.6)          # 380Hz 只占前 1.6% 的谱线，取 60% 以外
    floor_before, floor_after = float(db[:, far:].max()), float(db2[:, far:].max())
    assert floor_after > floor_before + 6.0, \
        f"2ms 爆音未抬高高频段底噪（{floor_before:.1f} → {floor_after:.1f}dB），瞬态被帧间空隙吃掉"
    return (f"nfft 由 512 抬到 {nfft}（hop={hop}），峰 {peak:.0f}Hz，列间波动 "
            f"{float(band.max() - band.min()):.2f}dB，2ms 爆音 {floor_before:.0f}→{floor_after:.0f}dB 可检出")


@check("缩略图命中区判定：窄框必须可平移，不被边缘吸附吞掉")
def t_hit_zone() -> str:
    from app.ui.overview_bar import EDGE_GRAB, hit_zone
    # 宽框：边缘 / 内部 / 外部
    assert hit_zone(100, 100, 500) == "left" and hit_zone(104, 100, 500) == "left"
    assert hit_zone(500, 100, 500) == "right" and hit_zone(496, 100, 500) == "right"
    assert hit_zone(300, 100, 500) == "inside"
    assert hit_zone(93, 100, 500) == "outside" and hit_zone(560, 100, 500) == "outside"
    # 放大到窗口框只有 5.4px：整框必须算「可平移」，否则用户拖不动
    narrow = (100.0, 105.4)
    for x in (100.0, 102.0, 105.0, 105.4):
        assert hit_zone(x, *narrow) == "inside", f"窄框 x={x} 被判成 {hit_zone(x, *narrow)}，无法平移"
    assert hit_zone(90, *narrow) == "outside" and hit_zone(120, *narrow) == "outside"
    # 退化框（零宽）不崩
    assert hit_zone(100, 100, 100) == "inside" and hit_zone(120, 100, 100) == "outside"
    # 恰好 2×grab 的临界：仍按窄框处理，不出现两条边互相抢占
    w = 2 * EDGE_GRAB
    assert all(hit_zone(100 + k, 100, 100 + w) == "inside" for k in (0, 3, 6, 9, 12))
    return "宽框边缘/内部/外部正确；≤2×grab 的窄框整段可拖"


@check("分道布局函数：主图与缩略图同源、通道带互不重叠")
def t_lane_bands() -> str:
    from app.ui.view_state import lane_bands
    for n in (1, 2, 8, 12, 24):
        bands = lane_bands(n, 300.0, "lanes")
        assert len(bands) == n, f"{n} 道返回 {len(bands)} 条"
        tops = [cy - half for cy, half in bands]
        bots = [cy + half for cy, half in bands]
        assert all(b <= t + 1e-9 for b, t in zip(bots, tops[1:])), f"{n} 道上下重叠"
        assert all(h > 0 for _, h in bands), "出现零高度通道带"
        assert tops[0] >= 0 and bots[-1] <= 300.0, "通道带越出画布"
    ov = lane_bands(8, 75.0, "lanes")
    main = lane_bands(8, 75.0, "lanes")
    assert ov == main, "同一入参两次调用结果不一致"
    flat = lane_bands(4, 200.0, "overlay")
    assert len({c for c, _ in flat}) == 1, "叠加模式应共用同一中心线"
    assert lane_bands(0, 100.0, "lanes") == [], "0 通道未返回空"
    return "1/2/8/12/24 道均不重叠且不越界，叠加模式同中心"


@check("色标 LUT 与 dB 归一化单调可用")
def t_colormap() -> str:
    for name in ("viridis", "inferno", "turbo", "grayscale"):
        lut = dsp.colormap(name)
        assert lut.shape == (256, 3) and lut.dtype == np.uint8, f"{name} LUT 形状错"
    unit = dsp.db_to_unit(np.array([-90.0, -45.0, 0.0]), -90.0, 0.0)
    assert np.allclose(unit, [0.0, 0.5, 1.0]), f"归一化错 {unit}"
    rgb = dsp.colorize(unit[:, None], dsp.colormap("viridis"))
    assert rgb.shape == (3, 1, 3) and rgb[2, 0, 0] > rgb[0, 0, 0], "着色方向反了"
    return "4 个色标 + dB 归一化正常"


# ---------------------------------------------------------------------------
# 3. 模型：合并 / 解交织 / 导出
# ---------------------------------------------------------------------------


@check("多文件合并通道 → 新多声道源，数据与堆叠结果一致")
def t_merge() -> str:
    proj = Project()
    a = proj.open_file(os.path.join(SAMPLES, "tone_stereo_48k.wav"))
    b = proj.open_file(os.path.join(SAMPLES, "mic8_48k_24bit.wav"))
    proj.set_sample_rate(48000)
    lanes = a.lanes + b.lanes[:2]
    mix = proj.create_mix("MergeTest", lanes)
    assert mix.kind == "mix" and mix.channel_count == 4, f"合并源通道数 {mix.channel_count}"
    stacked = proj.read(lanes, 0, 500, 48000)
    mixed = proj.read(mix.lanes, 0, 500, 48000)
    assert rel_err(stacked, mixed) < 1e-6, "合并流与堆叠读取结果不一致"
    assert mix.frames == max(a.frames, b.frames), "合并源长度应取最长源"
    assert all(not l.visible for l in lanes), "并入后原通道应自动取消勾选"
    # 导出后回读
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "mix.wav")
        frames, chans = proj.export_lanes(out, mix.lanes, 48000, "PCM_16")
        assert chans == 4 and frames == mix.frames, f"导出尺寸错 {frames}x{chans}"
        back = load_wav(out)
        assert back.channels == 4, "回读通道数错"
        assert rel_err(back.read(0, 1000), proj.read(mix.lanes, 0, 1000, 48000)) < 0.01, "导出回读值不符（含 16bit 量化）"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"4 通道合并 + 导出回读一致（{frames:,} 帧）"


@check("单文件解交织导出：每个通道写成独立单声道 WAV")
def t_deinterleave() -> str:
    proj = Project()
    src = proj.open_file(os.path.join(SAMPLES, "pcm_48k_4ch_s16le.pcm"),
                         PcmFormat(sample_rate=48000, channels=4, sample_width=2))
    proj.set_sample_rate(48000)
    tmp = tempfile.mkdtemp()
    try:
        files = proj.export_deinterleaved(tmp, src.sid, "PCM_24")
        assert len(files) == 4, f"应写 4 个文件，写了 {len(files)}"
        for i, f in enumerate(files):
            mono = load_wav(f)
            assert mono.channels == 1, "解交织后不是单声道"
            ref = src.buffer.read(0, 2000, [i])[:, 0]
            got = mono.read(0, 2000)[:, 0]
            assert rel_err(got, ref) < 0.01, f"CH{i} 内容不符"
        return "4 个单声道文件内容与原通道逐一吻合"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("移除源后合并源自动收缩，不残留悬空通道")
def t_remove_cascade() -> str:
    proj = Project()
    a = proj.open_file(os.path.join(SAMPLES, "tone_stereo_48k.wav"))
    b = proj.open_file(os.path.join(SAMPLES, "mono_16k_int16.wav"))
    mix = proj.create_mix("M", a.lanes + b.lanes)
    assert mix.channel_count == 3
    proj.remove_sources([b.sid])
    assert mix.channel_count == 2, f"应剩 2 通道，实际 {mix.channel_count}"
    assert proj.lane(mix.lanes[0].key) is not None
    proj.remove_sources([a.sid])
    assert proj.is_empty(), "全部移除后仍有残留源"
    return "级联清理正确"


@check("裸 PCM 导出：写盘后按同样参数读回一致")
def t_export_pcm() -> str:
    proj = Project()
    src = proj.open_file(os.path.join(SAMPLES, "tone_stereo_48k.wav"))
    proj.set_sample_rate(48000)
    tmp = tempfile.mkdtemp()
    try:
        for fmt in (PcmFormat(sample_rate=48000, channels=2, sample_width=2),
                    PcmFormat(sample_rate=48000, channels=2, sample_width=4),
                    PcmFormat(sample_rate=48000, channels=2, sample_width=3),
                    PcmFormat(sample_rate=48000, channels=2, sample_width=1, signed=False),
                    PcmFormat(sample_rate=48000, channels=2, sample_width=2, big_endian=True)):
            path = os.path.join(tmp, f"out_{fmt.sample_width}_{int(fmt.is_float)}.pcm")
            nbytes = proj.export_pcm(path, src.lanes, fmt)
            assert nbytes == os.path.getsize(path), "导出字节数与文件不符"
            back = load_pcm(path, fmt)
            assert back.channels == 2 and back.frames == src.frames, f"{fmt.kind} 尺寸错"
            ref = proj.read(src.lanes, 0, 5000, 48000)
            got = back.read(0, 5000)
            tol = {1: 0.02, 2: 2e-4, 3: 2e-6, 4: 1e-6}[fmt.sample_width]
            assert rel_err(got, ref) < tol, f"{fmt.describe()} 往返误差 {rel_err(got, ref):.2e} > {tol}"
        return "16/32/24/8bit/大端 5 种往返一致"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. 视图数学
# ---------------------------------------------------------------------------


@check("ViewState：坐标换算 / 缩放锚点 / 边界钳制")
def t_view_state() -> str:
    v = ViewState()
    v.set_total(48000 * 10, 48000)
    assert abs(v.duration - 10.0) < 1e-9, "时长算错"
    v.set_window(2.0, 0.5)
    rect_w, rect_x = 800.0, 0.0
    for x in (0.0, 123.0, 400.0, 799.0):
        f = v.x_to_frame(x, rect_x, rect_w)
        back = v.seconds_to_x(f / v.rate, rect_x, rect_w)
        assert abs(back - x) < 0.02, f"帧↔像素往返误差 {back - x}"
    # 缩放锚点：光标处时间点不变
    anchor_x = 300.0
    t_before = v.x_to_seconds(anchor_x, rect_x, rect_w)
    v.zoom_time(4.0, (anchor_x - rect_x) / rect_w)
    t_after = v.x_to_seconds(anchor_x, rect_x, rect_w)
    assert abs(t_before - t_after) < 1e-6, f"锚点漂移 {t_after - t_before}"
    assert abs(v.win - 0.125) < 1e-9, "时间放大倍数错"
    # 边界钳制
    v.set_window(9.9, 1.0)
    assert v.t0 + v.win <= 10.0 + 1e-9 and v.t0 >= 0, "窗口越界未钳制"
    v.scroll_seconds(+1e6)
    assert abs(v.t0 - (v.duration - v.win)) < 1e-6, "右端未钳到末尾"
    v.scroll_seconds(-1e6)
    assert v.t0 == 0.0, "左端未钳到 0"
    v.set_amp(1e9)
    assert v.amp == 4096.0, "幅值未钳制"
    v.zoom_amp(0.5)
    assert abs(v.amp - 2048.0) < 1e-6
    # 适配与选区
    v.fit_all()
    assert abs(v.win - 10.0) < 1e-9, "适配全曲失败"
    v.set_selection(4800, 2400)
    assert v.has_selection and v.sel_start == 2400, "选区未排序"
    assert v.zoom_to_selection() and abs(v.win - 0.05) < 1e-9, "缩放到选区失败"
    changes = []
    v.view_changed.connect(lambda r: changes.append(r))
    v.set_window(v.t0, v.win)
    assert not changes, "无变化却发信号"
    return "像素↔帧往返 <0.02px，锚点零漂移，边界/信号均正确"


@check("ViewState：跨采样率变更后窗口保持")
def t_view_rate() -> str:
    v = ViewState()
    v.set_total(480000, 48000)
    v.set_window(4.0, 0.2)
    v.set_total(480000, 16000, keep_window=True)
    assert v.duration > 0 and v.win <= v.duration, "换采样率后窗口越界"
    v.set_total(0, 48000)
    assert v.win > 0, "空工程时窗口被清零"
    return "换采样率/清空都不炸"


# ---------------------------------------------------------------------------
# 5. 播放引擎（不开真实音频流）
# ---------------------------------------------------------------------------


class _FakeOut:
    """模拟 PortAudio 的 outdata 缓冲。"""

    def __init__(self, frames: int, chans: int) -> None:
        self.buf = np.zeros((frames, chans), np.float32)

    def __setitem__(self, key, value):
        self.buf[key] = value

    def __getitem__(self, key):
        return self.buf[key]

    def __len__(self) -> int:
        return self.buf.shape[0]


@check("Player：环形缓冲取数、位置推进、跳转与循环折返")
def t_player_math() -> str:
    proj = Project()
    src = proj.open_file(os.path.join(SAMPLES, "tone_stereo_48k.wav"))
    proj.set_sample_rate(48000)
    p = Player(proj)
    chans, warn = p.prepare(src.lanes, 48000, None)
    assert chans == 2 and warn is None, f"prepare 结果 {chans}/{warn}"
    p.total_frames = 4800 * 2          # 缩短，方便测试
    p._playing = True
    p.blocksize = 256

    # 预取 1024 帧后连续消费，逐块比对送出的采样
    data = proj.read(src.lanes, 0, 1024, 48000)
    with p._lock:
        p._ring[:1024] = data
        p._avail = 1024
        p._wpos = 1024
    consumed = []
    for _ in range(4):
        out = _FakeOut(256, 2)
        p._callback(out.buf, 256, None, None)
        consumed.append(out.buf.copy())
    got = np.concatenate(consumed, axis=0)
    assert np.array_equal(got, data), "环形缓冲输出与预取数据不一致"
    assert p.position == 1024, f"位置应为 1024，实际 {p.position}"
    assert p._avail == 0, "缓冲未清空"
    empty_out = _FakeOut(256, 2)
    p._callback(empty_out.buf, 256, None, None)
    assert np.allclose(empty_out.buf, 0) and p.position == 1024, "缓冲耗尽后应输出静音且位置不再前进"
    p._playing = False

    # 跳转
    p._do_seek(2000)
    assert p.position == 2000 and p._fill_pos == 2000 and p._avail == 0, "跳转后状态未复位"

    # 循环：越过 b 时播放指针折回 a
    p.set_loop((1000, 1500))
    assert p.loop == (1000, 1500), "循环区间设置失败"
    p._do_seek(1400)
    with p._lock:
        p._ring[:512] = proj.read(src.lanes, 1400, 512, 48000)
        p._avail = 512
        p._rpos = 0
        p._wpos = 512
        p._fill_pos = 1912
    p._playing = True
    out = _FakeOut(256, 2)
    for _ in range(3):
        out.buf[:] = 0
        p._callback(out.buf, 256, None, None)
    p._playing = False
    assert 1000 <= p.position < 1500, f"越过后未折回循环区：{p.position}"
    assert np.isfinite(p.position_seconds), "位置时间换算出现 NaN"

    # 无通道/空源时的错误路径
    empty = Player(Project())
    assert empty.play() is False and empty.error, "空工程应拒绝播放并给出原因"
    return "1024 帧顺序消费、跳转、循环折返与错误路径均正确"


@check("Player：采样率不同的源混播时按时间轴换算")
def t_player_rates() -> str:
    proj = Project()
    a = proj.open_file(os.path.join(SAMPLES, "tone_stereo_48k.wav"))
    b = proj.open_file(os.path.join(SAMPLES, "mono_16k_int16.wav"))
    proj.set_sample_rate(48000)
    lanes = a.lanes + b.lanes
    mat = proj.read(lanes, 0, 480, 48000)
    assert mat.shape == (480, 3) and np.isfinite(mat).all(), "混播矩阵形状/数值异常"
    proj.set_sample_rate(16000)
    mat16 = proj.read(lanes, 0, 160, 16000)
    assert mat16.shape == (160, 3), "16k 工程下形状错"
    # 同一时刻（第 0.01s）48k 与 16k 工程的 16k 源通道应基本一致
    at48 = proj.read(b.lanes, int(0.01 * 48000), 9, 48000)[:, 0]
    at16 = proj.read(b.lanes, int(0.01 * 16000), 3, 16000)[:, 0]
    assert rel_err(at48[::3], at16) < 1e-4, "48k 每 3 帧应与 16k 每 1 帧对齐"
    return "3 通道混播矩阵正常，时刻对齐 <2% 误差"


# ---------------------------------------------------------------------------
# 6. 配置
# ---------------------------------------------------------------------------


@check("配置：PCM 参数历史去重与预设读写")
def t_config() -> str:
    from app.core.config import Config
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "config.json")
        c = Config(__import__("pathlib").Path(path))
        f16 = PcmFormat(sample_rate=48000, channels=4, sample_width=2).to_dict()
        c.remember_pcm_format(f16)
        c.remember_pcm_format(f16)
        assert len(c["pcm_history"]) == 1, f"历史未去重：{len(c['pcm_history'])}"
        c.remember_pcm_format(PcmFormat(sample_rate=16000, channels=1, sample_width=2).to_dict())
        assert len(c["pcm_history"]) == 2 and c["pcm_history"][0]["sample_rate"] == 16000, "最新参数未置顶"
        c.add_preset(f16, "48k 4ch 16bit")
        c.add_preset(f16, "48k 4ch 16bit")
        assert len(c["pcm_presets"]) == 1, "预设未去重"
        c2 = Config(__import__("pathlib").Path(path))
        assert c2["pcm_presets"][0]["channels"] == 4, "重读配置丢失预设"
        assert PcmFormat.from_dict(c2["pcm_history"][0]).sample_rate == 16000
        c2.remove_preset("48k 4ch 16bit")
        assert not Config(__import__("pathlib").Path(path))["pcm_presets"], "删除预设未落盘"
        return "去重/置顶/落盘重读/删除 全部通过"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("素材清单：每个样例按 manifest 参数都能解析出正确时长")
def t_manifest() -> str:
    path = os.path.join(SAMPLES, "manifest.json")
    assert os.path.exists(path), "缺少 manifest.json"
    man = json.loads(open(path, encoding="utf-8").read())
    bad = []
    for name, meta in man.items():
        full = os.path.join(SAMPLES, name)
        if not os.path.exists(full):
            bad.append(f"{name} 文件缺失")
            continue
        buf = load_audio(full, PcmFormat(
            sample_rate=meta["rate"], channels=meta["channels"], sample_width=meta["sample_width"],
            is_float=meta["is_float"], signed=meta["signed"], big_endian=meta["big_endian"],
            offset=meta["offset"], planar=meta["planar"]) if meta["kind"] == "pcm" else None)
        if meta["kind"] == "pcm":
            expect_frames = (meta["bytes"] - meta["offset"]) // (meta["channels"] * meta["sample_width"])
            if buf.frames != expect_frames:
                bad.append(f"{name}: 帧数 {buf.frames} != {expect_frames}")
            if buf.channels != meta["channels"]:
                bad.append(f"{name}: 通道数 {buf.channels} != {meta['channels']}")
            if abs(buf.duration - expect_frames / meta["rate"]) > 1e-6:
                bad.append(f"{name}: 时长算错")
    if bad:
        raise AssertionError("；".join(bad))
    return f"{len(man)} 个素材解析参数全部吻合"


# ---------------------------------------------------------------------------


def main() -> int:
    if not os.path.isdir(SAMPLES):
        print("samples/ 不存在，先生成素材：.venv/bin/python tools/make_samples.py")
        return 2
    print("\n== 解码 ==")
    t_decode(); t_wav(); t_bounds()
    print("\n== DSP ==")
    t_envelope(); t_spectro_calib(); t_spectro_hop(); t_resample_align(); t_stats(); t_logfreq(); t_colormap()
    print("\n== 模型（合并/解交织/导出）==")
    t_merge(); t_deinterleave(); t_remove_cascade(); t_export_pcm()
    print("\n== 视图数学 ==")
    t_view_state(); t_view_rate(); t_hit_zone(); t_lane_bands()
    print("\n== 播放引擎 ==")
    t_player_math(); t_player_rates()
    print("\n== 配置与素材 ==")
    t_config(); t_manifest()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{'=' * 60}\n通过 {len(RESULTS) - len(failed)}/{len(RESULTS)}")
    for name, _, msg in failed:
        print(f"  失败：{name} → {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
