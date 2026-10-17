"""离屏 GUI 探针：真实构建主窗口、加载素材、驱动交互，抓 PNG 供人工/视觉核对。

用法：``QT_QPA_PLATFORM=offscreen .venv/bin/python tools/gui_probe.py [输出目录]``
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.core.audio_io import PcmFormat  # noqa: E402
from app.core.config import Config  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")


def pump(app, seconds: float = 0.4) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def spectro_peak_db(win) -> tuple[bool, str]:
    """谱图是否真的算出了能量（防「整张图静默全黑」这类不报错的错）。"""
    cur = getattr(win.spectro, "_current", None)
    if cur is None or getattr(cur, "img", None) is None or cur.img.isNull():
        return False, "没有图像"
    if cur.data is None:
        return False, "没有 dB 数据"
    peak = float(cur.data.max())
    return peak > -40.0, f"峰值 {peak:+.1f} dBFS，形状 {cur.data.shape}"


class _Vec:
    """QPoint 替身：只提供 y()/x()。"""

    def __init__(self, y: int = 0, x: int = 0) -> None:
        self._y, self._x = y, x

    def y(self) -> int:
        return self._y

    def x(self) -> int:
        return self._x


class _FakeEvent:
    """最小可用的鼠标/滚轮事件替身。"""

    def __init__(self, x: float, y: float, mods=Qt.NoModifier, delta: int = 0) -> None:
        self._p = QPointF(x, y)
        self._mods = mods
        self._delta = delta
        self.accepted = False

    def position(self) -> QPointF:
        return self._p

    def modifiers(self):
        return self._mods

    def angleDelta(self) -> _Vec:
        return _Vec(self._delta, 0)

    def pixelDelta(self) -> _Vec:
        return _Vec(0, 0)

    def accept(self) -> None:
        self.accepted = True


def _drag(widget, x0: float, x1: float) -> None:
    widget.mousePressEvent(_FakeEvent(x0, widget.plot_rect().center().y()))
    widget.mouseMoveEvent(_FakeEvent(x1, widget.plot_rect().center().y()))
    widget.mouseReleaseEvent(_FakeEvent(x1, widget.plot_rect().center().y()))


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "probe")
    os.makedirs(out_dir, exist_ok=True)
    app = QApplication(sys.argv)
    theme.apply_dark_theme(app)
    cfg = Config()
    win = MainWindow(cfg)
    win.resize(1440, 900)
    win.show()
    pump(app, 0.2)
    shot = 0

    def snap(tag: str) -> None:
        nonlocal shot
        shot += 1
        path = os.path.join(out_dir, f"{shot:02d}-{tag}.png")
        win.grab().save(path)
        print(f"  截图 {path}")

    problems: list[str] = []

    # ---- 1. 空界面 -----------------------------------------------------
    snap("empty")

    # ---- 2. 打开 8 通道 24bit WAV（解交织成 8 条道）--------------------
    src8 = win.project.open_file(os.path.join(SAMPLES, "mic8_48k_24bit.wav"))
    win.project.set_sample_rate(48000)
    win._after_open()
    pump(app, 0.8)
    snap("8ch-lanes")
    if len(src8.lanes) != 8:
        problems.append(f"8 通道 WAV 解交织出 {len(src8.lanes)} 条道")
    ok_spec, why = spectro_peak_db(win)
    if not ok_spec:
        problems.append(f"谱图未算出有效能量：{why}")
    else:
        print(f"  谱图（线性频率）：{why}")

    # ---- 3. 时间放大 + 幅值放大（⌘+滚轮 / ⌥+滚轮的等效调用）-----------
    win.view.set_window(2.0, 0.02)                # 20ms 窗口
    win.view.set_amp(4.0)
    pump(app, 0.8)
    snap("zoom-time-amp")
    if win.view.win > 0.0201:
        problems.append("时间放大未生效")

    # ---- 3b. 顶部全曲缩略图 -------------------------------------------
    ov = win.overview
    ov.request()
    pump(app, 1.0)
    if ov._img is None:
        problems.append(f"缩略图未生成全曲包络：{ov._error or '无错误信息'}")
    else:
        print(f"  缩略图全曲包络：{ov._img.width()}×{ov._img.height()} px")
    snap("overview-zoomed")
    rect = ov.plot_rect()
    dur = win.view.duration
    # 窗口框位置应与 ViewState 一致
    x0, x1 = ov.window_rect()
    exp0 = rect.x() + (win.view.t0 / dur) * rect.width()
    if abs(x0 - exp0) >= 1.0:
        problems.append(f"缩略图窗口框左边界 {x0:.1f} != 期望 {exp0:.1f}")
    # 拖框体 = 平移（窗口框只有几像素宽时也必须能拖住）
    t_before = win.view.t0
    _drag(ov, (x0 + x1) / 2, (x0 + x1) / 2 + 120)
    moved = win.view.t0 - t_before
    want = 120 / rect.width() * dur
    if abs(moved - want) > max(1e-4, want * 0.02):
        problems.append(f"窄框拖框平移量 {moved:.4f}s != 期望 {want:.4f}s")
    if abs(win.view.win - 0.02) > 1e-9:
        problems.append(f"拖框平移不应改窗口宽度，实际 {win.view.win}")
    # 拖右边 = 改宽度（在框足够宽的场景下），左边界不动
    win.view.set_window(2.0, 0.5)
    pump(app, 0.1)
    x0, x1 = ov.window_rect()
    if x1 - x0 < 2 * 6.5:
        problems.append(f"0.5s 窗口在缩略图上只有 {x1 - x0:.1f}px，测不了拖边")
    t0_fixed = win.view.t0
    _drag(ov, x1, x1 + 80)
    if abs(win.view.t0 - t0_fixed) > 1e-6:
        problems.append(f"拖右边时窗口起点漂了：{t0_fixed} → {win.view.t0}")
    want_win = 0.5 + 80 / rect.width() * dur
    if abs(win.view.win - want_win) > max(1e-4, want_win * 0.02):
        problems.append(f"拖右边后窗口宽度 {win.view.win:.4f}s != 期望 {want_win:.4f}s")
    # 拖左边 = 改宽度，右边界不动
    x0, x1 = ov.window_rect()
    t1_fixed = win.view.t0 + win.view.win
    _drag(ov, x0, x0 + 40)
    if abs((win.view.t0 + win.view.win) - t1_fixed) > 1e-6:
        problems.append(f"拖左边时窗口终点漂了：{t1_fixed} → {win.view.t0 + win.view.win}")
    snap("overview-resized")
    # 点空白 = 窗口中心跳过去
    win.view.set_window(2.0, 0.02)
    target_x = rect.x() + rect.width() * 0.75
    ov.mousePressEvent(_FakeEvent(target_x, rect.center().y()))
    ov.mouseReleaseEvent(_FakeEvent(target_x, rect.center().y()))
    center = win.view.t0 + win.view.win / 2
    want_center = 0.75 * dur
    if abs(center - want_center) > max(1e-3, want_center * 0.01):
        problems.append(f"点击跳转后窗口中心 {center:.3f}s != 期望 {want_center:.3f}s")
    # 滚轮 = 以光标为锚缩放（⌘+滚轮）
    win.view.set_window(2.0, 0.02)
    anchor_x = rect.x() + rect.width() * 0.4
    anchor_t = ov.x_to_seconds(anchor_x)
    ov.wheelEvent(_FakeEvent(anchor_x, rect.center().y(), Qt.ControlModifier, delta=120))
    if abs(ov.x_to_seconds(anchor_x) - anchor_t) > 1e-6:
        problems.append("缩略图滚轮缩放锚点漂移")
    if abs(win.view.win - 0.02) < 1e-9:
        problems.append("缩略图滚轮未改变窗口宽度")
    # 无修饰滚轮 = 滑动时间窗（不改宽度）
    win2 = win.view.win
    t2 = win.view.t0
    ov.wheelEvent(_FakeEvent(anchor_x, rect.center().y(), delta=120))
    if abs(win.view.win - win2) > 1e-9:
        problems.append("缩略图无修饰滚轮不该改窗口宽度（应只平移）")
    if abs(win.view.t0 - t2) < 1e-12:
        problems.append("缩略图无修饰滚轮未平移时间窗")
    # 双击 = 适配全曲
    ov.mouseDoubleClickEvent(_FakeEvent(anchor_x, rect.center().y()))
    if abs(win.view.win - dur) > 1e-6:
        problems.append(f"双击未适配全曲：{win.view.win} vs {dur}")
    win.view.set_window(2.0, 0.02)
    pump(app, 0.3)
    print(f"  缩略图交互：平移/改宽/跳转/锚点缩放/适配 全部校验")

    # ---- 4. 放大到样本级（折线模式）------------------------------------
    win.view.set_window(2.0, 0.0006)
    win.view.set_amp(1.0)
    win.cb_fill.setCurrentIndex(2)
    pump(app, 0.7)
    snap("per-sample-line")

    # ---- 5. dB 纵轴 + 叠加显示 -----------------------------------------
    win.cb_fill.setCurrentIndex(0)
    win.cb_yscale.setCurrentIndex(1)
    win.cb_mode.setCurrentIndex(1)
    win.view.set_window(0.0, 4.0)
    pump(app, 0.7)
    snap("overlay-db")
    win.cb_mode.setCurrentIndex(0)
    win.cb_yscale.setCurrentIndex(0)

    # ---- 6. 点击定位播放（模拟 seek）----------------------------------
    win.seek_frames(src8.frames // 2)
    pump(app, 0.4)
    if win.player.position != src8.frames // 2:
        problems.append(f"seek 后位置 {win.player.position} != {src8.frames // 2}")
    snap("seek-mid")

    # ---- 7. 裸 PCM（4 通道 16bit）-------------------------------------
    win.project.open_file(os.path.join(SAMPLES, "pcm_48k_4ch_s16le.pcm"),
                          PcmFormat(sample_rate=48000, channels=4, sample_width=2))
    win._after_open()
    win.view.fit_all()
    pump(app, 0.8)
    snap("pcm-4ch")
    if len(win.project.all_lanes()) != 12:
        problems.append(f"8+4 通道共应有 12 条道，实际 {len(win.project.all_lanes())}")

    # ---- 8. 合并通道：选中 8ch WAV + 4ch PCM --------------------------
    sids = [s.sid for s in win.project.file_sources() if s.kind == "file"]
    win.merge_sources(sids)
    pump(app, 0.8)
    snap("merged-12ch")
    mix = [s for s in win.project.file_sources() if s.kind == "mix"]
    if not mix or mix[0].channel_count != 12:
        problems.append(f"合并源通道数 {None if not mix else mix[0].channel_count} != 12")
    else:
        # 只播合并流：勾 mix、取消原文件
        for l in win.project.all_lanes():
            l.visible = False
        for l in mix[0].lanes[:8]:
            l.visible = True
        win._on_lanes_changed()
        pump(app, 0.6)
        snap("mix-play-lanes")
        chans, warn = win.player.prepare(win.project.visible_lanes(), 48000, None)
        if chans != 8:
            problems.append(f"合并流绑定到 {chans} 通道 != 8")

    # ---- 9. 大文件性能（46MB / 8ch / 60s）-----------------------------
    t0 = time.time()
    big = win.project.open_file(os.path.join(SAMPLES, "big8ch_60s_48k_s16le.pcm"),
                                PcmFormat(sample_rate=48000, channels=8, sample_width=2))
    win._after_open()
    win.view.fit_all()
    pump(app, 1.2)
    load = time.time() - t0
    snap("bigfile-fit-all")
    print(f"  大文件打开+全曲渲染：{load:.2f}s（含 1.2s 事件泵）")
    if big.frames != 48000 * 60:
        problems.append(f"大文件帧数 {big.frames}")

    # 只留大文件，滚轮缩放走一遍
    win.project.remove_sources([s.sid for s in win.project.file_sources() if s.sid != big.sid])
    win._after_open()
    win.view.fit_all()
    t0 = time.time()
    for _ in range(24):
        win.view.zoom_time(1.6)
    pump(app, 0.6)
    snap("bigfile-zoomed")
    print(f"  40× 连续放大到 {win.view.win * 1000:.3f} ms 渲染耗时 {time.time() - t0:.2f}s")

    # ---- 10. 滚动 + 播放跟随 -------------------------------------------
    win.view.set_window(30.0, 0.05)
    win._tick()
    pump(app, 0.5)
    snap("scroll-window")

    # ---- 11. 谱图切通道 / 换 FFT / 对数频率 ---------------------------
    idx = win.spectro_lane.findIndex(3) if hasattr(win.spectro_lane, "findIndex") else -1
    win.spectro_lane.setCurrentIndex(5 % win.spectro_lane.count())
    win.spectro_nfft.setCurrentIndex(win.spectro_nfft.findData(512))
    pump(app, 0.9)
    snap("spectro-nfft512")
    win.spectro_log.setChecked(True)
    win.view.fit_all()
    pump(app, 1.0)
    snap("spectro-logfreq")
    ok_spec, why = spectro_peak_db(win)
    print(f"  谱图（对数频率）：{why}")
    if not ok_spec:
        problems.append(f"对数频率谱图变空（频率轴重采样错）：{why}")
    win.spectro_log.setChecked(False)
    pump(app, 0.8)

    # ---- 12. 选区 → 循环 ----------------------------------------------
    win.view.set_selection(int(20 * 48000), int(25 * 48000))
    win.transport.btn_loop.click()
    pump(app, 0.4)
    if win.player.loop != (960000, 1200000):
        problems.append(f"循环区间 {win.player.loop}")
    snap("selection-loop")

    # ---- 13. 非交织 / 大端 / 8bit / 24bit 素材各开一个 ----------------
    for name, fmt in [
        ("pcm_48k_4ch_s16le_planar.pcm", PcmFormat(sample_rate=48000, channels=4, sample_width=2, planar=True)),
        ("pcm_48k_stereo_s16be.pcm", PcmFormat(sample_rate=48000, channels=2, sample_width=2, big_endian=True)),
        ("pcm_48k_mono_u8.pcm", PcmFormat(sample_rate=48000, channels=1, sample_width=1, signed=False)),
        ("pcm_48k_mono_s24le.pcm", PcmFormat(sample_rate=48000, channels=1, sample_width=3)),
        ("pcm_48k_mono_s16le_with44byteheader.pcm",
         PcmFormat(sample_rate=16000, channels=1, sample_width=2, offset=44)),
    ]:
        try:
            win.project.open_file(os.path.join(SAMPLES, name), fmt)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name} 打开失败：{exc}")
    win._after_open()
    win.view.fit_all()
    pump(app, 1.0)
    snap("all-formats-stacked")

    # ---- 14. 真实播放一轮（默认设备）---------------------------------
    win.project.remove_sources([s.sid for s in win.project.file_sources() if s.sid != big.sid])
    win._after_open()
    for l in win.project.all_lanes():
        l.visible = l.index < 2
    win._on_lanes_changed()
    win.view.set_window(0.0, 1.0)
    ok = win.player.play()
    if not ok:
        problems.append(f"真实播放失败：{win.player.error}")
    base = win.player.realtime_stats()
    pos0 = win.player.position
    wall0 = time.time()
    pump(app, 1.5)
    pos = win.player.position
    wall = time.time() - wall0
    stats = win.player.realtime_stats()
    d_frames = stats["frames"] - base["frames"]
    d_cbs = stats["callbacks"] - base["callbacks"]
    win._sync_playhead(pos)
    pump(app, 0.1)
    snap("playing")
    win.player.pause()
    if ok and pos < 12000:
        problems.append(f"播放 1.5s 只推进了 {pos} 帧（≈{pos / 48000:.2f}s）")
    played = (pos - pos0) / 48000
    print(f"  播放推进：Δ{pos - pos0:,} 帧 = {played:.2f}s 音频 / {wall:.2f}s 墙钟，"
          f"回调 {d_cbs} 次（块长 {stats['blocksizes']}），空回调 {stats['empty']}，underrun {stats['underruns']}")
    if d_cbs >= 5:
        fps = d_frames / max(wall, 1e-6)
        drift = abs(played - wall) / max(wall, 1e-6)
        if drift > 0.12:
            problems.append(f"播放速度偏离实时 {drift * 100:.0f}%"
                            f"（音频 {played:.2f}s vs 墙钟 {wall:.2f}s，等效 {fps:,.0f} 帧/秒）")
        if abs(fps - 48000) > 48000 * 0.12:
            problems.append(f"回调等效帧率 {fps:,.0f} 偏离 48000 超过 12%")

    # ---- 汇总 ---------------------------------------------------------
    print("\n" + "=" * 62)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print("  ✗", p)
    else:
        print("离屏渲染与交互全部通过，无异常")
    print(f"截图目录：{out_dir}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
