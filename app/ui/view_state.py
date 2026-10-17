"""共享视图状态：时间窗口、幅值放大、选区。时域图与谱图共用同一实例。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

MIN_WIN_SEC = 1e-4        # 最小时间窗（100µs）
MAX_AMP = 4096.0
MIN_AMP = 0.01


def lane_bands(n: int, height: float, mode: str = "lanes", gap: float = 4.0,
               pad: float = 2.0) -> list[tuple[float, float]]:
    """把高度切成 n 条通道带，返回 ``(中心 y, 半高)``。

    主波形图与顶部缩略图共用，保证两者的分道布局逐像素同构。
    """
    if n <= 0:
        return []
    if mode == "overlay":
        return [(height / 2.0, max(4.0, height / 2.0 - pad * 2))] * n
    g = gap if n > 1 else 0.0
    lane_h = (height - g * (n - 1)) / n
    return [(lane_h / 2.0 + i * (lane_h + g), max(3.0, lane_h / 2.0 - pad)) for i in range(n)]


class ViewState(QObject):
    """时间窗 + 幅值缩放，变更时发 ``view_changed(reason)``。"""

    view_changed = Signal(str)          # reason: "time" | "amp" | "range"
    selection_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.total_frames = 0
        self.rate = 48000
        self.t0 = 0.0                   # 窗口起点（秒）
        self.win = 1.0                  # 窗口时长（秒）
        self.amp = 1.0                  # 幅值放大：显示范围 ±1/amp
        self.sel_start = 0              # 选区（帧），sel 为空时两者相等
        self.sel_end = 0

    # ------------------------------------------------------------------
    def set_total(self, total_frames: int, rate: int, keep_window: bool = False) -> None:
        old_dur = self.duration
        self.total_frames = max(0, int(total_frames))
        self.rate = int(rate) or 48000
        dur = self.duration
        if not keep_window or dur <= 0:
            self.fit_all()
            return
        self.t0 = min(self.t0, max(0.0, dur - self.win))
        if old_dur != dur:
            self.view_changed.emit("range")

    @property
    def duration(self) -> float:
        return self.total_frames / self.rate if self.rate else 0.0

    @property
    def samples_in_window(self) -> float:
        return self.win * self.rate

    def fit_all(self) -> None:
        self.t0 = 0.0
        self.win = max(self.duration, MIN_WIN_SEC)
        self.view_changed.emit("time")

    # ------------------------------------------------------------------
    def set_window(self, t0: float, win: float, reason: str = "time") -> None:
        dur = self.duration
        win = float(np.clip(win, MIN_WIN_SEC, max(MIN_WIN_SEC, dur or MIN_WIN_SEC)))
        t0 = float(np.clip(t0, 0.0, max(0.0, dur - win))) if dur else 0.0
        changed = abs(t0 - self.t0) > 1e-12 or abs(win - self.win) > 1e-12
        self.t0, self.win = t0, win
        if changed:
            self.view_changed.emit(reason)

    def scroll_seconds(self, delta: float) -> None:
        dur = self.duration
        if dur <= 0:
            return
        # 滚动量与当前窗口成比例，缩得越小滚得越细
        step = delta * max(self.win * 0.25, 1.0 / self.rate)
        span = dur - self.win
        self.set_window(min(max(0.0, self.t0 + step), max(0.0, span)), self.win)

    def scroll_frames(self, delta_frames: float) -> None:
        self.scroll_seconds(delta_frames / self.rate if self.rate else 0.0)

    def zoom_time(self, factor: float, anchor: float | None = None) -> None:
        """以 ``anchor``（窗口内 0..1 比例，默认窗口中心）为锚点缩放时间轴。"""
        anchor = 0.5 if anchor is None else float(np.clip(anchor, 0.0, 1.0))
        new_win = float(np.clip(self.win / factor, MIN_WIN_SEC, max(MIN_WIN_SEC, self.duration or MIN_WIN_SEC)))
        anchor_t = self.t0 + anchor * self.win
        self.set_window(anchor_t - anchor * new_win, new_win)

    def set_amp(self, amp: float) -> None:
        amp = float(np.clip(amp, MIN_AMP, MAX_AMP))
        if abs(amp - self.amp) < 1e-9:
            return
        self.amp = amp
        self.view_changed.emit("amp")

    def zoom_amp(self, factor: float) -> None:
        self.set_amp(self.amp * factor)

    def center_on_seconds(self, t: float) -> None:
        self.set_window(t - self.win / 2.0, self.win)

    def ensure_visible(self, t: float) -> None:
        if t < self.t0:
            self.set_window(t, self.win)
        elif t > self.t0 + self.win:
            self.set_window(t - self.win * 0.95, self.win)

    # ------------------------------------------------------------------
    # 坐标换算
    # ------------------------------------------------------------------
    def x_to_seconds(self, x: float, plot_x: float, plot_w: float) -> float:
        if plot_w <= 0:
            return self.t0
        return self.t0 + (x - plot_x) / plot_w * self.win

    def seconds_to_x(self, t: float, plot_x: float, plot_w: float) -> float:
        if self.win <= 0:
            return plot_x
        return plot_x + (t - self.t0) / self.win * plot_w

    def x_to_frame(self, x: float, plot_x: float, plot_w: float) -> int:
        return int(round(self.x_to_seconds(x, plot_x, plot_w) * self.rate))

    def frame_range(self, plot_x: float, plot_w: float) -> tuple[int, int]:
        f0 = int(np.floor(self.t0 * self.rate))
        f1 = int(np.ceil((self.t0 + self.win) * self.rate))
        return max(0, f0), min(self.total_frames, max(f0, f1))

    # ------------------------------------------------------------------
    # 选区
    # ------------------------------------------------------------------
    def set_selection(self, a_frames: int, b_frames: int) -> None:
        a, b = sorted((int(a_frames), int(b_frames)))
        if a == self.sel_start and b == self.sel_end:
            return
        self.sel_start, self.sel_end = a, b
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        self.set_selection(0, 0)

    @property
    def has_selection(self) -> bool:
        return self.sel_end - self.sel_start > 1

    def zoom_to_selection(self) -> bool:
        if not self.has_selection:
            return False
        a = self.sel_start / self.rate
        b = self.sel_end / self.rate
        self.set_window(a, max(MIN_WIN_SEC, b - a))
        return True
