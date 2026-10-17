"""顶部全曲缩略图：分道画出整段音频包络，用亮框标出当前时间窗。

与主波形图共用 ``ViewState`` 与 ``lane_bands`` 布局，所以两者的分道结构逐像素同构。
全曲包络需要整段读盘（大文件可达几十 MB），因此放在后台线程算，主线程只画缓存 +
窗口框/选区/播放指针这些便宜的叠加层。

交互：拖框体=平移窗口；拖框体左右边=改窗口宽度（时间缩放）；点空白=窗口中心跳到那里；
滚轮=以光标为锚缩放；双击=适配全曲。
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core import dsp
from . import theme
from .view_state import MIN_WIN_SEC, lane_bands

CHUNK_SAMPLES = 1 << 18
HEADER_H = 14
EDGE_GRAB = 6.0
BAR_HEIGHT = 92


def hit_zone(x: float, rx0: float, rx1: float, grab: float = EDGE_GRAB) -> str:
    """缩略图上的命中区：左边缘 / 右边缘 / 框内 / 框外。

    放大很多时窗口框可能只有几个像素宽——若仍按固定半径吸附边缘，整框都会被判成
    「拖边」，平移就永远抓不到。所以框宽不足 2×grab 时整框一律算「框内」，
    改宽度交给 ⌘+滚轮 或主图上的时间缩放。
    """
    if rx1 - rx0 <= 2 * grab:
        return "inside" if rx0 - 1.5 <= x <= rx1 + 1.5 else "outside"
    if abs(x - rx0) <= grab:
        return "left"
    if abs(x - rx1) <= grab:
        return "right"
    if rx0 < x < rx1:
        return "inside"
    return "outside"


class _Bridge(QObject):
    done = Signal(object)


class _Job:
    __slots__ = ("seq", "keys", "total", "rate", "w", "h", "img", "error")

    def __init__(self, seq, keys, total, rate, w, h):
        self.seq, self.keys, self.total, self.rate, self.w, self.h = seq, keys, total, rate, w, h
        self.img = None
        self.error = ""


class OverviewBar(QWidget):
    interacted = Signal(str)

    def __init__(self, view, project, parent=None) -> None:
        super().__init__(parent)
        self.view = view
        self.project = project
        self.playhead = 0.0
        self._img: QImage | None = None
        self._img_keys: tuple = ()
        self._bridge = _Bridge()
        self._bridge.done.connect(self._on_done)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._quit = threading.Event()
        self._pending: _Job | None = None
        self._seq = 0
        self._applied = -1
        self._thread: threading.Thread | None = None
        self._error = ""
        self._drag_mode: str | None = None      # pan | left | right | jump
        self._drag_x0 = 0.0
        self._drag_t0 = 0.0
        self._drag_t1 = 0.0
        self.setFixedHeight(BAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        view.view_changed.connect(lambda *_: self.update())
        view.selection_changed.connect(lambda *_: self.update())

    # ------------------------------------------------------------------
    # 几何
    # ------------------------------------------------------------------
    def plot_rect(self) -> QRectF:
        r = self.rect()
        return QRectF(4, HEADER_H, max(10.0, r.width() - 8), max(10.0, r.height() - HEADER_H - 3))

    def x_to_seconds(self, x: float) -> float:
        rect = self.plot_rect()
        dur = self.view.duration
        frac = (x - rect.x()) / max(1.0, rect.width())
        return float(np.clip(frac, 0.0, 1.0)) * dur

    def seconds_to_x(self, t: float) -> float:
        rect = self.plot_rect()
        dur = self.view.duration
        if dur <= 0:
            return rect.x()
        return rect.x() + float(np.clip(t / dur, 0.0, 1.0)) * rect.width()

    def window_rect(self) -> tuple[float, float]:
        v = self.view
        return self.seconds_to_x(v.t0), self.seconds_to_x(v.t0 + v.win)

    # ------------------------------------------------------------------
    # 生命周期 / 计算
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._quit.clear()
        self._thread = threading.Thread(target=self._worker, name="overview-worker", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._quit.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def request(self) -> None:
        """通道集合/长度/尺寸变化时重算全曲包络。"""
        if self.project.is_empty() or self.view.total_frames <= 0:
            self._img = None
            self._img_keys = ()
            self.update()
            return
        rect = self.plot_rect()
        w, h = max(1, int(rect.width())), max(1, int(rect.height()))
        lanes = self.project.visible_lanes()
        keys = tuple(l.key for l in lanes)
        if not lanes:
            self._img = None
            self._img_keys = ()
            self.update()
            return
        if self._img is not None and self._img_keys == (keys, w, h, self.view.total_frames, self.view.rate):
            return
        with self._lock:
            self._seq += 1
            self._pending = _Job(self._seq, keys, self.view.total_frames, self.view.rate, w, h)
        self._wake.set()

    def _worker(self) -> None:  # pragma: no cover - 后台线程
        while not self._quit.is_set():
            self._wake.wait()
            if self._quit.is_set():
                return
            with self._lock:
                job, self._pending = self._pending, None
            self._wake.clear()
            if job is None:
                continue
            try:
                job.img = self._render(job)
            except Exception as exc:  # noqa: BLE001
                job.img, job.error = None, f"{type(exc).__name__}: {exc}"
            self._bridge.done.emit(job)

    def _render(self, job: _Job) -> QImage:
        lanes = self.project.lanes(job.keys)      # 与 job.keys 同序（均来自 visible_lanes 顺序）
        w, h = job.w, job.h
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor("#0e1014"))
        p = QPainter(img)
        bands = lane_bands(len(lanes), h, "lanes")
        total = job.total
        spp = total / w
        chunk_cols = max(1, min(w, int(CHUNK_SAMPLES / max(1.0, spp))))
        for c0 in range(0, w, chunk_cols):
            ncols = min(chunk_cols, w - c0)
            s0 = int(c0 * spp)
            s1 = min(total, int((c0 + ncols) * spp))
            s1 = max(s1, s0 + 1)
            block = self.project.read(lanes, s0, s1 - s0, job.rate)
            env = dsp.pixel_envelope(block, ncols, s0, s0, s1)
            xs = c0 + np.arange(ncols)
            for i, lane in enumerate(lanes):
                if i >= env["max"].shape[1]:
                    continue
                cy, half = bands[i]
                hi = np.clip(env["max"][:, i], -1.05, 1.05)
                lo = np.clip(env["min"][:, i], -1.05, 1.05)
                y_up = cy - hi * half * 0.94
                y_dn = cy - lo * half * 0.94
                color = QColor(lane.color)
                color.setAlpha(225)
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                top = np.minimum(y_up, y_dn)
                bot = np.maximum(y_up, y_dn)
                for k in range(ncols):
                    p.drawRect(int(xs[k]), int(top[k]), 1, max(1, int(bot[k] - top[k])))
        # 道分隔与零线（点线会在窄带上造成噪点，用低对比实线）
        pen = QPen(QColor(255, 255, 255, 14), 1)
        p.setPen(pen)
        for cy, half in bands:
            p.drawLine(QPointF(0, cy), QPointF(w, cy))
        p.end()
        return img

    def _on_done(self, job: _Job) -> None:
        if job.seq <= self._applied:
            return
        self._applied = job.seq
        if job.img is not None and not job.img.isNull():
            self._img = job.img
            self._img_keys = (job.keys, job.w, job.h, job.total, job.rate)
        self._error = job.error or ""
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        rect = self.plot_rect()
        p.fillRect(self.rect(), QColor(theme.PANEL))
        p.fillRect(rect, QColor("#0e1014"))
        v = self.view
        # 文字行
        font = QFont(p.font())
        font.setPointSizeF(9.0)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_DIM))
        if self.project.is_empty():
            p.drawText(QRectF(4, 0, rect.width(), HEADER_H), Qt.AlignLeft | Qt.AlignVCenter,
                       "全曲缩略图（打开文件后显示）")
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5))
            p.end()
            return
        win_ms = v.win * 1000
        pct0 = v.t0 / max(v.duration, 1e-9) * 100
        pct1 = (v.t0 + v.win) / max(v.duration, 1e-9) * 100
        zoomed = v.win < v.duration * 0.999
        label = (f"全曲 {v.duration:,.3f}s · {len(self.project.visible_lanes())} 道"
                 + (f"   窗口 {win_ms:,.2f}ms @{v.t0:,.3f}s  ({pct0:.2f}%–{pct1:.2f}%)" if zoomed
                    else "   已适配全曲"))
        bx0, bx1 = self.window_rect()
        if bx1 - bx0 <= 2 * EDGE_GRAB:
            # 框太窄时拖边不可用，明确告诉用户改宽度走哪里
            label += "   框过窄：拖框=平移，⌘+滚轮改宽度"
        p.drawText(QRectF(4, 0, rect.width() - 8, HEADER_H), Qt.AlignLeft | Qt.AlignVCenter, label)
        if self._img is None:
            p.setPen(QColor("#ff8a4c" if self._error else theme.TEXT_DIM))
            p.drawText(rect, Qt.AlignCenter, self._error or "全曲包络生成中…")
            if not self._error:
                self.request()
        else:
            p.drawImage(rect.topLeft(), self._img)

        # 选区
        if v.has_selection:
            sx0, sx1 = sorted((self.seconds_to_x(v.sel_start / v.rate), self.seconds_to_x(v.sel_end / v.rate)))
            p.fillRect(QRectF(sx0, rect.top(), max(1.0, sx1 - sx0), rect.height()),
                       QColor(79, 179, 255, 40))
        # 窗口外的压暗 + 窗口框：压暗要克制，否则放大很多时整条全曲包络都看不见
        x0, x1 = self.window_rect()
        dim = QColor(8, 9, 12, 96)
        p.fillRect(QRectF(rect.left(), rect.top(), max(0.0, x0 - rect.left()), rect.height()), dim)
        p.fillRect(QRectF(x1, rect.top(), max(0.0, rect.right() - x1), rect.height()), dim)
        p.setPen(QPen(QColor(theme.ACCENT), 1.4))
        p.setBrush(Qt.NoBrush)
        box = QRectF(x0, rect.top() + 0.5, max(1.0, x1 - x0), rect.height() - 1.0)
        p.drawRect(box)
        # 窗口内顶部/底部各画一条细亮线，框极窄时也能看出位置
        p.setPen(QPen(QColor(79, 179, 255, 120), 1))
        p.drawLine(QPointF(x0, rect.top() + 1.5), QPointF(x1, rect.top() + 1.5))
        p.drawLine(QPointF(x0, rect.bottom() - 1.5), QPointF(x1, rect.bottom() - 1.5))
        # 左右把手
        p.setPen(QPen(QColor(theme.ACCENT), 2.0))
        for hx in (x0, x1):
            p.drawLine(QPointF(hx, rect.top() + 2), QPointF(hx, rect.bottom() - 2))
        # 播放指针
        px = self.seconds_to_x(self.playhead)
        p.setPen(QPen(QColor(theme.PLAYHEAD), 1.2))
        p.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5))
        p.end()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = float(seconds)

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.view.duration <= 0:
            return
        rect = self.plot_rect()
        x = float(event.position().x())
        if not (rect.left() - 2 <= x <= rect.right() + 2):
            return
        x0, x1 = self.window_rect()
        zone = hit_zone(x, x0, x1)
        t = self.x_to_seconds(x)
        v = self.view
        if zone == "left":
            self._drag_mode = "left"
        elif zone == "right":
            self._drag_mode = "right"
        elif zone == "inside":
            self._drag_mode = "pan"
        else:
            self._drag_mode = "jump"
            v.center_on_seconds(t)
        self._drag_x0 = x
        self._drag_t0, self._drag_t1 = v.t0, v.t0 + v.win
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        rect = self.plot_rect()
        x = float(event.position().x())
        x0, x1 = self.window_rect()
        if self._drag_mode is None:
            zone = hit_zone(x, x0, x1)
            self.setCursor(Qt.SizeHorCursor if zone in ("left", "right") else Qt.PointingHandCursor)
            return
        v = self.view
        dt = (x - self._drag_x0) / max(1.0, rect.width()) * v.duration
        if self._drag_mode == "pan":
            v.set_window(self._drag_t0 + dt, v.win)
        elif self._drag_mode == "jump":
            v.center_on_seconds(self.x_to_seconds(x))
        elif self._drag_mode == "right":
            new_t1 = np.clip(self.x_to_seconds(x), self._drag_t0 + MIN_WIN_SEC, v.duration)
            v.set_window(self._drag_t0, float(new_t1 - self._drag_t0))
        elif self._drag_mode == "left":
            new_t0 = np.clip(self.x_to_seconds(x), 0.0, self._drag_t1 - MIN_WIN_SEC)
            v.set_window(float(new_t0), float(self._drag_t1 - new_t0))
        self.interacted.emit(self._drag_mode)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_mode = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.view.fit_all()
        self.interacted.emit("fit")
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802
        """与主图保持同一套约定：滚轮=滑窗，⌘/Ctrl=时间缩放，⌥/Shift=幅值缩放。"""
        rect = self.plot_rect()
        ad = event.angleDelta()
        unit = ad.y() / 120.0 if abs(ad.y()) >= 8 else ad.y() / 40.0
        anchor = (event.position().x() - rect.x()) / max(1.0, rect.width())
        mods = event.modifiers()
        if mods & (Qt.ControlModifier | Qt.MetaModifier):
            self.view.zoom_time(1.18 ** (unit * 3.0), anchor)
        elif mods & (Qt.AltModifier | Qt.ShiftModifier):
            self.view.zoom_amp(1.18 ** (unit * 3.0))
        else:
            self.view.scroll_seconds(-unit * self.view.win * 0.5)
        self.interacted.emit("wheel")
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.request()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_0:
            self.view.fit_all()
        else:
            super().keyPressEvent(event)
