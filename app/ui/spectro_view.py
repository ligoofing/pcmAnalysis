"""谱图视图：STFT 时频二维热力图，异步线程计算，与波形共用时间窗口。

* 计算放在后台线程，主线程只做绘制；窗口快速变化时用 QTimer 合并请求（防抖）。
* 频率轴支持线性/对数；显示行数按控件高度重采样，行数少时不做无谓计算。
* 鼠标悬停读出「时间 · 频率 · 幅度 dB」。
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

from ..core import dsp
from . import theme
from .time_view_base import AXIS_L, TimeAxisView

FLOOR_MIN = -140.0


class _Bridge(QObject):
    result = Signal(object)


class _Task:
    __slots__ = ("req_id", "lane_key", "f0", "f1", "cols", "rows", "nfft", "hop",
                 "window", "floor", "top", "log", "col_index", "rate", "data", "img", "error")

    def __init__(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)
        self.data = None
        self.img = None


class SpectroView(TimeAxisView):
    def __init__(self, view, project, parent=None) -> None:
        super().__init__(view, project, parent)
        self.opts = {
            "nfft": 2048,
            "window": "hann",
            "colormap": "viridis",
            "floor_db": -90.0,
            "top_db": 0.0,
            "freq_log": False,
        }
        self.lane_key: str | None = None
        self.playhead = 0.0
        self._bridge = _Bridge()
        self._bridge.result.connect(self._on_result)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._quit = threading.Event()
        self._pending: _Task | None = None
        self._seq = 0
        self._applied = -1
        self._thread: threading.Thread | None = None
        self._current: _Task | None = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(90)
        self._debounce.timeout.connect(self._request)
        view.view_changed.connect(lambda *_: self._debounce.start())
        view.selection_changed.connect(self.update)
        self.setMinimumSize(200, 140)

    # ------------------------------------------------------------------
    def set_options(self, **kw) -> None:
        changed = False
        for k, v in kw.items():
            if k in self.opts and self.opts[k] != v:
                self.opts[k] = v
                changed = True
        if changed:
            self._debounce.start(0)

    def set_lane(self, key: str | None) -> None:
        if key == self.lane_key:
            return
        self.lane_key = key
        self._debounce.start(0)

    def playhead_seconds(self) -> float:
        return self.playhead

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(800, 260)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._quit.clear()
        self._thread = threading.Thread(target=self._worker, name="spectro-worker", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._quit.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    # ------------------------------------------------------------------
    # 请求 / 计算
    # ------------------------------------------------------------------
    def current_lane(self):
        lanes = [l for l in self.project.all_lanes() if l.key == self.lane_key]
        if lanes:
            return lanes[0]
        vis = self.project.visible_lanes()
        return vis[0] if vis else None

    def _request(self) -> None:
        if not self.isVisible() or self.project.is_empty():
            return
        rect = self.plot_rect()
        lane = self.current_lane()
        if lane is None:
            return
        v = self.view
        f0, f1 = v.frame_range(rect.x(), rect.width())
        cols = max(1, int(rect.width()))
        rows = max(1, int(rect.height()))
        if f1 - f0 < 8:
            return
        with self._lock:
            self._seq += 1
            self._pending = _Task(
                req_id=self._seq, lane_key=lane.key, f0=f0, f1=f1, cols=cols, rows=rows,
                nfft=int(self.opts["nfft"]), hop=None, window=str(self.opts["window"]),
                floor=float(self.opts["floor_db"]), top=float(self.opts["top_db"]),
                log=bool(self.opts["freq_log"]), col_index=0, rate=v.rate,
            )
        self._wake.set()

    def _worker(self) -> None:  # pragma: no cover - 后台线程
        while not self._quit.is_set():
            self._wake.wait()
            if self._quit.is_set():
                return
            with self._lock:
                task, self._pending = self._pending, None
            self._wake.clear()
            if task is None:
                continue
            try:
                self._compute(task)
            except Exception as exc:  # noqa: BLE001
                task.data = None
                task.img = None
                task.error = str(exc)  # type: ignore[attr-defined]
            self._bridge.result.emit(task)

    def _compute(self, task: _Task) -> None:
        lane = self.current_lane()
        if lane is None:
            return

        def read(start: int, length: int) -> np.ndarray:
            return self.project.read([lane], start, length, self.view.rate)

        db, nfft, hop = dsp.stft_spectrogram_stream(
            read, task.f0, task.f1, task.cols, nfft=task.nfft, hop=None, window=task.window)
        rows = dsp.resample_freq_rows(db, task.rows, task.log, fmax_hz=self.view.rate / 2.0)
        unit = dsp.db_to_unit(rows, task.floor, task.top)
        lut = dsp.colormap(str(self.opts["colormap"]))
        # resample 返回 (时间列, 频率行)，QImage 需要 (高=频率行, 宽=时间列)，且图像第 0 行是最高频
        grid = np.flipud(np.asarray(unit).T)
        rgb = dsp.colorize(grid, lut)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)])
        rgba = np.ascontiguousarray(rgba)
        h, w = rgb.shape[0], rgb.shape[1]
        img = QImage(rgba.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        task.data = db
        task.nfft, task.hop = nfft, hop
        task.img = img
        task.error = ""  # type: ignore[attr-defined]

    def _on_result(self, task: _Task) -> None:
        if task.req_id < self._applied:
            return
        self._applied = task.req_id
        self._current = task
        self.update()

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def paint_content(self, p: QPainter, rect: QRectF) -> None:
        task = self._current
        lane = self.current_lane()
        if lane is None:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(rect, Qt.AlignCenter, "无可显示的通道")
            return
        if task is None or task.img is None or task.img.isNull():
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(rect, Qt.AlignCenter, "谱图计算中…")
            self._request()
            return
        if task.lane_key != lane.key:
            self._request()
        p.save()
        p.setClipRect(rect)
        smooth = task.img.width() > rect.width() * 0.6
        p.setRenderHint(QPainter.SmoothPixmapTransform, smooth)
        p.drawImage(rect, task.img)
        p.restore()
        # 色标（右侧 dB 条）
        self._paint_colorbar(p, rect)

    def _paint_colorbar(self, p: QPainter, rect: QRectF) -> None:
        x = rect.right() - 34
        if x < rect.left() + 60:
            return
        lut = dsp.colormap(str(self.opts["colormap"]))
        col = np.repeat(lut[:, None, :], 8, axis=1)[::-1]
        rgba = np.dstack([col, np.full(col.shape[:2] + (1,), 255, np.uint8)])
        img = QImage(rgba.tobytes(), col.shape[1], col.shape[0], col.shape[1] * 4,
                     QImage.Format.Format_RGBA8888).copy()
        p.drawImage(QRectF(x, rect.top() + 6, 12, rect.height() - 12), img)
        p.setPen(QColor(theme.TEXT_DIM))
        font = QFont(p.font())
        font.setPointSizeF(8.5)
        p.setFont(font)
        for frac in (0.0, 0.5, 1.0):
            dbv = self.opts["top_db"] + frac * (self.opts["floor_db"] - self.opts["top_db"])
            y = rect.top() + 6 + frac * (rect.height() - 12)
            p.drawText(QRectF(x + 14, y - 7, 30, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{dbv:.0f}")

    def paint_amp_axis(self, p: QPainter, rect: QRectF) -> None:
        """纵轴是频率。"""
        v = self.view
        fmax = v.rate / 2.0
        font = QFont(p.font())
        font.setPointSizeF(9.0)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_DIM))
        ticks = dsp.freq_ticks(fmax, max(2, int(rect.height() / 42))) if not self.opts["freq_log"] else \
            [f for f in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000) if f <= fmax]
        for f in ticks:
            y = self.freq_to_y(f, rect, fmax)
            if y < rect.top() - 1 or y > rect.bottom() + 1:
                continue
            p.setPen(QPen(QColor(theme.GRID), 1, Qt.DotLine))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(QRectF(2, y - 8, AXIS_L - 10, 16), Qt.AlignRight | Qt.AlignVCenter,
                       dsp.format_freq(f))

    def freq_to_y(self, f: float, rect: QRectF, fmax: float | None = None) -> float:
        fmax = fmax or (self.view.rate / 2.0) or 1.0
        if not self.opts["freq_log"]:
            return rect.bottom() - (f / fmax) * rect.height()
        lo = max(1.0, 20.0)
        return rect.bottom() - (np.log(max(f, 1.0) / lo) / np.log(fmax / lo)) * rect.height()

    def y_to_freq(self, y: float, rect: QRectF) -> float:
        fmax = self.view.rate / 2.0
        frac = (rect.bottom() - y) / max(1.0, rect.height())
        if not self.opts["freq_log"]:
            return frac * fmax
        lo = max(1.0, 20.0)
        return float(lo * (fmax / lo) ** frac)

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        super().mouseMoveEvent(event)
        rect = self.plot_rect()
        pos = event.position()
        if rect.contains(pos) and self._current is not None and self._current.data is not None:
            f = self.y_to_freq(pos.y(), rect)
            db_arr = self._current.data
            cols, bins = db_arr.shape
            ci = int(np.clip((pos.x() - rect.left()) / max(1.0, rect.width()) * cols, 0, cols - 1))
            bi = int(np.clip(f / max(1.0, self.view.rate / 2.0) * (bins - 1), 0, bins - 1))
            self.hover_info = (f"{dsp.format_freq(f)}  {db_arr[ci, bi]:+.1f} dBFS   "
                               f"nfft={self._current.nfft} hop={self._current.hop}")
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hover_info = ""
        self.update()

    def description(self) -> str:
        return f"谱图 {self.opts['nfft']} {self.opts['window']} {self.opts['colormap']}"
