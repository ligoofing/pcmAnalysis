"""时间轴视图基类：滚轮滑窗、⌘/Ctrl+滚轮时间缩放、⌥/Shift+滚轮幅值缩放、
点击定位播放、拖拽选区、双击适配、坐标轴与播放指针绘制。

时域波形与谱图共用本类，两个视图的时间窗口/缩放因此天然同步。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core import dsp
from . import theme
from .view_state import ViewState

AXIS_L = 64          # 左侧幅值轴宽度
AXIS_B = 24          # 底部时间轴高度
HEADER_H = 20        # 顶部信息行高度
DRAG_THRESHOLD = 5   # 超过该像素位移算拖拽选区，否则算点击定位


class TimeAxisView(QWidget):
    """带时间轴的通用可缩放视图。子类只需实现 :meth:`paint_content`。"""

    seek_requested = Signal(int)              # 帧号
    selection_made = Signal(int, int)         # 帧号区间
    cursor_moved = Signal(float, int)         # 秒, 帧
    view_interacted = Signal()                # 供状态栏刷新缩放提示

    def __init__(self, view: ViewState, project, parent=None) -> None:
        super().__init__(parent)
        self.view = view
        self.project = project
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(120)
        self._drag_origin: QPointF | None = None
        self._drag_pos: QPointF | None = None
        self._panning = False
        self._pan_anchor = 0.0
        self._cursor_x: float | None = None
        self.hover_info = ""
        view.view_changed.connect(self._on_view_changed)
        view.selection_changed.connect(self.update)

    # ------------------------------------------------------------------
    # 几何
    # ------------------------------------------------------------------
    def plot_rect(self) -> QRectF:
        r = self.rect()
        return QRectF(AXIS_L, HEADER_H, max(10, r.width() - AXIS_L - 8),
                      max(10, r.height() - HEADER_H - AXIS_B))

    def _on_view_changed(self, reason: str) -> None:
        self.update()

    # ------------------------------------------------------------------
    # 绘制骨架
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        rect = self.plot_rect()
        p.fillRect(self.rect(), QColor(theme.BG))
        p.fillRect(rect, QColor("#101216"))
        if self.project.is_empty() or self.view.total_frames <= 0:
            self._draw_placeholder(p)
            p.end()
            return
        self.paint_frame(p, rect)
        self.paint_content(p, rect)
        self.paint_overlay(p, rect)
        p.end()

    def _draw_placeholder(self, p: QPainter) -> None:
        p.setPen(QColor(theme.TEXT_DIM))
        f = QFont(p.font())
        f.setPointSize(12)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter,
                   "打开 PCM / WAV 文件开始分析\n\n⌘+滚轮 = 时间缩放     滚轮 = 滑动时间窗     "
                   "⌥+滚轮 = 幅值缩放     点击 = 定位播放")

    def paint_content(self, p: QPainter, rect: QRectF) -> None:
        raise NotImplementedError

    # ---- 网格 / 轴 ----------------------------------------------------
    def paint_frame(self, p: QPainter, rect: QRectF) -> None:
        v = self.view
        p.save()
        p.setClipRect(rect)
        # 时间竖线
        ticks = dsp.time_ticks(v.t0, v.t0 + v.win, max(3, int(rect.width() / 110)))
        font = QFont(p.font())
        font.setPointSizeF(9.0)
        for t in ticks:
            x = v.seconds_to_x(t, rect.x(), rect.width())
            p.setPen(QPen(QColor(theme.GRID), 1, Qt.DotLine))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        p.restore()
        # 时间刻度文字
        p.setPen(QColor(theme.TEXT_DIM))
        p.setFont(font)
        step = (ticks[1] - ticks[0]) if len(ticks) > 1 else max(v.win / 6, 1e-6)
        for t in ticks:
            x = v.seconds_to_x(t, rect.x(), rect.width())
            label = dsp.format_tick_time(t, step)
            p.drawText(QRectF(x + 3, rect.bottom() + 4, 130, AXIS_B - 4),
                       Qt.AlignLeft | Qt.AlignTop, label)
        # 选区
        if v.has_selection:
            x0 = v.seconds_to_x(v.sel_start / v.rate, rect.x(), rect.width())
            x1 = v.seconds_to_x(v.sel_end / v.rate, rect.x(), rect.width())
            band = QRectF(min(x0, x1), rect.top(), abs(x1 - x0), rect.height())
            p.fillRect(band, QColor(79, 179, 255, 34))
            p.setPen(QPen(QColor(theme.ACCENT), 1))
            p.drawLine(band.topLeft(), band.bottomLeft())
            p.drawLine(band.topRight(), band.bottomRight())
        self.paint_amp_axis(p, rect)
        self.paint_header(p, rect)

    def paint_amp_axis(self, p: QPainter, rect: QRectF) -> None:
        """默认线性幅值轴；子类可覆盖换成频率轴。"""
        v = self.view
        font = QFont(p.font())
        font.setPointSizeF(9.0)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_DIM))
        for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
            y = rect.center() .y() - frac * rect.height() / 2.0
            value = frac / v.amp
            p.setPen(QPen(QColor(theme.GRID if frac else theme.ZERO), 1,
                          Qt.SolidLine if frac == 0 else Qt.DotLine))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(QColor(theme.TEXT_DIM))
            text = f"{value:+.2f}" if v.amp >= 1 else f"{value:+.1f}"
            p.drawText(QRectF(2, y - 8, AXIS_L - 8, 16),
                       Qt.AlignRight | Qt.AlignVCenter, text)

    def paint_header(self, p: QPainter, rect: QRectF) -> None:
        font = QFont(p.font())
        font.setPointSizeF(9.5)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_DIM))
        v = self.view
        span_ms = v.win * 1000
        spp = v.win * v.rate / max(1.0, rect.width())
        spp_txt = f"{spp:.4f}" if spp < 0.01 else (f"{spp:.2f}" if spp < 10 else f"{spp:.0f}")
        info = f"窗口 {span_ms:,.2f} ms   幅值 ×{v.amp:g}   {spp_txt} 样本/像素"
        p.drawText(QRectF(AXIS_L, 2, rect.width(), HEADER_H - 2),
                   Qt.AlignLeft | Qt.AlignVCenter, info)
        if self.hover_info:
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(AXIS_L, 2, rect.width(), HEADER_H - 2),
                       Qt.AlignRight | Qt.AlignVCenter, self.hover_info)

    def paint_overlay(self, p: QPainter, rect: QRectF) -> None:
        v = self.view
        p.save()
        p.setClipRect(rect)
        # 播放指针
        pos_sec = self.playhead_seconds()
        x = v.seconds_to_x(pos_sec, rect.x(), rect.width())
        if rect.left() - 1 <= x <= rect.right() + 1:
            pen = QPen(QColor(theme.PLAYHEAD), 1.4)
            p.setPen(pen)
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            p.setBrush(QColor(theme.PLAYHEAD))
            p.drawPath(self._triangle(x, rect.top()))
        # 拖拽中的临时选区
        if self._drag_origin is not None and self._drag_pos is not None:
            x0, x1 = sorted((self._drag_origin.x(), self._drag_pos.x()))
            band = QRectF(x0, rect.top(), x1 - x0, rect.height())
            p.fillRect(band, QColor(79, 179, 255, 46))
        p.restore()
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5))

    @staticmethod
    def _triangle(x: float, y: float) -> QPainterPath:
        path = QPainterPath()
        path.moveTo(x - 5, y)
        path.lineTo(x + 5, y)
        path.lineTo(x, y + 7)
        path.closeSubpath()
        return path

    def playhead_seconds(self) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802
        v = self.view
        mods = event.modifiers()
        ad = event.angleDelta()
        pd = event.pixelDelta()
        rect = self.plot_rect()
        zoom_key = mods & (Qt.ControlModifier | Qt.MetaModifier)
        amp_key = mods & (Qt.AltModifier | Qt.ShiftModifier)
        unit = (ad.y() or ad.x()) / 120.0
        if abs(ad.y()) < 8 and abs(ad.x()) < 8 and (pd.y() or pd.x()):
            unit = (pd.y() or pd.x()) / 40.0
        if zoom_key:                              # 时间维度放大
            anchor = (event.position().x() - rect.x()) / max(1.0, rect.width())
            v.zoom_time(1.18 ** (unit * 3.0), anchor)
        elif amp_key:                             # 幅值放大
            v.zoom_amp(1.18 ** (unit * 3.0))
        else:                                     # 滑动时间窗口
            if pd.y() or pd.x():
                px = pd.y() or pd.x()
                v.scroll_seconds(-px / max(1.0, rect.width()) * v.win)
            else:
                v.scroll_seconds(-(ad.y() or ad.x()) / 120.0 * v.win * 0.18)
        self.view_interacted.emit()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        rect = self.plot_rect()
        pos = event.position()
        if not rect.contains(pos):
            return
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_anchor = pos.x()
            self._pan_t0 = self.view.t0
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._drag_origin = pos
            self._drag_pos = pos
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self.contextMenuEvent(pos)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        rect = self.plot_rect()
        pos = event.position()
        self._cursor_x = pos.x()
        if self._panning:
            dx = pos.x() - self._pan_anchor
            self.view.set_window(self._pan_t0 - dx / max(1.0, rect.width()) * self.view.win, self.view.win)
            self.view_interacted.emit()
            event.accept()
            return
        if self._drag_origin is not None:
            self._drag_pos = pos
            if abs(pos.x() - self._drag_origin.x()) > DRAG_THRESHOLD:
                self.view.set_selection(self._frame_at(self._drag_origin.x()), self._frame_at(pos.x()))
            self.update()
            event.accept()
            return
        if rect.contains(pos):
            self.cursor_moved.emit(self.view.x_to_seconds(pos.x(), rect.x(), rect.width()),
                                   self._frame_at(pos.x()))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        if event.button() != Qt.LeftButton or self._drag_origin is None:
            return
        dragged = abs(event.position().x() - self._drag_origin.x()) > DRAG_THRESHOLD
        a, b = self._frame_at(self._drag_origin.x()), self._frame_at(event.position().x())
        self._drag_origin = self._drag_pos = None
        if dragged:
            self.view.set_selection(a, b)
            self.selection_made.emit(min(a, b), max(a, b))
        else:
            self.view.clear_selection()
            self.seek_requested.emit(a)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        v = self.view
        if v.has_selection and not v.zoom_to_selection():
            v.fit_all()
        else:
            v.fit_all()
        self.view_interacted.emit()
        event.accept()

    def contextMenuEvent(self, pos_or_event) -> None:  # noqa: N802
        from PySide6.QtWidgets import QMenu
        pos = pos_or_event if isinstance(pos_or_event, QPointF) else pos_or_event.pos()
        menu = QMenu(self)
        act_fit = menu.addAction("适配全曲 (双击 / 数字 0)")
        act_sel = menu.addAction("缩放到选区")
        menu.addSeparator()
        act_copy = menu.addAction("复制图像到剪贴板")
        chosen = menu.exec(self.mapToGlobal(pos if isinstance(pos_or_event, QPointF)
                                           else pos_or_event.globalPosition().toPoint()))
        if chosen is act_fit:
            self.view.fit_all()
        elif chosen is act_sel:
            self.view.zoom_to_selection()
        elif chosen is act_copy:
            img = self.grab()
            QGuiApplication.clipboard().setImage(img.toImage())
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        v = self.view
        key = event.key()
        step = v.win * (0.1 if event.modifiers() & Qt.ShiftModifier else 0.02)
        if key == Qt.Key_Left:
            v.scroll_seconds(-step)
        elif key == Qt.Key_Right:
            v.scroll_seconds(step)
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            v.zoom_time(1.5)
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            v.zoom_time(1 / 1.5)
        elif key == Qt.Key_A:
            v.zoom_amp(1.4)
        elif key == Qt.Key_Z:
            v.zoom_amp(1 / 1.4)
        elif key == Qt.Key_0:
            v.fit_all()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _frame_at(self, x: float) -> int:
        rect = self.plot_rect()
        return self.view.x_to_frame(x, rect.x(), rect.width())
