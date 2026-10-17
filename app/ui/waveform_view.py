"""时域波形视图：多通道分道/叠加显示、按像素精确抽稀、包络填充、线性/dB 幅值轴。

三点工程考虑
------------
1. **大文件保护**：一次最多读 ``CHUNK_SAMPLES`` 个样本，按像素列分块求包络，
   整首 10 分钟 8 通道文件也不会撑爆内存。
2. **播放不重算**：波形渲染结果缓存成透明 ``QImage``，播放指针每帧移动只重绘缓存 +
   指针；窗口/幅值/通道集合变化才失效。
3. **削波可见**：达到 ±0.999 的像素列用红色短标提示，排查录音过载很常用。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF

from ..core import dsp
from . import theme
from .time_view_base import AXIS_L, TimeAxisView
from .view_state import lane_bands as compute_bands

CHUNK_SAMPLES = 1 << 18
DB_FLOOR = -120.0
CLIP_LEVEL = 0.999


class WaveView(TimeAxisView):
    def __init__(self, view, project, parent=None) -> None:
        super().__init__(view, project, parent)
        self.mode = "lanes"          # lanes | overlay
        self.y_scale = "linear"      # linear | db
        self.fill = "envelope"       # envelope | rms | line
        self.playhead = 0.0
        self._cache: QImage | None = None
        self._cache_key: tuple | None = None
        self._stats: dict[str, dict] = {}
        view.view_changed.connect(self._invalidate)
        view.selection_changed.connect(self._invalidate)

    # ------------------------------------------------------------------
    def set_options(self, mode: str | None = None, y_scale: str | None = None,
                    fill: str | None = None) -> None:
        changed = ((mode and mode != self.mode) or (y_scale and y_scale != self.y_scale)
                   or (fill and fill != self.fill))
        self.mode = mode or self.mode
        self.y_scale = y_scale or self.y_scale
        self.fill = fill or self.fill
        if changed:
            self._cache_key = None
            self.update()

    def _invalidate(self, reason: str = "") -> None:
        self._cache_key = None
        self.update()

    def lanes(self) -> list:
        return self.project.visible_lanes()

    def playhead_seconds(self) -> float:
        return self.playhead

    # 幅值刻度在控件坐标系里画（缓存图内没有左侧轴的空间）
    def paint_amp_axis(self, p: QPainter, rect: QRectF) -> None:
        font = QFont(p.font())
        font.setPointSizeF(9.0)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_DIM))
        for y_img, text in getattr(self, "_amp_ticks", []):
            p.drawText(QRectF(2, rect.top() + y_img - 7, AXIS_L - 8, 14),
                       Qt.AlignRight | Qt.AlignVCenter, text)

    # ------------------------------------------------------------------
    # 布局
    # ------------------------------------------------------------------
    def lane_bands(self, w: float, h: float, n: int) -> list[tuple[float, float]]:
        """返回每道 (中心 y, 半高)；与顶部缩略图共用同一套布局函数。"""
        return compute_bands(n, h, self.mode)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def paint_content(self, p: QPainter, rect: QRectF) -> None:
        lanes = self.lanes()
        if not lanes:
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(rect, Qt.AlignCenter, "左栏勾选通道后即可显示波形与谱图")
            return
        w, h = max(1, int(rect.width())), max(1, int(rect.height()))
        v = self.view
        key = (
            round(v.t0, 9), round(v.win, 9), round(v.amp, 6), self.mode, self.y_scale,
            self.fill, (w, h), tuple(l.key for l in lanes), v.total_frames, v.rate,
        )
        if key != self._cache_key or self._cache is None:
            self._cache = self._render(lanes, w, h)
            self._cache_key = key
        p.drawImage(rect.topLeft(), self._cache)
        if self._stats:
            p.setPen(QColor(theme.TEXT_DIM))
            p.setFont(QFont(p.font().family(), 9))

    def _render(self, lanes: list, w: int, h: int) -> QImage:
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, False)
        bands = self.lane_bands(w, h, len(lanes))
        v = self.view
        f0, f1 = v.frame_range(0.0, float(w))
        f1 = max(f1, f0 + 1)
        spp = (f1 - f0) / w
        chunk_cols = max(1, min(w, int(CHUNK_SAMPLES / max(1.0, spp))))
        self._stats = {}
        for c0 in range(0, w, chunk_cols):
            ncols = min(chunk_cols, w - c0)
            s0 = int(f0 + c0 * spp)
            s1 = int(f0 + (c0 + ncols) * spp)
            if s1 <= s0:
                s1 = s0 + 1
            block = self.project.read(lanes, s0, s1 - s0, v.rate)
            env = dsp.pixel_envelope(block, ncols, s0, s0, s1)
            self._merge_stats(block, lanes)
            self._paint_columns(p, lanes, bands, env, c0, ncols)
        self._paint_lane_decoration(p, lanes, bands, w, h)
        p.end()
        return img

    def _merge_stats(self, block, lanes) -> None:
        if block.size == 0:
            return
        s = dsp.signal_stats(block)
        for i, lane in enumerate(lanes):
            if i >= block.shape[1]:
                continue
            cur = self._stats.setdefault(lane.key, {"peak": 0.0, "rms_sq": 0.0, "n": 0, "clip": 0})
            cur["peak"] = max(cur["peak"], float(s["peak"][i]))
            cur["rms_sq"] += float(s["rms"][i]) ** 2 * block.shape[0]
            cur["n"] += block.shape[0]
            cur["clip"] += int(s["clip"][i])

    def _paint_columns(self, p: QPainter, lanes, bands, env, c0: int, ncols: int) -> None:
        amp = self.view.amp
        mins, maxs, rms = env["min"], env["max"], env["rms"]
        per_sample = bool(env.get("per_sample"))
        xs = c0 + np.arange(ncols, dtype=np.float64)
        for i, lane in enumerate(lanes):
            if i >= mins.shape[1]:
                continue
            cy, half = bands[i]
            color = QColor(lane.color)
            y_hi = self._map(maxs[:, i], cy, half, amp)
            y_lo = self._map(mins[:, i], cy, half, amp)
            if self.fill == "rms":
                y_up = self._map(rms[:, i], cy, half, amp)
                y_dn = self._map(-rms[:, i], cy, half, amp)
            else:
                y_up, y_dn = y_hi, y_lo

            if per_sample or self.fill == "line":
                # 放大到样本级：连成折线，样本报红
                p.setPen(QPen(color, 1.0))
                pts = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, y_hi)])
                p.drawPolyline(pts)
                hot = np.where(maxs[:, i] >= CLIP_LEVEL)[0]
                if hot.size:
                    p.setPen(QPen(QColor("#ff4d4d"), 1.6))
                    for k in hot[:4000]:
                        p.drawPoint(QPointF(float(xs[k]), float(y_hi[k])))
                continue

            top = np.minimum(y_up, y_dn)
            bot = np.maximum(y_up, y_dn)
            thin = bot - top < 1.0
            bot = np.where(thin, top + 1.0, bot)
            brush = QColor(color)
            brush.setAlpha(160)
            p.setPen(Qt.NoPen)
            p.setBrush(brush)
            for k in range(ncols):
                p.drawRect(int(xs[k]), int(top[k]), 1, max(1, int(bot[k] - top[k])))
            hot = np.where(np.maximum(np.abs(maxs[:, i]), np.abs(mins[:, i])) >= CLIP_LEVEL)[0]
            if hot.size:
                p.setBrush(QColor(255, 77, 77, 230))
                for k in hot[:3000]:
                    p.drawRect(int(xs[k]), int(top[k]), 1, 2)

    def _map(self, values, cy: float, half: float, amp: float) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        if self.y_scale == "db":
            db = 20.0 * np.log10(np.maximum(np.abs(v), 1e-7))
            norm = np.clip((db - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0)
            sign = np.where(v < 0, -1.0, 1.0)
            return cy - sign * norm * half
        headroom = 0.92 if self.mode == "lanes" else 0.46
        return cy - np.clip(v * amp * headroom, -1.15, 1.15) * half

    def _paint_lane_decoration(self, p: QPainter, lanes, bands, w: int, h: int) -> None:
        font = QFont(p.font())
        font.setPointSizeF(9.0)
        p.setFont(font)
        amp = self.view.amp
        self._amp_ticks = []
        for i, lane in enumerate(lanes):
            cy, half = bands[i]
            if self.mode == "overlay":
                continue
            p.setPen(QPen(QColor(theme.GRID), 1, Qt.DotLine))
            p.drawLine(QPointF(0, cy - half - 2), QPointF(w, cy - half - 2))
        for i, lane in enumerate(lanes):
            cy, half = bands[i]
            st = self._stats.get(lane.key)
            pk = st["peak"] if st else 0.0
            rms = np.sqrt(st["rms_sq"] / st["n"]) if st and st["n"] else 0.0
            if self.mode == "lanes":
                tag = (f"  pk {self._db(pk)}  rms {self._db(rms)}"
                       + (f"  削波×{st['clip']}" if st and st["clip"] else ""))
                label = f"{lane.label}{tag}"
                # 首道标签贴着带上沿，其余画在分隔线上方
                y = max(1.0, cy - half - 15) if i == 0 else cy - half - 15
                p.setPen(QColor(lane.color))
                p.drawText(QRectF(2, y, w - 6, 14), Qt.AlignLeft | Qt.AlignTop, label)
            # 零线
            p.setPen(QPen(QColor(theme.ZERO), 1))
            p.drawLine(QPointF(0, cy), QPointF(w, cy))
            # 幅值刻度：分道时各道量程相同，只标第一道，避免左侧一列重复数字
            if self.mode == "overlay" or i == 0:
                headroom = 0.92 if self.mode == "lanes" else 0.46
                for value in (-1.0, -0.5, 0.5, 1.0):
                    # 首道上方贴着通道名，只标下半轴，避免与标签叠字
                    if self.mode == "lanes" and i == 0 and value > 0:
                        continue
                    y = cy - value * amp * headroom * half
                    if abs(y - cy) > half:      # 放大后该刻度已出道界，不标
                        continue
                    txt = (f"{20 * np.log10(max(abs(value), 1e-3)):+.0f}" if self.y_scale == "db"
                           else f"{value:+.2f}")
                    self._amp_ticks.append((y, txt))
        if self.mode == "overlay":
            self._paint_legend(p, lanes, w)

    @staticmethod
    def _db(value: float) -> str:
        return "-inf" if value <= 1e-6 else f"{20 * np.log10(value):+.1f}dB"

    def _paint_legend(self, p: QPainter, lanes, w: int) -> None:
        """叠加模式：图例排一行，避免每道标签叠在一起。"""
        x = 2.0
        for lane in lanes:
            st = self._stats.get(lane.key)
            pk = st["peak"] if st else 0.0
            text = f"{lane.label} {self._db(pk)}"
            p.setPen(QColor(lane.color))
            p.drawText(QRectF(x, 0, 220, 14), Qt.AlignLeft | Qt.AlignTop, text)
            x += p.fontMetrics().horizontalAdvance(text) + 14
            if x > w - 120:
                break

    # ------------------------------------------------------------------
    def stats_text(self) -> str:
        out = []
        for lane in self.lanes()[:10]:
            st = self._stats.get(lane.key)
            if not st:
                continue
            rms = np.sqrt(st["rms_sq"] / st["n"]) if st["n"] else 0.0
            clip = f"  削波×{st['clip']}" if st["clip"] else ""
            out.append(f"{lane.label} pk {self._db(st['peak'])}/rms {self._db(rms)}{clip}")
        return "   ".join(out)

    def description(self) -> str:
        return f"{self.mode} · {self.y_scale} · {self.fill}"
