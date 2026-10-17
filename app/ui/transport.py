"""传输控制条：播放/暂停/停止、进度与拖拽定位、循环、跟随播放、音量、输出设备、采样率。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QWidget,
)

from ..core import dsp
from . import theme

SLIDER_STEPS = 20000


class TransportBar(QWidget):
    play = Signal()
    pause = Signal()
    stop = Signal()
    seek = Signal(int)                 # 帧
    loop_toggled = Signal(bool)
    follow_toggled = Signal(bool)
    device_changed = Signal(object)    # int | None
    rate_changed = Signal(int)
    volume_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._total_frames = 0
        self._rate = 48000
        self._scrubbing = False
        self._scrub_frames = 0

        self.btn_play = QPushButton("播放")
        self.btn_play.setToolTip("播放 / 暂停（空格）")
        self.btn_play.clicked.connect(self._on_play)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setToolTip("停止并回到开头（S）")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_loop = QPushButton("循环")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setToolTip("循环选区；无选区时循环整曲（L）")
        self.btn_loop.clicked.connect(lambda: self.loop_toggled.emit(self.btn_loop.isChecked()))
        self.btn_follow = QPushButton("跟随")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setChecked(True)
        self.btn_follow.setToolTip("播放时自动滚动时间窗")
        self.btn_follow.clicked.connect(lambda: self.follow_toggled.emit(self.btn_follow.isChecked()))
        self.btn_sel = QPushButton("选区→循环")
        self.btn_sel.setToolTip("把当前选区设为循环区间并缩放到选区")
        self.btn_sel.clicked.connect(self._on_sel_loop)

        self.pos_label = QLabel("0:00.000 / 0:00.000")
        f = QFont(self.pos_label.font())
        f.setStyleHint(QFont.Monospace)
        f.setPointSizeF(10.5)
        self.pos_label.setFont(f)
        self.frame_label = QLabel("帧 0")
        self.frame_label.setObjectName("hint")
        self.frame_label.setFont(f)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, SLIDER_STEPS)
        self.scrub.sliderPressed.connect(lambda: setattr(self, "_scrubbing", True))
        self.scrub.sliderReleased.connect(self._on_scrub)
        self.scrub.sliderMoved.connect(self._on_scrub_move)

        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 150)
        self.volume.setValue(100)
        self.volume.setFixedWidth(86)
        self.volume.setToolTip("输出音量")
        self.volume.valueChanged.connect(lambda v: self.volume_changed.emit(v / 100.0))

        self.device = QComboBox()
        self.device.setToolTip("输出设备（多通道声卡可选 8/10 通道输出）")
        self.device.setMinimumWidth(150)
        self.device.currentIndexChanged.connect(self._on_device)

        self.rate_box = QComboBox()
        self.rate_box.setToolTip("工程采样率：不同采样率的源会按时间轴线性换算后混合播放")
        self.rate_box.currentIndexChanged.connect(lambda *_: self.rate_changed.emit(self.current_rate()))

        self.status_hint = QLabel("")
        self.status_hint.setObjectName("hint")

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        for w in (self.btn_play, self.btn_stop, self.btn_loop, self.btn_follow, self.btn_sel,
                  self.pos_label, self.frame_label, self.scrub, QLabel("音量"), self.volume,
                  QLabel("输出"), self.device, QLabel("采样率"), self.rate_box):
            row1.addWidget(w)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addLayout(row1)

    # ------------------------------------------------------------------
    def set_devices(self, devices: list[dict], current: int | None) -> None:
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem("系统默认", None)
        for d in devices:
            self.device.addItem(f"{d['name']} ({d['channels']}ch)", d["index"])
        if current is not None:
            idx = self.device.findData(current)
            if idx >= 0:
                self.device.setCurrentIndex(idx)
        self.device.blockSignals(False)

    def set_rates(self, rates: list[int], current: int | None) -> None:
        self.rate_box.blockSignals(True)
        self.rate_box.clear()
        for r in rates:
            self.rate_box.addItem(f"{r:,} Hz", r)
        if current and current not in rates and rates:
            self.rate_box.addItem(f"{current:,} Hz", current)
        if current:
            idx = self.rate_box.findData(current)
            if idx >= 0:
                self.rate_box.setCurrentIndex(idx)
        self.rate_box.blockSignals(False)

    def current_rate(self) -> int:
        return int(self.rate_box.currentData() or 48000)

    def current_device(self):
        return self.device.currentData()

    def set_total(self, total_frames: int, rate: int) -> None:
        self._total_frames = max(0, int(total_frames))
        self._rate = int(rate) or 48000
        self.set_position(self.position_frames(), self._total_frames)

    def position_frames(self) -> int:
        if self._scrubbing or self._total_frames <= 0:
            return self._scrub_frames
        return int(self.scrub.value() / SLIDER_STEPS * self._total_frames)

    # ------------------------------------------------------------------
    def set_position(self, frames: int, total_frames: int | None = None) -> None:
        if total_frames is not None:
            self._total_frames = max(0, int(total_frames))
        dur = self._total_frames / self._rate
        pos = min(frames, self._total_frames) / self._rate
        if not self._scrubbing:
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(pos / max(dur, 1e-9) * SLIDER_STEPS) if dur else 0)
            self.scrub.blockSignals(False)
        self.pos_label.setText(f"{dsp.format_time(pos, True)} / {dsp.format_time(dur, True)}")
        self.frame_label.setText(f"帧 {frames:,} · {self._rate:,}Hz")
        self._scrub_frames = frames

    def set_playing(self, playing: bool) -> None:
        self.btn_play.setText("暂停" if playing else "播放")

    def set_hint(self, text: str) -> None:
        self.status_hint.setText(text)

    # ------------------------------------------------------------------
    def _on_play(self) -> None:
        self.play.emit()

    def _on_stop(self) -> None:
        self.stop.emit()

    def _on_scrub_move(self, value: int) -> None:
        frames = int(value / SLIDER_STEPS * self._total_frames)
        self._scrub_frames = frames
        self.pos_label.setText(f"{dsp.format_time(frames / self._rate, True)} · 拖拽中")

    def _on_scrub(self) -> None:
        self._scrubbing = False
        self.seek.emit(self._scrub_frames)

    def _on_device(self, *_):
        self.device_changed.emit(self.current_device())

    def _on_sel_loop(self) -> None:
        self.btn_loop.setChecked(True)
        self.loop_toggled.emit(True)
