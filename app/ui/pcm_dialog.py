"""裸 PCM 参数弹窗：手动指定采样率/通道/位深/字节序/偏移/排布，带历史与预设。"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..core.audio_io import PcmFormat

RATES = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000,
         128000, 176400, 192000, 256000, 352800, 384000]

WIDTHS = [
    ("8 bit 无符号 (uint8)", 1, False, True),
    ("8 bit 有符号 (int8)", 1, False, False),
    ("16 bit 小端整型", 2, False, True),
    ("24 bit 整型 (3 字节 packed)", 3, False, True),
    ("32 bit 整型", 4, False, True),
    ("32 bit 单精度浮点", 4, True, True),
    ("64 bit 双精度浮点", 8, True, True),
]


class PcmFormatDialog(QDialog):
    def __init__(self, parent, path: str, config, initial: PcmFormat | None = None) -> None:
        super().__init__(parent)
        self.path = path
        self.config = config
        self.setWindowTitle(f"PCM 参数 · {os.path.basename(path)}")
        self.setMinimumWidth(430)

        size = os.path.getsize(path) if os.path.exists(path) else 0
        self.size_label = QLabel(f"文件大小：{size:,} 字节")
        self.result_label = QLabel()
        self.result_label.setObjectName("hint")

        self.rate = QComboBox()
        self.rate.setEditable(True)
        self.rate.addItems([str(r) for r in RATES])
        self.rate.setCurrentText("48000")
        self.rate.currentIndexChanged.connect(self._refresh)

        self.channels = QSpinBox()
        self.channels.setRange(1, 64)
        self.channels.valueChanged.connect(self._refresh)

        self.width = QComboBox()
        for label, *_ in WIDTHS:
            self.width.addItem(label)
        self.width.currentIndexChanged.connect(self._on_width)

        self.endian = QComboBox()
        self.endian.addItems(["小端 LE", "大端 BE"])
        self.endian.currentIndexChanged.connect(self._refresh)

        self.offset = QSpinBox()
        self.offset.setRange(0, 1 << 24)
        self.offset.setSuffix(" 字节")
        self.offset.valueChanged.connect(self._refresh)

        self.planar = QCheckBox("平面排布（每通道连续存放）")
        self.planar.toggled.connect(self._refresh)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("采样率", self.rate)
        form.addRow("通道数", self.channels)
        form.addRow("样本格式", self.width)
        form.addRow("字节序", self.endian)
        form.addRow("跳过文件头", self.offset)
        form.addRow("", self.planar)

        box = QGroupBox("参数")
        box.setLayout(form)

        # ---- 历史 / 预设 ------------------------------------------------
        self.history = QComboBox()
        self.presets = QComboBox()
        self._reload_saved()
        self.history.currentIndexChanged.connect(lambda *_: self._apply_saved(self.history, True))
        self.presets.currentIndexChanged.connect(lambda *_: self._apply_saved(self.presets, False))

        save_btn = QPushButton("存为预设…")
        save_btn.clicked.connect(self._save_preset)
        del_btn = QPushButton("删除预设")
        del_btn.clicked.connect(self._delete_preset)
        row = QHBoxLayout()
        row.addWidget(QLabel("最近"))
        row.addWidget(self.history, 2)
        row.addWidget(QLabel("预设"))
        row.addWidget(self.presets, 2)
        row.addWidget(save_btn)
        row.addWidget(del_btn)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addLayout(row)
        layout.addWidget(self.size_label)
        layout.addWidget(self.result_label)
        layout.addWidget(btns)

        start = initial or (self._first_recent() or PcmFormat())
        self._set_from_format(start)
        self._apply_initial_guess(size)
        self._refresh()

    # ------------------------------------------------------------------
    def _first_recent(self) -> PcmFormat | None:
        hist = self.config["pcm_history"]
        return PcmFormat.from_dict(hist[0]) if hist else None

    def _reload_saved(self) -> None:
        self.history.clear()
        self.history.addItem("（选择最近的参数）")
        for entry in self.config["pcm_history"]:
            self.history.addItem(self._describe(entry))
        self.presets.clear()
        self.presets.addItem("（选择预设）")
        for entry in self.config["pcm_presets"]:
            self.presets.addItem(entry.get("name", "preset"))

    @staticmethod
    def _describe(entry: dict) -> str:
        fmt = PcmFormat.from_dict(entry)
        return fmt.describe()

    def _apply_saved(self, combo: QComboBox, is_history: bool) -> None:
        idx = combo.currentIndex()
        if idx <= 0:
            return
        store = self.config["pcm_history"] if is_history else self.config["pcm_presets"]
        self._set_from_format(PcmFormat.from_dict(store[idx - 1]))
        combo.setCurrentIndex(0)

    def _set_from_format(self, fmt: PcmFormat) -> None:
        self.rate.setCurrentText(str(fmt.sample_rate))
        self.channels.setValue(fmt.channels)
        for i, (_, width, is_float, signed) in enumerate(WIDTHS):
            if width == fmt.sample_width and is_float == fmt.is_float and signed == fmt.signed:
                self.width.setCurrentIndex(i)
                break
        self.endian.setCurrentIndex(1 if fmt.big_endian else 0)
        self.offset.setValue(fmt.offset)
        self.planar.setChecked(fmt.planar)

    def _on_width(self, *_):
        self._refresh()

    def _apply_initial_guess(self, size: int) -> None:
        """WAV 伪装成 .pcm 之类情况：若长度恰为 44/78 字节的整数倍，提示可能的头长。"""
        if size < 44:
            return
        for guess in (44, 78, 60):
            if (size - guess) % 4 == 0:
                self.result_label.setText(
                    f"提示：若文件其实带 WAV 头，可把「跳过文件头」设为 {guess} 字节。"
                    .replace("。", "。"))
                break

    # ------------------------------------------------------------------
    def build_format(self) -> PcmFormat:
        _, width, is_float, signed = WIDTHS[self.width.currentIndex()]
        try:
            rate = int(float(self.rate.currentText()))
        except ValueError:
            rate = 48000
        return PcmFormat(
            sample_rate=max(1, rate),
            channels=self.channels.value(),
            sample_width=width,
            is_float=is_float,
            signed=signed,
            big_endian=self.endian.currentIndex() == 1,
            offset=self.offset.value(),
            planar=self.planar.isChecked(),
        )

    def _refresh(self, *_):
        fmt = self.build_format()
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        usable = max(0, size - fmt.offset)
        frames = usable // fmt.block_align
        dur = frames / fmt.sample_rate if fmt.sample_rate else 0
        self.result_label.setText(
            f"解析结果：{fmt.describe()}\n"
            f"帧数 {frames:,} · 时长 {dur:.3f} s · 每帧 {fmt.block_align} 字节"
            + ("　⚠ 尾部有 %d 字节余数，参数可能不对" % (usable % fmt.block_align)
               if usable % fmt.block_align else ""))
        self.result_label.setStyleSheet("color:%s" % ("#ff8a4c" if usable % fmt.block_align else "#8b93a7"))

    def accept(self) -> None:  # noqa: N802
        fmt = self.build_format()
        size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        if (size - fmt.offset) // fmt.block_align <= 0:
            QMessageBox.warning(self, "参数无效", "按这组参数解析出的帧数为 0，请调整通道数/位深/偏移。")
            return
        self.config.remember_pcm_format(fmt.to_dict())
        super().accept()

    def _save_preset(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "保存预设", "预设名称：")
        if ok and name.strip():
            self.config.add_preset(self.build_format().to_dict(), name.strip())
            self._reload_saved()

    def _delete_preset(self) -> None:
        idx = self.presets.currentIndex()
        if idx <= 0:
            return
        name = self.config["pcm_presets"][idx - 1].get("name", "")
        self.config.remove_preset(name)
        self._reload_saved()
