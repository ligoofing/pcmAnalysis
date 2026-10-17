"""左栏文件/通道面板：显示已打开的文件，勾选通道控制显示与播放；
多选文件可合并通道，单文件可解交织导出。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme

SID = Qt.UserRole + 1
LANE_KEY = Qt.UserRole + 2


class FilePanel(QWidget):
    open_clicked = Signal()
    remove_clicked = Signal(list)            # sids
    merge_clicked = Signal(list)             # sids（按选择顺序）
    export_clicked = Signal(list)            # lane keys（勾选的通道）
    deinterleave_clicked = Signal(str)       # sid
    lanes_changed = Signal()
    lane_activated = Signal(str)             # 双击通道 → 谱图切到该通道
    source_activated = Signal(str)

    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["名称", "通道", "采样率", "时长", "格式"])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._on_changed)
        self.tree.itemDoubleClicked.connect(self._on_double)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.tree.setMinimumWidth(320)

        self.status = QLabel("未打开文件")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)

        btn_open = QPushButton("打开文件…")
        btn_open.clicked.connect(self.open_clicked)
        btn_merge = QPushButton("合并所选文件通道")
        btn_merge.setToolTip("在左栏用 ⌘/Shift 多选文件后点击：把这些文件的通道串成一个新的多声道源，\n"
                            "可整体显示、整体播放，也可导出为一个多声道 WAV。")
        btn_merge.clicked.connect(self._on_merge)
        btn_deint = QPushButton("解交织导出…")
        btn_deint.setToolTip("把选中文件的每个通道写成独立的单声道 WAV。")
        btn_deint.clicked.connect(self._on_deint)
        btn_export = QPushButton("导出勾选通道…")
        btn_export.setToolTip("把当前勾选的通道（含合并源）导出为一个多声道 WAV。")
        btn_export.clicked.connect(self._on_export)
        btn_remove = QPushButton("移除")
        btn_remove.clicked.connect(self._on_remove)

        all_btn = QPushButton("全选")
        all_btn.clicked.connect(lambda: self._check_all(True))
        none_btn = QPushButton("全不选")
        none_btn.clicked.connect(lambda: self._check_all(False))

        rows = QHBoxLayout()
        rows.addWidget(btn_open)
        rows.addWidget(btn_remove)
        rows.addStretch(1)
        rows.addWidget(all_btn)
        rows.addWidget(none_btn)
        grid_top = QHBoxLayout()
        grid_top.addWidget(btn_merge)
        grid_top.addWidget(btn_deint)
        grid_bot = QHBoxLayout()
        grid_bot.addWidget(btn_export)
        grid_bot.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QLabel("已打开的文件与通道", objectName="title"))
        layout.addWidget(self.tree, 1)
        layout.addLayout(rows)
        layout.addLayout(grid_top)
        layout.addLayout(grid_bot)
        layout.addWidget(self.status)

    # ------------------------------------------------------------------
    def sync(self) -> None:
        """按 project 重建树（保持展开状态与滚动位置）。"""
        expanded = {i.data(0, SID) for i in self._source_items() if i.isExpanded()}
        self.tree.blockSignals(True)
        self.tree.clear()
        for src in self.project.file_sources():
            top = QTreeWidgetItem([
                ("◆ " if src.kind == "mix" else "") + src.label,
                str(src.channel_count),
                f"{src.sample_rate:,}" if src.sample_rate else "-",
                f"{src.duration:.3f}s",
                src.format_desc,
            ])
            top.setData(0, SID, src.sid)
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            checked = [l.visible for l in src.lanes] or [False]
            top.setCheckState(0, Qt.Checked if all(checked) else (Qt.PartiallyChecked if any(checked) else Qt.Unchecked))
            font = QFont(top.font(0))
            if src.kind == "mix":
                font.setItalic(True)
                top.setForeground(0, QColor(theme.ACCENT))
            top.setFont(0, font)
            tip = src.path
            if not tip:
                names = []
                for l in src.lanes:
                    origin = self.project.sources.get(l.raw_ref()[0])
                    names.append(f"{origin.label} CH{l.raw_ref()[1]}" if origin else "?")
                tip = "合并通道源：" + " + ".join(names)
            top.setToolTip(0, tip)
            for lane in src.lanes:
                child = QTreeWidgetItem([lane.label, "", "", "", ""])
                child.setData(0, SID, src.sid)
                child.setData(0, LANE_KEY, lane.key)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                child.setCheckState(0, Qt.Checked if lane.visible else Qt.Unchecked)
                child.setForeground(0, QColor(lane.color))
                raw_oid, raw_ch = lane.raw_ref()
                origin = self.project.sources.get(raw_oid)
                if origin and origin.sid != src.sid:
                    child.setText(4, f"← {origin.label} CH{raw_ch}")
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            if src.sid in expanded:
                top.setExpanded(True)
        self.tree.blockSignals(False)
        self._update_status()

    def _source_items(self):
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    def _update_status(self) -> None:
        n_src = len(self.project.order)
        vis = self.project.visible_lanes()
        total = self.project.total_frames
        rate = self.project.sample_rate or 0
        if not n_src:
            self.status.setText("未打开文件")
            return
        if not rate:
            self.status.setText(f"{n_src} 个源 · 勾选 {len(vis)}/{len(self.project.all_lanes())} 通道（采样率待定）")
            return
        self.status.setText(
            f"{n_src} 个源 · 勾选 {len(vis)}/{len(self.project.all_lanes())} 通道\n"
            f"总时长 {total / rate:.3f} s @ {rate:,} Hz")

    # ------------------------------------------------------------------
    def _on_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self.tree.signalsBlocked():
            return
        sid = item.data(0, SID)
        key = item.data(0, LANE_KEY)
        checked = item.checkState(0) == Qt.Checked
        self.tree.blockSignals(True)
        if key:                                  # 通道级
            self._set_lane_visible(key, checked)
            self._refresh_parent_check(item.parent())
        elif sid:                                # 源级：整棵树同步
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, item.checkState(0))
                self._set_lane_visible(child.data(0, LANE_KEY), checked)
        self.tree.blockSignals(False)
        self.lanes_changed.emit()
        self._update_status()

    def _set_lane_visible(self, key: str, visible: bool) -> None:
        lane = self.project.lane(key)
        if lane is not None:
            lane.visible = visible

    def _refresh_parent_check(self, parent: QTreeWidgetItem | None) -> None:
        if parent is None:
            return
        states = {parent.child(i).checkState(0) for i in range(parent.childCount())}
        self.tree.blockSignals(True)
        parent.setCheckState(0, Qt.Checked if states == {Qt.Checked} else
                            (Qt.Unchecked if states == {Qt.Unchecked} else Qt.PartiallyChecked))
        self.tree.blockSignals(False)

    def _on_double(self, item: QTreeWidgetItem, column: int) -> None:
        key = item.data(0, LANE_KEY)
        if key:
            self.lane_activated.emit(key)
        else:
            self.source_activated.emit(item.data(0, SID))

    # ------------------------------------------------------------------
    def selected_sids(self) -> list[str]:
        out = []
        for item in self.tree.selectedItems():
            sid = item.data(0, SID)
            if sid and sid not in out:
                out.append(sid)
        return out

    def _on_merge(self) -> None:
        self.merge_clicked.emit(self.selected_sids())

    def _on_deint(self) -> None:
        sids = self.selected_sids()
        if sids:
            self.deinterleave_clicked.emit(sids[0])

    def _on_export(self) -> None:
        self.export_clicked.emit([l.key for l in self.project.visible_lanes()])

    def _on_remove(self) -> None:
        self.remove_clicked.emit(self.selected_sids())

    def _check_all(self, checked: bool) -> None:
        for top in self._source_items():
            top.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
            for i in range(top.childCount()):
                top.child(i).setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                self._set_lane_visible(top.child(i).data(0, LANE_KEY), checked)
        self.lanes_changed.emit()
        self._update_status()

    def checked_lane_keys(self) -> list[str]:
        return [l.key for l in self.project.visible_lanes()]
