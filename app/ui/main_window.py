"""主窗口：装配文件面板、时域波形、谱图、传输控制，负责交互与播放联动。"""

from __future__ import annotations

import os
import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QDockWidget,
)

from ..core import dsp, player as player_mod
from ..core.audio_io import PcmFormat
from ..core.model import Project
from . import theme
from .file_panel import FilePanel
from .overview_bar import OverviewBar
from .pcm_dialog import PcmFormatDialog
from .spectro_view import SpectroView
from .transport import TransportBar
from .view_state import ViewState
from .waveform_view import WaveView

AUDIO_EXT = (".wav", ".wave", ".pcm", ".raw", ".dat", ".bin", ".int16", ".int32", ".f32", ".aiff", ".flac", ".ogg")

HELP_TEXT = """
<h3>鼠标</h3>
<table>
<tr><td><b>滚轮</b></td><td>滑动时间窗口（触控板双指同向）</td></tr>
<tr><td><b>⌘ / Ctrl + 滚轮</b></td><td>时间维度放大（以光标位置为锚点）</td></tr>
<tr><td><b>⌥ Option / Shift + 滚轮</b></td><td>幅值放大</td></tr>
<tr><td><b>单击波形/谱图</b></td><td>定位播放位置</td></tr>
<tr><td><b>按住拖动</b></td><td>框选时间选区（可用于循环/缩放）</td></tr>
<tr><td><b>中键拖动</b></td><td>平移时间窗</td></tr>
<tr><td><b>双击</b></td><td>适配全曲；有选区时缩放到选区</td></tr>
<tr><td><b>右键</b></td><td>适配 / 缩放到选区 / 复制图像</td></tr>
</table>
<h3>键盘</h3>
<table>
<tr><td><b>空格</b></td><td>播放 / 暂停</td></tr>
<tr><td><b>S</b></td><td>停止</td></tr>
<tr><td><b>← / →</b></td><td>微调窗口；按住 Shift 幅度更大</td></tr>
<tr><td><b>, / .</b></td><td>播放位置后退 / 前进一帧（Zoom 时更细）</td></tr>
<tr><td><b>+ / -</b></td><td>时间放大 / 缩小</td></tr>
<tr><td><b>A / Z</b></td><td>幅值放大 / 缩小</td></tr>
<tr><td><b>0</b></td><td>适配全曲</td></tr>
<tr><td><b>L</b></td><td>循环开关</td></tr>
<tr><td><b>⌘O / ⌘W</b></td><td>打开 / 关闭全部</td></tr>
</table>
<h3>多通道工作流</h3>
<ul>
<li>左栏每个文件展开后就是<b>解交织</b>后的单声道列表，勾上/取消即控制显示与播放。</li>
<li>⌘/Shift 多选文件 → <b>合并所选文件通道</b>：生成一个新的多声道源（可整体播放、可导出成一个多声道 WAV）。</li>
<li><b>导出勾选通道</b>：把当前勾选的通道合成一个 WAV；<b>解交织导出</b>：每个通道写成独立单声道 WAV。</li>
<li>裸 PCM 用弹窗指定参数，参数会自动记入「最近」，也可存为预设。</li>
</ul>
"""


class MainWindow(QMainWindow):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.project = Project()
        self.view = ViewState(self)
        self.player = player_mod.Player(self.project)
        self.setWindowTitle("PCM / WAV 多通道播放器与分析")
        self.setAcceptDrops(True)
        self.resize(1440, 900)

        self.wave = WaveView(self.view, self.project)
        self.spectro = SpectroView(self.view, self.project)
        self.spectro.start()
        self.overview = OverviewBar(self.view, self.project)
        self.overview.start()
        self.panel = FilePanel(self.project)
        self.transport = TransportBar()

        self._build_dock()
        self._build_central()
        self._build_transport_bar()
        self._build_menus()
        self._build_toolbars()
        self._connect()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self._restore()
        self._refresh_all()
        self.setStatusBar(self.statusBar())
        self.statusBar().showMessage("打开 PCM / WAV 文件开始分析")

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def _build_dock(self) -> None:
        dock = QDockWidget("文件与通道", self)
        dock.setObjectName("fileDock")
        dock.setWidget(self.panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.dock = dock

    def _build_central(self) -> None:
        self.split = QSplitter(Qt.Vertical)
        self.split.addWidget(self.wave)
        self.split.addWidget(self._spectro_box())
        self.split.setSizes([560, 300])
        self.split.setCollapsible(0, False)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)
        lay.addWidget(self.overview)
        lay.addWidget(self.split, 1)
        self.setCentralWidget(central)

    def _spectro_box(self) -> QWidget:
        from PySide6.QtWidgets import QHBoxLayout
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        bar = QWidget()
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(2, 2, 2, 2)
        h = QHBoxLayout()
        h.setContentsMargins(2, 0, 2, 0)
        h.addWidget(QLabel("谱图通道"))
        self.spectro_lane = QComboBox()
        self.spectro_lane.setMinimumWidth(240)
        self.spectro_lane.currentIndexChanged.connect(self._on_spectro_lane)
        h.addWidget(self.spectro_lane)
        h.addWidget(QLabel("FFT"))
        self.spectro_nfft = QComboBox()
        for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
            self.spectro_nfft.addItem(str(n), n)
        self.spectro_nfft.setCurrentIndex(3)
        self.spectro_nfft.currentIndexChanged.connect(lambda *_: self._apply_spectro_opts())
        h.addWidget(self.spectro_nfft)
        h.addWidget(QLabel("窗"))
        self.spectro_win = QComboBox()
        self.spectro_win.addItems(["hann", "hamming", "blackman", "rect"])
        self.spectro_win.currentIndexChanged.connect(lambda *_: self._apply_spectro_opts())
        h.addWidget(self.spectro_win)
        h.addWidget(QLabel("色标"))
        self.spectro_cmap = QComboBox()
        self.spectro_cmap.addItems(["viridis", "inferno", "turbo", "grayscale"])
        self.spectro_cmap.currentIndexChanged.connect(lambda *_: self._apply_spectro_opts())
        h.addWidget(self.spectro_cmap)
        h.addWidget(QLabel("底 dB"))
        self.spectro_floor = QSpinBox()
        self.spectro_floor.setRange(-140, 0)
        self.spectro_floor.setValue(-90)
        self.spectro_floor.valueChanged.connect(lambda *_: self._apply_spectro_opts())
        h.addWidget(self.spectro_floor)
        self.spectro_log = QCheckBox("对数频率")
        self.spectro_log.toggled.connect(lambda *_: self._apply_spectro_opts())
        h.addWidget(self.spectro_log)
        h.addStretch(1)
        outer.addLayout(h)
        lay.addWidget(bar)
        lay.addWidget(self.spectro, 1)
        return box

    def _build_transport_bar(self) -> None:
        bar = QWidget()
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.transport)
        self.centralWidget().layout().addWidget(bar)

    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("文件(&F)")
        self._act(m_file, "打开…", "⌘O", self.choose_files)
        self._act(m_file, "以指定参数重新打开 PCM…", None, self.reopen_pcm)
        m_file.addSeparator()
        self._act(m_file, "导出勾选通道为 WAV…", "⌘E", lambda: self.export_lanes(self.project.visible_lanes()))
        self._act(m_file, "导出勾选通道为裸 PCM…", "⌘⇧E", lambda: self.export_pcm(self.project.visible_lanes()))
        self._act(m_file, "解交织导出选中文件…", "⌘⇧D", self.deinterleave_selected)
        m_file.addSeparator()
        self._act(m_file, "关闭全部", "⌘W", self.close_all)
        self._act(m_file, "退出", "⌘Q", self.close)

        m_view = self.menuBar().addMenu("视图(&V)")
        self._act(m_view, "适配全曲", "0", self.view.fit_all)
        self._act(m_view, "时间放大", "+", lambda: self.view.zoom_time(1.5))
        self._act(m_view, "时间缩小", "-", lambda: self.view.zoom_time(1 / 1.5))
        self._act(m_view, "幅值放大", "A", lambda: self.view.zoom_amp(1.4))
        self._act(m_view, "幅值缩小", "Z", lambda: self.view.zoom_amp(1 / 1.4))
        m_view.addSeparator()
        self._act(m_view, "缩放到选区", "⌘⇧S", self.view.zoom_to_selection)
        self._act(m_view, "清除选区", "Esc", self.view.clear_selection)

        m_help = self.menuBar().addMenu("帮助(&H)")
        self._act(m_help, "快捷键与工作流", "?", self.show_help)
        self._act(m_help, "关于", None, self.show_about)

    def _act(self, menu, text, shortcut, slot) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _build_toolbars(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction("打开", self.choose_files)
        tb.addSeparator()
        tb.addAction("时间+", lambda: self.view.zoom_time(2.0))
        tb.addAction("时间−", lambda: self.view.zoom_time(0.5))
        tb.addAction("幅值+", lambda: self.view.zoom_amp(1.5))
        tb.addAction("幅值−", lambda: self.view.zoom_amp(1 / 1.5))
        tb.addAction("适配", self.view.fit_all)
        tb.addSeparator()
        tb.addWidget(QLabel(" 布局 "))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["分道显示", "叠加显示"])
        self.cb_mode.currentIndexChanged.connect(self._apply_view_opts)
        tb.addWidget(self.cb_mode)
        tb.addWidget(QLabel(" 纵轴 "))
        self.cb_yscale = QComboBox()
        self.cb_yscale.addItems(["线性幅度", "dBFS"])
        self.cb_yscale.currentIndexChanged.connect(self._apply_view_opts)
        tb.addWidget(self.cb_yscale)
        tb.addWidget(QLabel(" 波形 "))
        self.cb_fill = QComboBox()
        self.cb_fill.addItems(["包络(min/max)", "RMS", "样本折线"])
        self.cb_fill.currentIndexChanged.connect(self._apply_view_opts)
        tb.addWidget(self.cb_fill)
        tb.addSeparator()
        tb.addAction("复制图像", self._copy_image)

        hint = QToolBar("提示")
        hint.setObjectName("hintToolbar")
        hint.setMovable(False)
        self.addToolBar(Qt.BottomToolBarArea, hint)
        self.hint_label = QLabel("   滚轮=滑动时间窗    ⌘+滚轮=时间放大    ⌥+滚轮=幅值放大    "
                                 "单击=定位播放    拖动=选区    空格=播放")
        self.hint_label.setObjectName("hint")
        hint.addWidget(self.hint_label)

    def _connect(self) -> None:
        self.panel.open_clicked.connect(self.choose_files)
        self.panel.merge_clicked.connect(self.merge_sources)
        self.panel.remove_clicked.connect(self.remove_sources)
        self.panel.export_clicked.connect(lambda keys: self.export_lanes(self.project.lanes(keys)))
        self.panel.deinterleave_clicked.connect(self.deinterleave_source)
        self.panel.lanes_changed.connect(self._on_lanes_changed)
        self.panel.lanes_changed.connect(self.overview.request)
        self.overview.interacted.connect(lambda *_: self._update_zoom_label())
        self.panel.lane_activated.connect(self._focus_lane)
        self.view.view_changed.connect(self._on_view_changed)
        self.view.selection_changed.connect(self._on_selection_changed)
        for w in (self.wave, self.spectro):
            w.seek_requested.connect(self.seek_frames)
            w.cursor_moved.connect(self._on_cursor)
            w.view_interacted.connect(self._update_zoom_label)
        self.transport.play.connect(self._on_play)
        self.transport.pause.connect(self._on_pause)
        self.transport.stop.connect(self._on_stop)
        self.transport.seek.connect(self.seek_frames)
        self.transport.loop_toggled.connect(self._on_loop)
        self.transport.device_changed.connect(lambda *_: self._rebind(True))
        self.transport.rate_changed.connect(self._on_rate)
        self.transport.follow_toggled.connect(lambda v: None)
        self.transport.volume_changed.connect(self.player.set_volume)

    # ------------------------------------------------------------------
    # 打开文件
    # ------------------------------------------------------------------
    def choose_files(self) -> None:
        start = self.config.get("last_dir") or os.path.expanduser("~")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "打开 PCM / WAV 文件", start,
            "音频 ( *.wav *.wave *.pcm *.raw *.dat *.bin *.int16 *.int32 *.f32 *.aiff *.flac *.ogg );;所有文件 (*)")
        if paths:
            self.config["last_dir"] = os.path.dirname(paths[0])
            self.open_paths(paths)

    def open_paths(self, paths: list[str]) -> None:
        opened = []
        problems = []
        for path in paths:
            fmt = None
            from ..core.audio_io import audio_kind
            if audio_kind(path) != "wav":
                dlg = PcmFormatDialog(self, path, self.config, fmt=None)
                if dlg.exec() != PcmFormatDialog.Accepted:
                    continue
                fmt = dlg.build_format()
            try:
                src = self.project.open_file(path, fmt)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{os.path.basename(path)}：{exc}")
                continue
            opened.append(src)
        if opened:
            self._after_open()
            names = "、".join(s.label for s in opened)
            self.statusBar().showMessage(f"已打开 {names}", 6000)
        if problems:
            QMessageBox.warning(self, "部分文件未能打开", "\n".join(problems))

    def reopen_pcm(self) -> None:
        sids = self.panel.selected_sids()
        if not sids:
            QMessageBox.information(self, "提示", "请先在左栏选中一个 PCM 源。")
            return
        src = self.project.sources[sids[0]]
        if not src.path:
            return
        dlg = PcmFormatDialog(self, src.path, self.config, self.project.buffer_of(src.sid).fmt)
        if dlg.exec() != PcmFormatDialog.Accepted:
            return
        path = src.path
        self.project.remove_sources([src.sid])
        self.open_paths([path])
        self._after_open()

    def close_all(self) -> None:
        self.player.stop()
        self.project = Project()
        self.player.project = self.project
        self.wave.project = self.project
        self.spectro.project = self.project
        self.panel.project = self.project
        self._after_open()

    def _after_open(self) -> None:
        rate = self.project.sample_rate or 48000
        self.view.set_total(self.project.total_frames, rate)
        self.spectro.opts.update(self.config.spectro)
        self._sync_spectro_controls()
        self._rebind(True)
        self._refresh_all()

    def _refresh_all(self) -> None:
        self.panel.sync()
        self._fill_spectro_lanes()
        self.wave._cache_key = None
        self.spectro._current = None
        self.overview.request()
        self.wave.update()
        self.spectro._request()
        self._update_zoom_label()

    # ------------------------------------------------------------------
    # 通道 / 播放绑定
    # ------------------------------------------------------------------
    def _on_lanes_changed(self) -> None:
        self._rebind(False)
        self._fill_spectro_lanes()
        self._update_zoom_label()

    def _rebind(self, restart: bool) -> None:
        rate = self.project.sample_rate or self.transport.current_rate() or 48000
        lanes = self.project.visible_lanes()
        self.view.set_total(self.project.total_frames, rate)
        self.transport.set_total(self.project.total_frames, rate)
        self.transport.set_rates(self.project.available_rates() or [rate], rate)
        chans, warn = self.player.prepare(lanes, rate, self.transport.current_device())
        if warn:
            self.statusBar().showMessage("⚠ " + warn, 8000)
        self.player.set_volume(self.transport.volume.value() / 100.0)
        if restart and self.player.is_playing():
            self.player.play()

    def _on_rate(self, rate: int) -> None:
        self.project.set_sample_rate(rate)
        self.view.set_total(self.project.total_frames, rate, keep_window=True)
        self._rebind(True)

    # ------------------------------------------------------------------
    # 合并 / 导出
    # ------------------------------------------------------------------
    def merge_sources(self, sids: list[str]) -> None:
        if len(sids) < 1:
            QMessageBox.information(self, "提示", "请在左栏用 ⌘/Shift 多选要合并的文件（至少 1 个）。")
            return
        lanes: list = []
        for sid in sids:
            src = self.project.sources.get(sid)
            if src:
                lanes.extend([l for l in src.lanes if l.visible] or src.lanes)
        names = "+".join(self.project.sources[s].label for s in sids if s in self.project.sources)
        try:
            mix = self.project.create_mix(f"Merge({names})", lanes)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "合并失败", str(exc))
            return
        self._after_open()
        self.statusBar().showMessage(
            f"已合并 {len(lanes)} 个通道 → {mix.label}（原始通道已取消勾选以免重复播放，勾回即可）", 9000)

    def remove_sources(self, sids: list[str]) -> None:
        if not sids:
            return
        self.player.pause()
        self.project.remove_sources(sids)
        self._after_open()

    def _as_lanes(self, keys_or_lanes) -> list:
        if not keys_or_lanes:
            return []
        if isinstance(keys_or_lanes[0], str):
            return self.project.lanes(keys_or_lanes)
        return list(keys_or_lanes)

    def export_lanes(self, keys_or_lanes) -> None:
        lanes = self._as_lanes(keys_or_lanes)
        if not lanes:
            QMessageBox.information(self, "提示", "没有勾选任何通道。")
            return
        start = self.config.get("last_dir") or os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(self, "导出多声道 WAV",
                                              os.path.join(start, "merged.wav"), "WAV (*.wav)")
        if not path:
            return
        try:
            frames, chans = self.project.export_lanes(path, lanes, self.project.sample_rate)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", f"{exc}\n{traceback.format_exc()}")
            return
        self.statusBar().showMessage(f"已导出 {frames:,} 帧 × {chans} 通道 → {os.path.basename(path)}", 8000)
        self.config["last_dir"] = os.path.dirname(path)

    def export_pcm(self, keys_or_lanes) -> None:
        lanes = self._as_lanes(keys_or_lanes)
        if not lanes:
            QMessageBox.information(self, "提示", "没有勾选任何通道。")
            return
        start = self.config.get("last_dir") or os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(self, "导出裸 PCM", os.path.join(start, "out.pcm"), "PCM (*.pcm)")
        if not path:
            return
        fmt = PcmFormat(sample_rate=self.project.sample_rate or 48000, channels=len(lanes),
                        sample_width=2, is_float=False)
        dlg = PcmFormatDialog(self, path, self.config, fmt)
        if dlg.exec() != PcmFormatDialog.Accepted:
            return
        fmt = dlg.build_format()
        fmt.channels = len(lanes)
        try:
            nbytes = self.project.export_pcm(path, lanes, fmt)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.statusBar().showMessage(f"已导出 {nbytes:,} 字节 → {os.path.basename(path)}（{fmt.describe()}）", 8000)

    def deinterleave_selected(self) -> None:
        sids = self.panel.selected_sids()
        if sids:
            self.deinterleave_source(sids[0])

    def deinterleave_source(self, sid: str) -> None:
        src = self.project.sources.get(sid)
        if src is None:
            return
        start = os.path.dirname(src.path) if src.path else (self.config.get("last_dir") or os.path.expanduser("~"))
        outdir = QFileDialog.getExistingDirectory(self, f"解交织导出 {src.label} 的 {src.channel_count} 个通道", start)
        if not outdir:
            return
        try:
            files = self.project.export_deinterleaved(outdir, sid)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "解交织完成", f"已写出 {len(files)} 个单声道文件：\n" + "\n".join(
            os.path.basename(f) for f in files[:12]))

    # ------------------------------------------------------------------
    # 播放控制
    # ------------------------------------------------------------------
    def _on_play(self) -> None:
        if self.player.is_playing():
            self.player.pause()
            self.transport.set_playing(False)
            return
        ok = self.player.play()
        self.transport.set_playing(ok)
        if not ok:
            QMessageBox.warning(self, "无法播放", self.player.error or "未知错误")

    def _on_pause(self) -> None:
        self.player.pause()
        self.transport.set_playing(False)

    def _on_stop(self) -> None:
        self.player.stop()
        self.transport.set_playing(False)
        self._sync_playhead(0)

    def seek_frames(self, frames: int) -> None:
        self.player.seek(frames)
        self._sync_playhead(frames)
        if self.transport.btn_follow.isChecked():
            self.view.ensure_visible(frames / max(1, self.view.rate))

    def _on_loop(self, enabled: bool) -> None:
        if enabled:
            v = self.view
            if v.has_selection:
                self.player.set_loop((v.sel_start, v.sel_end))
            else:
                self.player.set_loop((0, self.project.total_frames))
        else:
            self.player.set_loop(None)

    def _on_selection_changed(self) -> None:
        if self.transport.btn_loop.isChecked() and self.view.has_selection:
            self.player.set_loop((self.view.sel_start, self.view.sel_end))
        v = self.view
        if v.has_selection:
            dur = (v.sel_end - v.sel_start) / max(1, v.rate)
            self.statusBar().showMessage(
                f"选区 {dsp.format_time(v.sel_start / max(1, v.rate))} → "
                f"{dsp.format_time(v.sel_end / max(1, v.rate))}（{dur * 1000:.1f} ms）", 4000)
        self.wave.update()
        self.spectro.update()

    def _tick(self) -> None:
        if self.player.is_playing():
            self._sync_playhead(self.player.position)
            if self.transport.btn_follow.isChecked():
                t = self.player.position_seconds
                v = self.view
                if t >= v.t0 + v.win * 0.98 or t < v.t0:
                    v.set_window(t - v.win * 0.05, v.win)
            if self.player.error:
                QMessageBox.warning(self, "播放错误", self.player.error)
                self.player.error = None
                self.transport.set_playing(False)
            self.transport.set_playing(self.player.is_playing())
        else:
            self.transport.set_position(self.player.position, self.project.total_frames)

    def _sync_playhead(self, frames: int) -> None:
        rate = max(1, self.view.rate)
        self.wave.playhead = frames / rate
        self.spectro.playhead = frames / rate
        self.overview.set_playhead(frames / rate)
        self.overview.update()
        self.transport.set_position(frames, self.project.total_frames)
        self.wave.update()
        self.spectro.update()

    # ------------------------------------------------------------------
    # 视图选项
    # ------------------------------------------------------------------
    def _apply_view_opts(self, *_):
        self.wave.set_options(
            mode=("lanes" if self.cb_mode.currentIndex() == 0 else "overlay"),
            y_scale=("linear" if self.cb_yscale.currentIndex() == 0 else "db"),
            fill=("envelope" if self.cb_fill.currentIndex() == 0 else
                  ("rms" if self.cb_fill.currentIndex() == 1 else "line")),
        )
        self._save_view_opts()

    def _save_view_opts(self) -> None:
        self.config.view = {**self.config.view,
                            "mode": self.wave.mode, "y_scale": self.wave.y_scale,
                            "fill": self.wave.fill, "time_zoom": self.view.win}

    def _apply_spectro_opts(self, *_):
        self.spectro.set_options(
            nfft=int(self.spectro_nfft.currentData()),
            window=self.spectro_win.currentText(),
            colormap=self.spectro_cmap.currentText(),
            floor_db=float(self.spectro_floor.value()),
            freq_log=self.spectro_log.isChecked(),
        )
        self.config.spectro = {**self.config.spectro,
                               "nfft": int(self.spectro_nfft.currentData()),
                               "window": self.spectro_win.currentText(),
                               "colormap": self.spectro_cmap.currentText(),
                               "floor_db": float(self.spectro_floor.value()),
                               "freq_log": self.spectro_log.isChecked()}

    def _sync_spectro_controls(self) -> None:
        opts = self.spectro.opts
        idx = self.spectro_nfft.findData(int(opts.get("nfft", 2048)))
        self.spectro_nfft.setCurrentIndex(max(0, idx))
        self.spectro_win.setCurrentText(str(opts.get("window", "hann")))
        self.spectro_cmap.setCurrentText(str(opts.get("colormap", "viridis")))
        self.spectro_floor.setValue(int(opts.get("floor_db", -90)))
        self.spectro_log.setChecked(bool(opts.get("freq_log", False)))

    def _fill_spectro_lanes(self) -> None:
        cur = self.spectro.lane_key
        self.spectro_lane.blockSignals(True)
        self.spectro_lane.clear()
        for lane in self.project.all_lanes():
            src = self.project.sources.get(lane.source_id)
            label = f"{src.label} · {lane.label}" if src else lane.label
            self.spectro_lane.addItem(label, lane.key)
        if cur:
            idx = self.spectro_lane.findData(cur)
            if idx >= 0:
                self.spectro_lane.setCurrentIndex(idx)
        elif self.spectro_lane.count():
            first_visible = next((l for l in self.project.visible_lanes()), None)
            if first_visible:
                idx = self.spectro_lane.findData(first_visible.key)
                self.spectro_lane.setCurrentIndex(max(0, idx))
        self.spectro_lane.blockSignals(False)
        self._on_spectro_lane()

    def _on_spectro_lane(self, *_):
        key = self.spectro_lane.currentData()
        if key:
            self.spectro.set_lane(key)

    def _focus_lane(self, key: str) -> None:
        idx = self.spectro_lane.findData(key)
        if idx >= 0:
            self.spectro_lane.setCurrentIndex(idx)
        self.statusBar().showMessage(f"谱图通道切换为 {key}", 3000)

    def _on_view_changed(self, reason: str) -> None:
        self._update_zoom_label()
        if reason in ("time", "range"):
            self._save_view_opts()
        if reason == "range":
            self.overview.request()       # 总时长变了，全曲包络要重算

    def _update_zoom_label(self) -> None:
        v = self.view
        self.setWindowTitle(f"PCM / WAV 多通道播放器与分析 — 窗口 {v.win * 1000:,.2f} ms · 幅值 ×{v.amp:g}")
        # 提示语已在底部工具栏常驻，状态栏只放通道统计，避免同一句话出现两遍
        self.statusBar().showMessage(self.wave.stats_text())

    def _on_cursor(self, seconds: float, frames: int) -> None:
        rate = max(1, self.view.rate)
        self.statusBar().showMessage(
            f"光标 {dsp.format_time(seconds, True)}  帧 {frames:,}  "
            f"({frames % max(1, rate) / rate * 1000:.3f} ms 偏移)   " + (self.wave.stats_text() or ""))

    def _copy_image(self) -> None:
        pix = self.split.grab()
        QApplication.clipboard().setPixmap(pix)
        self.statusBar().showMessage("已把当前视图复制到剪贴板", 3000)

    # ------------------------------------------------------------------
    # 拖放 / 快捷键 / 生命周期
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self.open_paths(paths)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Space:
            self._on_play()
        elif key == Qt.Key_S:
            self._on_stop()
        elif key == Qt.Key_L:
            self.transport.btn_loop.click()
        elif key == Qt.Key_Comma:
            self.seek_frames(max(0, self.player.position - 1))
        elif key == Qt.Key_Period:
            self.seek_frames(self.player.position + 1)
        elif key == Qt.Key_Home:
            self.seek_frames(0)
        elif key == Qt.Key_End:
            self.seek_frames(max(0, self.project.total_frames - 1))
        else:
            super().keyPressEvent(event)

    def _restore(self) -> None:
        geo = self.config.get("geometry")
        if isinstance(geo, list) and len(geo) == 4:
            self.setGeometry(*[int(v) for v in geo])
        self.cb_mode.setCurrentIndex(0 if self.config.view.get("mode", "lanes") == "lanes" else 1)
        self.cb_yscale.setCurrentIndex(0 if self.config.view.get("y_scale", "linear") == "linear" else 1)
        fills = {"envelope": 0, "rms": 1, "line": 2}
        self.cb_fill.setCurrentIndex(fills.get(self.config.view.get("fill", "envelope"), 0))
        self._apply_view_opts()
        devices = player_mod.list_output_devices()
        self.transport.set_devices(devices, self.config.get("device"))
        self.transport.volume.setValue(int(self.config.get("volume", 1.0) * 100))
        self._sync_spectro_controls()
        self._apply_spectro_opts()

    def closeEvent(self, event) -> None:  # noqa: N802
        g = self.geometry()
        self.config["geometry"] = [g.x(), g.y(), g.width(), g.height()]
        self.config["volume"] = self.player.volume
        self.config["device"] = self.transport.current_device()
        self.config["last_dir"] = self.config.get("last_dir")
        self.timer.stop()
        self.spectro.shutdown()
        self.overview.shutdown()
        self.player.close()
        super().closeEvent(event)

    def show_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("快捷键与工作流")
        box.setTextFormat(Qt.RichText)
        box.setText(HELP_TEXT)
        box.exec()

    def show_about(self) -> None:
        QMessageBox.about(self, "关于", "PCM / WAV 多通道播放器与分析\n\n"
                                        "· 裸 PCM 支持 8/16/24/32bit、float32/64、大小端、planar、文件头偏移\n"
                                        "· 多通道播放走 CoreAudio（PortAudio），可选输出设备与通道数\n"
                                        "· 大文件用内存映射 + 分块抽稀，不整段读入内存")
