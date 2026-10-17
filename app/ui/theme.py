"""深色主题配色与全局样式。"""

from __future__ import annotations

BG = "#14161a"
PANEL = "#1b1e25"
PANEL_ALT = "#22262f"
BORDER = "#2e333d"
GRID = "#272b34"
GRID_MAJOR = "#3a4150"
TEXT = "#dfe3ea"
TEXT_DIM = "#8b93a7"
ACCENT = "#4fb3ff"
PLAYHEAD = "#ff5d5d"
SELECT = "#4fb3ff"
ZERO = "#4a5262"

DARK_QSS = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 12px;
}}
QMainWindow::separator {{ background: {BORDER}; width: 1px; height: 1px; }}
QTreeWidget, QListWidget, QTreeView, QListView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    outline: 0;
}}
QTreeWidget::item, QListWidget::item {{ padding: 2px 0; }}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {PANEL_ALT};
    color: {TEXT};
}}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: #20242d; }}
QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 10px;
}}
QPushButton:hover {{ background: #2b303b; }}
QPushButton:pressed {{ background: #191c23; }}
QPushButton:disabled {{ color: {TEXT_DIM}; }}
QPushButton:checked {{ background: #24405a; border-color: {ACCENT}; }}
QToolButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 8px;
}}
QToolButton:hover {{ background: #2b303b; }}
QToolButton:checked {{ background: #24405a; border-color: {ACCENT}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 5px;
    selection-background-color: {ACCENT};
}}
QComboBox::drop-down {{ border: 0; width: 16px; }}
QComboBox QAbstractItemView {{
    background: {PANEL};
    border: 1px solid {BORDER};
    selection-background-color: {PANEL_ALT};
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 12px;
    padding-top: 6px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {TEXT_DIM}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px; margin: -5px 0; border-radius: 6px; background: {TEXT};
}}
QScrollBar:horizontal {{ height: 12px; background: {PANEL}; }}
QScrollBar:vertical {{ width: 12px; background: {PANEL}; }}
QScrollBar::handle {{ background: #3a4150; border-radius: 5px; min-height: 20px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QStatusBar {{ background: {PANEL}; color: {TEXT_DIM}; }}
QStatusBar::item {{ border: 0; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {PANEL_ALT}; }}
QLabel#hint {{ color: {TEXT_DIM}; }}
QLabel#title {{ font-size: 13px; font-weight: 600; }}
"""


def apply_dark_theme(app) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette, QColor

    app.setStyleSheet(DARK_QSS)
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.Base, QColor(PANEL))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL_ALT))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#0c0d10"))
    pal.setColor(QPalette.ToolTipBase, QColor(PANEL))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    app.setPalette(pal)
    app.setStyle("Fusion")
