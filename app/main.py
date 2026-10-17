"""入口：``python -m app.main [文件…]`` 或 ``pcm-analyzer [文件…]``。"""

from __future__ import annotations

import sys


def _apply_icon(app) -> str:
    """给 Dock / 窗口标题栏设图标。图标是 tools/make_icon.py 的产物，缺失时静默降级。

    .app 由 Info.plist 的 CFBundleIconFile 决定图标，setWindowIcon 只影响 Dock 与
    窗口本身——用 run.sh 直接跑 python 时没有 bundle，就靠它。
    """
    from pathlib import Path

    from PySide6.QtGui import QIcon

    root = Path(__file__).resolve().parent.parent
    for cand in (root / "assets" / "AppIcon.icns",
                 root / "assets" / "icon.iconset" / "icon_256x256.png",
                 root / "PCMAnalyzer.app" / "Contents" / "Resources" / "AppIcon.icns"):
        if cand.exists():
            icon = QIcon(str(cand))
            if not icon.isNull():
                app.setWindowIcon(icon)
                return f"{cand.name} ({len(icon.availableSizes())} 档)"
    return "无（跑 tools/make_icon.py 生成）"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from PySide6.QtWidgets import QApplication

    # Qt6 默认已启用高分屏缩放与高 DPI 位图，再显式 setAttribute 会打 DeprecationWarning
    app = QApplication(argv)
    app.setApplicationName("PCM Analyzer")
    app.setOrganizationName("QwenWork")
    app.setApplicationDisplayName("PCM / WAV 分析")
    icon_src = _apply_icon(app)

    from .core.config import Config
    from .ui import theme
    from .ui.main_window import MainWindow

    theme.apply_dark_theme(app)
    config = Config()
    win = MainWindow(config)
    win.show()

    paths = [a for a in argv if not a.startswith("-")]
    if paths:
        win.open_paths(paths)

    # 冒烟自检：PCM_ANALYZER_SMOKE=1 时开窗口、跑一圈事件循环后自动退出
    import os

    if os.environ.get("PCM_ANALYZER_SMOKE"):
        from PySide6.QtCore import QTimer

        def _smoke() -> None:
            print(f"SMOKE ok window={win.isVisible()} lanes={len(win.project.all_lanes())} "
                  f"frames={win.project.total_frames} icon={icon_src} "
                  f"windowIconNull={win.windowIcon().isNull()}")
            app.quit()

        QTimer.singleShot(1200, _smoke)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
