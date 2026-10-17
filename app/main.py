"""入口：``python -m app.main [文件…]`` 或 ``pcm-analyzer [文件…]``。"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName("PCM Analyzer")
    app.setOrganizationName("QwenWork")
    app.setApplicationDisplayName("PCM / WAV 分析")

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
                  f"frames={win.project.total_frames}")
            app.quit()

        QTimer.singleShot(1200, _smoke)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
