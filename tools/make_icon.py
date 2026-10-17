"""生成应用图标：深色瓷砖 + 频谱柱 + 波形线 + 红色播放指针。

为什么不画一张 1024 大图再缩放：栅格缩放会让 16/32px 的线条发灰、细节糊成一团。
这里每个尺寸都**按归一化坐标重新绘制**，线宽取「比例」与「最小可辨像素」的较大值，
小尺寸自动减少柱子数量并去掉装饰，保证 Dock/程序坞/关于框里都能看清。

用法：
    .venv/bin/python tools/make_icon.py            # 生成 assets/icon.iconset + AppIcon.icns
    QT_QPA_PLATFORM=offscreen .venv/bin/python tools/make_icon.py --preview
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QIcon, QImage, QLinearGradient, QPainter, QPen, QPolygonF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
ICONSET = os.path.join(ASSETS, "icon.iconset")
ICNS = os.path.join(ASSETS, "AppIcon.icns")

# Apple 要求的尺寸集合：(文件名, 像素边长)
SIZES = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

WAVE_COLOR = "#eaf2ff"
PLAYHEAD_COLOR = "#ff5d5d"


def draw(p: QPainter, S: int) -> None:
    """在 S×S 画布上按归一化坐标绘制，并按尺寸分级简化元素。

    小尺寸不是「缩小」而是「换一套构图」：16/32px 只留 3 根柱 + 2 个周期的波形，
    去掉播放指针，否则线条会互相糊成灰块。
    """
    p.setRenderHint(QPainter.Antialiasing, True)

    tier = 0 if S <= 40 else (1 if S <= 96 else 2)     # 0=极小 1=中 2=大
    n_bars = (3, 5, 7)[tier]
    cycles = (1.75, 2.25, 3.0)[tier]
    heights = [0.42, 0.78, 0.30, 0.60, 0.36, 0.68, 0.48][:n_bars]
    colors = ["#4fb3ff", "#5ddb7a", "#ff8a4c", "#f2d64b", "#c78bff", "#4ce0d8", "#ff6b8b"]

    margin = 0.055 * S
    tile = QRectF(margin, margin, S - 2 * margin, S - 2 * margin)
    radius = 0.225 * tile.width()

    # 底板：上亮下暗的垂直渐变
    grad = QLinearGradient(tile.topLeft(), tile.bottomLeft())
    grad.setColorAt(0.0, QColor("#2c3342"))
    grad.setColorAt(0.5, QColor("#1a1e27"))
    grad.setColorAt(1.0, QColor("#0b0d11"))
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(tile, radius, radius)

    # 内容区：上下留白按尺寸收紧，避免大尺寸底部空一片
    inner = tile.adjusted(tile.width() * 0.13, tile.height() * 0.16,
                          -tile.width() * 0.13, -tile.height() * (0.10 if tier else 0.14))

    # 频谱柱
    step = inner.width() / n_bars
    bw = step * (0.58 if tier == 0 else 0.44)
    base = inner.bottom()
    p.setPen(Qt.NoPen)
    for i in range(n_bars):
        h = heights[i] * inner.height() * 0.62
        x = inner.left() + i * step + (step - bw) / 2.0
        col = QColor(colors[i % len(colors)])
        col.setAlpha(200 if tier else 230)
        p.setBrush(col)
        r = min(bw / 2.0, 0.045 * S)
        p.drawRoundedRect(QRectF(x, base - h, bw, h), r, r)

    # 波形线：主角
    amp = inner.height() * 0.15
    yc = inner.top() + inner.height() * 0.34
    n = max(64, S * 2)
    pts = [QPointF(inner.left() + (k / n) * inner.width(),
                   yc + amp * _wave(k / n, cycles)) for k in range(n + 1)]
    w = max(2.0, min(0.030 * S, inner.height() * 0.16))
    p.setPen(QPen(QColor(WAVE_COLOR), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawPolyline(QPolygonF(pts))

    # 播放指针：只在大/中尺寸出现，且刻意细、半透明——它是点缀，不能抢波形
    if tier:
        ph = inner.left() + inner.width() * (0.70 if tier == 2 else 0.66)
        col = QColor(PLAYHEAD_COLOR)
        col.setAlpha(225)
        p.setPen(QPen(col, max(1.2, 0.011 * S), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(ph, inner.top() - amp * 1.1), QPointF(ph, base))

    # 内描边高光：浅色桌面上不显脏
    p.setPen(QPen(QColor(255, 255, 255, 36), max(1.0, 0.010 * S)))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(tile.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


def _wave(f: float, cycles: float) -> float:
    """归一化波形：主正弦 + 少量二次谐波，比纯正弦更像真实音频。"""
    t = f * cycles * 2 * math.pi
    return math.sin(t) * 0.88 + math.sin(t * 2 + 0.7) * 0.12


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    draw(p, size)
    p.end()
    return img


def build() -> list[str]:
    os.makedirs(ICONSET, exist_ok=True)
    written = []
    for name, px in SIZES:
        path = os.path.join(ICONSET, name)
        img = render(px)
        if not img.save(path):
            raise RuntimeError(f"写入失败：{path}")
        written.append(path)
    return written


def make_icns() -> bool:
    """调用系统 iconutil 打包 .icns。"""
    try:
        subprocess.run(["iconutil", "-c", "icns", ICONSET, "-o", ICNS], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"iconutil 打包失败：{exc}（PNG 仍可用于窗口图标）")
        return False


def preview() -> None:
    """把几个关键尺寸拼成一张对照图，便于肉眼检查小尺寸是否糊。"""
    from PySide6.QtGui import QPixmap
    sizes = [16, 32, 64, 128, 256, 512]
    gap = 16
    total_w = sum(sizes) + gap * (len(sizes) + 1)
    canvas = QImage(total_w, max(sizes) + 2 * gap, QImage.Format_ARGB32_Premultiplied)
    canvas.fill(QColor("#c9ccd3"))          # 浅灰桌面背景，检验边缘是否发脏
    p = QPainter(canvas)
    x = gap
    for s in sizes:
        p.drawImage(QRectF(x, gap, s, s), render(s))
        x += s + gap
    p.end()
    out = os.path.join(ASSETS, "icon-preview.png")
    canvas.save(out)
    print(f"对照图：{out}（浅灰底，16/32/64/128/256/512 并排）")
    ic = QIcon()
    for s in (16, 32, 128, 256, 512):
        ic.addPixmap(QPixmap.fromImage(render(s)))
    print(f"QIcon 可加载：{not ic.isNull()}，可用尺寸 {len(ic.availableSizes())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="只输出多尺寸对照图")
    ap.add_argument("--no-icns", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841

    os.makedirs(ASSETS, exist_ok=True)
    if args.preview:
        preview()
        return 0
    files = build()
    print(f"已生成 {len(files)} 张 PNG → {ICONSET}")
    ok = True
    if not args.no_icns:
        ok = make_icns()
        if ok:
            print(f"已打包 {ICNS}（{os.path.getsize(ICNS) / 1024:.0f} KB）")
    preview()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
