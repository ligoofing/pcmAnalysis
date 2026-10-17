"""应用配置：PCM 参数历史、预设、界面状态。存到 macOS 标准 Application Support。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

APP_NAME = "PCMAnalyzer"
MAX_HISTORY = 20

DEFAULTS: dict[str, Any] = {
    "pcm_history": [],          # [{...PcmFormat, "name"}] 最近使用在前
    "pcm_presets": [],          # 用户保存的常用参数
    "last_dir": str(Path.home()),
    "geometry": None,
    "splitter": None,
    "device": None,
    "volume": 1.0,
    "sample_rate_override": None,
    "spectro": {
        "nfft": 2048,
        "window": "hann",
        "colormap": "viridis",
        "floor_db": -90.0,
        "top_db": 0.0,
        "freq_log": False,
    },
    "view": {
        "mode": "lanes",        # lanes | overlay
        "y_scale": "linear",    # linear | db
        "fill": "envelope",     # envelope | rms | line
        "time_zoom": 0.02,      # 初始窗口占全曲比例
    },
}


def config_dir() -> Path:
    base = Path(os.environ.get("HOME", str(Path.home()))) / "Library" / "Application Support" / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


class Config:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (config_dir() / "config.json")
        self.data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        for k, v in raw.items():
            if k in ("spectro", "view") and isinstance(v, dict):
                merged = dict(DEFAULTS[k])
                merged.update(v)
                self.data[k] = merged
            else:
                self.data[k] = v

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), "utf-8")
        except OSError:
            pass

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()

    def get(self, k, default=None):
        return self.data.get(k, default)

    # ---- PCM 参数历史 ----------------------------------------------------
    def remember_pcm_format(self, fmt_dict: dict[str, Any], name: str = "") -> None:
        entry = dict(fmt_dict)
        entry["name"] = name or f"{entry.get('sample_rate')}Hz {entry.get('channels')}ch {entry.get('sample_width')}B"
        hist = [h for h in self.data["pcm_history"] if not _same_fmt(h, entry)]
        hist.insert(0, entry)
        self.data["pcm_history"] = hist[:MAX_HISTORY]
        self.save()

    def add_preset(self, fmt_dict: dict[str, Any], name: str) -> None:
        entry = dict(fmt_dict)
        entry["name"] = name
        presets = [p for p in self.data["pcm_presets"] if p.get("name") != name]
        presets.insert(0, entry)
        self.data["pcm_presets"] = presets[:MAX_HISTORY]
        self.save()

    def remove_preset(self, name: str) -> None:
        self.data["pcm_presets"] = [p for p in self.data["pcm_presets"] if p.get("name") != name]
        self.save()

    @property
    def spectro(self) -> dict[str, Any]:
        return self.data["spectro"]

    @spectro.setter
    def spectro(self, v: dict[str, Any]) -> None:
        self.data["spectro"] = v
        self.save()

    @property
    def view(self) -> dict[str, Any]:
        return self.data["view"]

    @view.setter
    def view(self, v: dict[str, Any]) -> None:
        self.data["view"] = v
        self.save()


_KEYS = ("sample_rate", "channels", "sample_width", "is_float", "signed", "big_endian", "planar")


def _same_fmt(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(k) == b.get(k) for k in _KEYS)
