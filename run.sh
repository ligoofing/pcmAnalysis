#!/bin/bash
# PCM / WAV 分析器启动脚本。用法：./run.sh [文件...]
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "首次运行：创建虚拟环境并安装依赖…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
    uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
  else
    python3 -m venv "$VENV"
    "$PY" -m pip install -U pip
    "$PY" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy soundfile sounddevice PySide6
  fi
fi

# 让 python -m app.main 找到包
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
exec "$PY" -m app.main "$@"
