#!/bin/bash
# 生成可双击的 PCMAnalyzer.app（包装 venv 里的 python，不打包解释器）。
# 用法：./make_app.sh    产物：./PCMAnalyzer.app
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
APP="PCMAnalyzer.app"
BIN="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "缺少 .venv，请先执行 ./run.sh 完成依赖安装" >&2
  exit 1
fi

# 旧产物不删除，改名保留（可随时恢复或自行清理）
if [[ -e "$APP" ]]; then
  BACKUP="$APP.bak-$(date +%Y%m%d-%H%M%S)"
  mv "$APP" "$BACKUP"
  echo "已有产物改名保留为 $BACKUP"
fi
mkdir -p "$BIN" "$RES"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>PCM Analyzer</string>
  <key>CFBundleDisplayName</key><string>PCM / WAV 分析</string>
  <key>CFBundleIdentifier</key><string>cn.qwenwork.pcm-analyzer</string>
  <key>CFBundleVersion</key><string>1.0.0</string>
  <key>CFBundleShortVersionString</key><string>1.0.0</string>
  <key>CFBundleExecutable</key><string>PCMAnalyzer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFTypeName</key><string>Audio</string>
      <key>LSItemContentTypes</key>
      <array>
        <string>public.audiovisual-content</string>
        <string>public.data</string>
        <string>public.item</string>
      </array>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>CFBundleTypeExtensions</key>
      <array>
        <string>wav</string><string>pcm</string><string>raw</string>
        <string>dat</string><string>bin</string><string>f32</string><string>int16</string>
      </array>
    </dict>
  </array>
</dict>
</plist>
PLIST

cat > "$BIN/PCMAnalyzer" <<LAUNCH
#!/bin/bash
# 由 make_app.sh 生成：项目根目录 = $ROOT
export PYTHONPATH="$ROOT:\${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m app.main "\$@" 2>>"\$HOME/Library/Logs/pcm-analyzer.log"
LAUNCH
chmod +x "$BIN/PCMAnalyzer"

touch "$RES/.keep"
/usr/bin/plutil -lint "$APP/Contents/Info.plist" >/dev/null
echo "已生成 ${APP}（指向 $ROOT/.venv）"
echo "启动：open $APP    或    ./$BIN/PCMAnalyzer samples/tone_stereo_48k.wav"
