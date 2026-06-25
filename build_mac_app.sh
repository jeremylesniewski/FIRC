#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV='.venv'
APP_NAME="FIRC"
SCRIPT="FIRC.py"
ICON_ICNS=icon/FIRC.icns

# --- Create venv if missing
if [ ! -d "$VENV" ]; then
  $PYTHON -m venv "$VENV"
fi
source "$VENV/bin/activate"
VENV_PY="$(pwd)/$VENV/bin/python"
APP_VERSION="$("$VENV_PY" -c "from firc import __version__; print(__version__)")"

# --- Install deps
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

# --- Check for optional helper binaries
SAS_BIN=$(which SwitchAudioSource || true)

# --- Clean previous builds 
rm -rf build dist "${APP_NAME}.spec" vendor

# --- Fetch & vendor CamillaDSP 
CAMILLADSP_VERSION="${CAMILLADSP_VERSION:-3.0.0}"
ARCH="$(uname -m)"
case "$ARCH" in
  arm64)  CDSP_TARBALL="camilladsp-macos-aarch64.tar.gz" ;;
  x86_64) CDSP_TARBALL="camilladsp-macos-amd64.tar.gz" ;;
  *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

CDSP_URL="https://github.com/HEnquist/camilladsp/releases/download/v${CAMILLADSP_VERSION}/${CDSP_TARBALL}"
VENDOR_DIR="vendor/bin"
mkdir -p "$VENDOR_DIR"

TMPDIR="$(mktemp -d)"
curl -L "$CDSP_URL" -o "$TMPDIR/$CDSP_TARBALL"
tar -xzf "$TMPDIR/$CDSP_TARBALL" -C "$TMPDIR"
mv "$TMPDIR/camilladsp" "$VENDOR_DIR/camilladsp"
chmod +x "$VENDOR_DIR/camilladsp"
rm -rf "$TMPDIR"

CAMILLADSP_BIN="$(pwd)/$VENDOR_DIR/camilladsp"
echo "Vendored CamillaDSP -> $CAMILLADSP_BIN"

# --- Create PyInstaller spec file 
cat > "${APP_NAME}.spec" << EOL
# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['$SCRIPT'],
    binaries=[
        (r'$CAMILLADSP_BIN', '.'),
        *([(r'$SAS_BIN', '.')] if '$SAS_BIN' else []),
    ],
    datas=[
        ('resources/config.yml', 'resources'),
        ('resources/config_bypass.yml', 'resources'),
        ('resources/test_config.yml', 'resources'),
        ('resources/impulse_L.wav', 'resources'),
        ('resources/impulse_R.wav', 'resources'),
        ('resources/images/General.webp', 'resources/images'),
    ],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='$APP_NAME',
    console=False,
    argv_emulation=True,
    icon='${ICON_ICNS}'
)
coll = COLLECT(exe, a.binaries, a.datas, name='$APP_NAME')
app = BUNDLE(
    coll,
    name='${APP_NAME}.app',
    icon='${ICON_ICNS}',
    bundle_identifier='com.github.jeremysalwen.firfiltercorrection',
    info_plist={
        'CFBundleName': 'FIRC',
        'CFBundleDisplayName': 'FIRC',
        'CFBundleShortVersionString': '${APP_VERSION}',
        'CFBundleVersion': '${APP_VERSION}',
        'NSMicrophoneUsageDescription': 'FIRC needs access to audio input devices.',
    }
)
EOL

# --- Build the app 
"$VENV_PY" -m PyInstaller --noconfirm "${APP_NAME}.spec"
echo "Build finished: dist/${APP_NAME}.app"

# --- Optional codesign
if [ -n "${CODESIGN_IDENTITY:-}" ]; then
  codesign --force --options runtime --timestamp \
    --sign "$CODESIGN_IDENTITY" \
    "dist/$APP_NAME.app/Contents/MacOS/camilladsp"

  codesign --force --options runtime --timestamp \
    --sign "$CODESIGN_IDENTITY" \
    "dist/$APP_NAME.app"
else
  echo "Skipping codesign. Set CODESIGN_IDENTITY to sign the app."
fi
