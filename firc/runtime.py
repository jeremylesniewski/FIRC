import os
import shutil
import sys
from pathlib import Path


def resource_path(rel):
    """Resolve a path relative to source or the macOS PyInstaller bundle."""
    rel_path = Path(rel)
    candidates = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass)
        candidates.extend([
            base / rel_path,
            base.parent / "Resources" / rel_path,
            base.parent.parent / "Resources" / rel_path,
        ])

    candidates.append(Path(__file__).resolve().parent.parent / rel_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


BUNDLED_CONFIG_PATH = resource_path("resources/config.yml")
BUNDLED_BYPASS_CONFIG_PATH = resource_path("resources/config_bypass.yml")
TEST_CONFIG_INTERNAL_PATH = resource_path("resources/test_config.yml")
BUNDLED_IMPULSE_L_PATH = resource_path("resources/impulse_L.wav")
BUNDLED_IMPULSE_R_PATH = resource_path("resources/impulse_R.wav")
APP_DATA_DIR = Path(os.environ.get("FIRC_CONFIG_DIR", Path.home() / "Library" / "Application Support" / "FIRC"))
CONFIG_INTERNAL_PATH = APP_DATA_DIR / "config.yml"
BYPASS_CONFIG_PATH = APP_DATA_DIR / "config_bypass.yml"
IMPULSE_L_PATH = APP_DATA_DIR / "impulse_L.wav"
IMPULSE_R_PATH = APP_DATA_DIR / "impulse_R.wav"


def ensure_runtime_files():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in (
        (BUNDLED_CONFIG_PATH, CONFIG_INTERNAL_PATH),
        (BUNDLED_BYPASS_CONFIG_PATH, BYPASS_CONFIG_PATH),
        (BUNDLED_IMPULSE_L_PATH, IMPULSE_L_PATH),
        (BUNDLED_IMPULSE_R_PATH, IMPULSE_R_PATH),
    ):
        if dst.exists():
            continue
        try:
            if src.exists():
                shutil.copy2(src, dst)
        except Exception:
            pass
