import os
import re
import subprocess
import sys
from pathlib import Path

from .deps import _AUDIO_OK, sd


def run(cmd, check=False, capture=True, text=True, timeout=30):
    try:
        if capture:
            if isinstance(cmd, str):
                p = subprocess.run(
                    cmd,
                    shell=True,
                    check=check,
                    capture_output=True,
                    text=text,
                    timeout=timeout,
                )
            else:
                p = subprocess.run(
                    cmd,
                    check=check,
                    capture_output=True,
                    text=text,
                    timeout=timeout,
                )
            return p.stdout, p.stderr, p.returncode
        return subprocess.Popen(cmd)
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", 124
    except FileNotFoundError as e:
        return "", str(e), 127


def which(name):
    from shutil import which as _which
    try:
        app_dir = Path(os.path.realpath(sys.argv[0])).parent
        for cand in (
            app_dir / name,
            app_dir / "bin" / name,
            app_dir.parent / "Frameworks" / name,
            app_dir.parent / "MacOS" / name,
            app_dir.parent / "Resources" / name,
        ):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    except Exception:
        pass
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass)
        for cand in (base / name, base.parent / "Frameworks" / name, base.parent / "MacOS" / name):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    p = _which(name)
    if p:
        return p
    for path in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/sbin"]:
        exe = Path(path) / name
        if exe.is_file() and os.access(exe, os.X_OK):
            return str(exe)
    return None


def parse_devices(listing: str):
    cap, play, mode = [], [], None
    for raw in listing.splitlines():
        line = raw.strip()
        if not line: continue
        if "Available capture devices:" in line:
            mode = "cap"
            m = re.findall(r'\["(.*?)"\]', line)
            if m: cap.extend(m[0].split('\",\"'))
            continue
        if "Available playback devices:" in line:
            mode = "play"
            m = re.findall(r'\["(.*?)"\]', line)
            if m: play.extend(m[0].split('\",\"'))
            continue
        m2 = re.findall(r'"([^"]+)"', line)
        if m2:
            if mode == "cap": cap.extend(m2)
            elif mode == "play": play.extend(m2)
    cap = list(dict.fromkeys(cap))
    play = list(dict.fromkeys(play))
    return cap, play


def _sd_list_devices():
    if not _AUDIO_OK:
        return [], []
    try:
        devices = sd.query_devices()
    except Exception:
        return [], []

    cap, play = [], []
    for dev in devices:
        # Keep exact CoreAudio names (some devices have trailing spaces).
        name = dev.get("name") or ""
        if not name.strip():
            continue
        if dev.get("max_input_channels", 0) > 0:
            cap.append(name)
        if dev.get("max_output_channels", 0) > 0:
            play.append(name)
    return list(dict.fromkeys(cap)), list(dict.fromkeys(play))


def _canonical_device_name(name, kind):
    """Resolve to the exact device string CoreAudio/CamillaDSP expects."""
    if not name or not _AUDIO_OK:
        return name
    try:
        devices = sd.query_devices()
    except Exception:
        return name

    wanted = name.strip()
    fuzzy = None
    for dev in devices:
        devname = dev.get("name") or ""
        if not devname.strip():
            continue
        if kind == "input" and dev.get("max_input_channels", 0) <= 0:
            continue
        if kind == "output" and dev.get("max_output_channels", 0) <= 0:
            continue
        if devname == name:
            return devname
        if devname.strip() == wanted:
            return devname
        if devname.strip().lower() == wanted.lower() and fuzzy is None:
            fuzzy = devname
    return fuzzy or name


def _matching_device_name(name, choices):
    wanted = (name or "").strip()
    if not wanted:
        return None
    for choice in choices:
        if choice == name or choice.strip() == wanted:
            return choice
    for choice in choices:
        if choice.strip().lower() == wanted.lower():
            return choice
    return None


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text or "")


def summarize_camilla_output(text):
    """Pull a short user-facing message from verbose CamillaDSP logs."""
    clean = strip_ansi(text)
    errors = []
    for line in clean.splitlines():
        if "Playback error:" in line:
            errors.append(line.split("Playback error:", 1)[-1].strip())
        elif "Capture error:" in line:
            errors.append(line.split("Capture error:", 1)[-1].strip())
        elif "ERROR" in line:
            errors.append(line.strip())
    if errors:
        return "\n".join(dict.fromkeys(errors))
    tail = clean.strip()
    if not tail:
        return "CamillaDSP reported an unknown error. See Logs for details."
    return tail[-800:] if len(tail) > 800 else tail

def _sd_find_device_index(name, kind):
    if not _AUDIO_OK: return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    for i, dev in enumerate(devices):
        devname = (dev.get('name') or '')
        if name and name.lower() in devname.lower():
            if kind == 'input' and dev.get('max_input_channels', 0) > 0: return i
            if kind == 'output' and dev.get('max_output_channels', 0) > 0: return i
    return None

def _sd_try_output_loopback(playback_name):
    return _sd_find_device_index(playback_name, 'input')

MONITOR_TAP_CANDIDATES = [
    "BlackHole 2ch","BlackHole 16ch","BlackHole 64ch",
    "Soundflower (64ch)","Loopback Audio","iShowU Audio Capture"
]
def _sd_find_monitor_tap(capture_name=""):
    if not _AUDIO_OK: return None, None
    try:
        devices = sd.query_devices()
    except Exception:
        return None, None
    for i, dev in enumerate(devices):
        name = (dev.get('name') or '')
        if any(c in name for c in MONITOR_TAP_CANDIDATES) and name != capture_name and dev.get('max_input_channels', 0) > 0:
            return i, name
    return None, None


def maybe_switch_output_to_blackhole(capture_dev):
    if which("SwitchAudioSource") is None:
        return False, "SwitchAudioSource not installed. Manually set system output to: " + capture_dev
    out, err, rc = run(["SwitchAudioSource", "-t", "output", "-s", capture_dev])
    if rc != 0:
        return False, out or err
    return True, "Switched system output to: " + capture_dev
