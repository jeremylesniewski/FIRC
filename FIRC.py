import os
import sys
import re
import subprocess
import threading
import queue
import time
from pathlib import Path
import wave
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import websocket


# OptionalYAML 
try:
    import yaml
    _YAML_OK = True
except Exception:
    _YAML_OK = False


# Matplotlib 
def _ensure_mplconfigdir():
    try:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        mpldir = Path(os.environ.get("MPLCONFIGDIR", base / "matplotlib"))
        mpldir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpldir)
    except Exception:
        pass
_ensure_mplconfigdir()

# Optional deps
_AUDIO_OK = True   # numpy + sounddevice
_PLOT_OK  = True   # matplotlib plotting

try:
    import numpy as np
    import sounddevice as sd
except Exception:
    _AUDIO_OK = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.collections import PolyCollection
except Exception:
    _PLOT_OK = False

# WebSocket
_WS_OK = True
try:
    import json
    from websocket import create_connection, WebSocketConnectionClosedException
except Exception:
    _WS_OK = False

# ====================== Paths / resources ======================
def resource_path(rel):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel

CONFIG_INTERNAL_PATH = resource_path('resources/config.yml')
BYPASS_CONFIG_PATH   = resource_path('resources/config_bypass.yml')
TEST_CONFIG_INTERNAL_PATH = resource_path('resources/test_config.yml')
IMPULSE_L_PATH = resource_path('resources/impulse_L.wav')
IMPULSE_R_PATH = resource_path('resources/impulse_R.wav')

# ====================== Subprocess helpers ======================
def run(cmd, check=False, capture=True, text=True):
    try:
        if capture:
            if isinstance(cmd, str):
                p = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=text)
            else:
                p = subprocess.run(cmd, check=check, capture_output=True, text=text)
            return p.stdout, p.stderr, p.returncode
        return subprocess.Popen(cmd)
    except FileNotFoundError as e:
        return "", str(e), 127

def which(name):
    from shutil import which as _which
    try:
        app_dir = Path(os.path.realpath(sys.argv[0])).parent
        cand = app_dir / name
        if cand.is_file() and os.access(cand, os.X_OK): return str(cand)
        cand2 = app_dir / "bin" / name
        if cand2.is_file() and os.access(cand2, os.X_OK): return str(cand2)
    except Exception:
        pass
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / name
        if cand.is_file() and os.access(cand, os.X_OK): return str(cand)
    p = _which(name)
    if p: return p
    for path in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/sbin"]:
        exe = Path(path) / name
        if exe.is_file() and os.access(exe, os.X_OK): return str(exe)
    return None

def write_test_config():
    return TEST_CONFIG_INTERNAL_PATH

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

def maybe_switch_output_to_blackhole(capture_dev):
    if which("SwitchAudioSource") is None:
        return False, "SwitchAudioSource not installed. Manually set system output to: " + capture_dev
    out, err, rc = run(["SwitchAudioSource", "-t", "output", "-s", capture_dev])
    if rc != 0: return False, out or err
    return True, "Switched system output to: " + capture_dev

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

# =================== Post-FIR WebSocket (Camilla) ===================
class CamillaWS:
    def __init__(self, url="ws://127.0.0.1:1234"):
        self.url = url
        self.ws = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_playback_peak_db = None
        self.last_playback_rms_db  = None

    def start(self):
        if not _WS_OK or self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        with self.lock:
            try:
                if self.ws: self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _connect(self):
        try:
            ws = create_connection(self.url, timeout=1.0)
            try:
                ws.send(json.dumps({"SetUpdateInterval": 30})); _ = ws.recv()
            except Exception:
                pass
            return ws
        except Exception:
            return None

    def _loop(self):
        while self.running:
            if self.ws is None:
                self.ws = self._connect()
                if self.ws is None:
                    time.sleep(0.5); continue
            try:
                self.ws.send(json.dumps("GetPlaybackSignalPeak"))
                r1 = self.ws.recv()
                if r1:
                    try:
                        obj = json.loads(r1)
                        v = obj.get("GetPlaybackSignalPeak", {}).get("value", [])
                        if isinstance(v, list) and v: self.last_playback_peak_db = float(max(v))
                    except Exception:
                        pass
                try:
                    self.ws.send(json.dumps("GetPlaybackSignalRms"))
                    r2 = self.ws.recv()
                    if r2:
                        obj2 = json.loads(r2)
                        v2 = obj2.get("GetPlaybackSignalRms", {}).get("value", [])
                        if isinstance(v2, list) and v2: self.last_playback_rms_db = float(max(v2))
                except Exception:
                    pass
            except (WebSocketConnectionClosedException, OSError):
                with self.lock:
                    try:
                        if self.ws: self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                time.sleep(0.2)
            except Exception:
                time.sleep(0.1)
            time.sleep(0.05)

# ============ Utility: create identity impulse WAVs (bypass) ============
def _ensure_impulse_wav(path: Path, samplerate: int, n_samples: int = 2048):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        frames = [32767] + [0] * (max(1, n_samples) - 1)
        raw = b"".join(struct.pack("<h", s) for s in frames)
        wf.writeframes(raw)

# ============================ Analyzer (FFT) ============================
class Analyzer(ttk.LabelFrame):
    def __init__(self, master, get_capture_name, get_playback_name, get_samplerate, get_monitor_mode):
        super().__init__(master)
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate    = get_samplerate
        self.get_monitor_mode  = get_monitor_mode

        self.running = False
        self.stream = None
        self.buffer = queue.Queue(maxsize=24)
        self.update_job = None
        self.loopback_unavailable = False
        self.using_fallback_input_for_output = False

        self.y_min_db = -60
        self.y_max_db = 0
        self._last_freqs    = None
        self._smoothed_mag  = None
        self._alpha         = 0.25
        self._fft_size      = 2048

        status_row = ttk.Frame(self); status_row.pack(fill=tk.X, padx=6, pady=(6, 4))
        self._monitor_label_text = tk.StringVar(value="")
        ttk.Label(status_row, textvariable=self._monitor_label_text).pack(side=tk.LEFT)

        body = ttk.Frame(self); body.pack(fill=tk.BOTH, expand=True)
        if _AUDIO_OK and _PLOT_OK:
            self.fig = Figure(figsize=(9.5, 3.2), dpi=100, facecolor="#2b2b2b")
            self.ax  = self.fig.add_subplot(111, facecolor="#2b2b2b")
            self.ax.set_xscale('log'); self._set_axes_limits_once()
            self.ax.grid(True, which='both', alpha=0.25, color="#555")
            self.ax.tick_params(labelbottom=True, labelleft=False, colors="#d2d4d6", labelsize=9)
            for spine in self.ax.spines.values(): spine.set_color("#8a8d93")
            xticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            self.ax.set_xticks(xticks)
            self.ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}")
            (self.line,) = self.ax.plot([], [], linewidth=1.2)
            self.fill_poly = None
            self.canvas = FigureCanvasTkAgg(self.fig, master=body)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 3), pady=(0, 6))
            self.overlay = tk.Label(self.canvas_widget, text="Output spectrum unavailable", fg="#d2d4d6", bg="#2b2b2b")
            self.overlay.place_forget()
        else:
            ttk.Label(self, text="Install: pip install numpy sounddevice matplotlib", foreground="orange").pack(fill=tk.X, padx=6, pady=8)

    def start(self):
        if self.running or not _AUDIO_OK or not _PLOT_OK: return
        sr = self._safe_sr()
        if not sr: return

        self._smoothed_mag = None
        self._last_freqs = None
        self.loopback_unavailable = False
        self.using_fallback_input_for_output = False
        self._update_overlay()

        mode = (self.get_monitor_mode() or "input")
        label = ""
        dev_idx = None
        if mode == "input":
            dev_idx = _sd_find_device_index(self.get_capture_name(), 'input'); label = "Input monitor: capture"
        else:
            dev_idx = _sd_try_output_loopback(self.get_playback_name())
            if dev_idx is not None:
                label = "Output monitor: DAC loopback"
            else:
                tap_idx, tap_name = _sd_find_monitor_tap(self.get_capture_name())
                if tap_idx is not None:
                    dev_idx = tap_idx; label = f"Output monitor: {tap_name}"
                else:
                    dev_idx = _sd_find_device_index(self.get_capture_name(), 'input')
                    self.using_fallback_input_for_output = True
                    self.loopback_unavailable = True
                    label = "Output monitor: (fallback to capture)"

        self._monitor_label_text.set(label)

        def cb(indata, frames, time_info, status):
            x = indata
            if x.ndim > 1: x = np.mean(x[:, :min(2, x.shape[1])], axis=1)
            try: self.buffer.put_nowait(x.astype(np.float64, copy=False))
            except queue.Full: pass

        if dev_idx is not None:
            try:
                try: ch = max(1, int(sd.query_devices(dev_idx).get('max_input_channels', 1)))
                except Exception: ch = 1
                ch = 2 if ch >= 2 else 1
                self.stream = sd.InputStream(device=dev_idx, channels=ch, samplerate=sr, blocksize=256, dtype='float32', callback=cb)
                self.stream.start()
            except Exception:
                self.stream = None
        else:
            self.stream = None

        self.running = True
        self._schedule_update(self._fft_size, sr, period_ms=33)
        self._update_overlay()

    def stop(self):
        if not self.running: return
        self.running = False
        if self.update_job is not None:
            self.after_cancel(self.update_job); self.update_job = None
        try:
            if self.stream: self.stream.stop(); self.stream.close()
        finally:
            self.stream = None
        if _PLOT_OK:
            self.line.set_data([], [])
            if self.fill_poly is not None: self.fill_poly.remove(); self.fill_poly = None
            self.canvas.draw_idle()
        self._update_overlay()

    def _safe_sr(self):
        try: return int(self.get_samplerate())
        except Exception: return None

    def _schedule_update(self, fft_size, sr, period_ms=33):
        if not self.running: return
        self._update_plot(fft_size, sr)
        self.update_job = self.after(period_ms, lambda: self._schedule_update(fft_size, sr, period_ms))

    def _set_axes_limits_once(self):
        self.ax.set_xlim(20, 20000); self.ax.set_ylim(self.y_min_db, self.y_max_db)

    def _rfft_dbfs(self, x, sr):
        N = len(x)
        if N == 0: return np.array([0.0]), np.array([self.y_min_db])
        w = np.hanning(N); cg = np.sum(w) / N
        X = np.fft.rfft(x * w); A = np.abs(X) / (N * cg)
        if A.size > 2: A[1:-1] *= 2.0
        mag_db = 20.0 * np.log10(np.maximum(A, 1e-12))
        freqs = np.fft.rfftfreq(N, 1.0 / sr)
        return freqs, mag_db

    def _smooth(self, new_mag):
        if self._smoothed_mag is None or len(self._smoothed_mag) != len(new_mag):
            self._smoothed_mag = new_mag.copy()
        else:
            self._smoothed_mag = self._alpha * new_mag + (1.0 - self._alpha) * self._smoothed_mag
        return self._smoothed_mag

    def _update_fill(self, freqs, y):
        verts = np.column_stack([freqs, y]); bottom = self.y_min_db
        poly = np.vstack([[freqs[0], bottom], verts, [freqs[-1], bottom]])
        if self.fill_poly is None:
            self.fill_poly = PolyCollection([poly], facecolor=(0.3, 0.6, 1.0, 0.25), edgecolor='none')
            self.ax.add_collection(self.fill_poly)
        else:
            self.fill_poly.set_verts([poly])

    def _update_plot(self, fft_size, sr):
        if not _PLOT_OK or self.buffer.empty(): return
        need = fft_size; frames = []
        while need > 0 and not self.buffer.empty():
            x = self.buffer.get_nowait(); frames.append(x); need -= len(x)
        if not frames: return
        x = np.concatenate(frames)
        if len(x) < fft_size: x = np.pad(x, (0, fft_size - len(x)))
        else: x = x[-fft_size:]
        freqs, mag_db = self._rfft_dbfs(x, sr)
        mag_db = np.clip(mag_db, self.y_min_db, self.y_max_db)
        if self._last_freqs is None or len(self._last_freqs) != len(freqs) or not np.allclose(self._last_freqs, freqs):
            self._last_freqs = freqs; self.line.set_data(freqs, mag_db)
        mag_sm = self._smooth(mag_db)
        self.line.set_data(freqs, mag_sm); self._update_fill(freqs, mag_sm)
        self.canvas.draw_idle()

    def _update_overlay(self):
        if not _PLOT_OK: return
        mode = (self.get_monitor_mode() or "input")
        use_overlay = (mode == "output" and (self.stream is None or self.loopback_unavailable is True) and not self.using_fallback_input_for_output)
        if use_overlay: self.overlay.place(relx=0.5, rely=0.5, anchor="center")
        else: self.overlay.place_forget()

# ============================== Meter ==============================
class LogicMeter(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.db_min = -60.0
        self.db_max = 0.0
        self.peak_hold = None
        self.peak_hold_time = time.time()
        self.peak_decay_rate = 20.0
        self.peak_hold_duration = 2.0
        self.canvas = tk.Canvas(self, height=36, highlightthickness=0, bg="#2b2b2b")
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.label_row = tk.Canvas(self, height=16, highlightthickness=0, bg="#2b2b2b")
        self.label_row.grid(row=1, column=0, sticky="ew", pady=(2,0))
        self.bind("<Configure>", lambda e: self._redraw())
        self.canvas.bind("<Button-1>", lambda e: self._reset_peak())

    def draw_meter(self, db_current, db_peak):
        now = time.time()
        if db_peak is not None:
            db_peak = max(self.db_min, min(self.db_max, float(db_peak)))
            if self.peak_hold is None or db_peak > self.peak_hold:
                self.peak_hold = db_peak; self.peak_hold_time = now
        if db_current is not None:
            db_current = max(self.db_min, min(self.db_max, float(db_current)))
        self._redraw(db_current)

    def _reset_peak(self):
        self.peak_hold = None; self.peak_hold_time = time.time(); self._redraw()

    def _db_to_x(self, db, w):
        db = max(self.db_min, min(self.db_max, db))
        return int((db - self.db_min) / (self.db_max - self.db_min) * w)

    def _redraw(self, db_current=None):
        self.update_idletasks()
        w = max(self.winfo_width(), 400); h = 36
        c = self.canvas; c.configure(height=h); c.delete("all")
        xg1 = self._db_to_x(-12, w); xy1 = self._db_to_x(-3, w); xr1 = self._db_to_x(0, w)
        c.create_rectangle(0, 0, xg1, h, fill="#1f7a7a", width=0)
        c.create_rectangle(xg1, 0, xy1, h, fill="#b8a21a", width=0)
        c.create_rectangle(xy1, 0, xr1, h, fill="#b0413e", width=0)
        if db_current is not None:
            xcur = self._db_to_x(db_current, w)
            c.create_rectangle(0, 0, xcur, h, fill="#5fd0d0", width=0)
        if self.peak_hold is not None:
            now = time.time(); elapsed = now - self.peak_hold_time
            if elapsed > self.peak_hold_duration:
                self.peak_hold -= self.peak_decay_rate * (elapsed - self.peak_hold_duration)
                self.peak_hold_time = now
                if self.peak_hold < self.db_min: self.peak_hold = None
        if self.peak_hold is not None:
            xpk = self._db_to_x(self.peak_hold, w)
            c.create_line(xpk, 0, xpk, h, fill="#00ffcc", width=1)
            box_w, box_h = 56, 18
            px = min(max(2, xpk - box_w // 2), w - box_w - 2); py = 2
            c.create_rectangle(px, py, px + box_w, py + box_h, fill="#22262b", outline="#00ffcc")
            c.create_text(px + box_w/2, py + box_h/2, text=f"{self.peak_hold:.1f} dB", fill="#d2d4d6", font=("TkDefaultFont", 9))
        lr = self.label_row; lr.delete("all")
        for t in [-60, -48, -36, -24, -12, -6, -3, 0]:
            xt = self._db_to_x(t, w)
            lr.create_line(xt, 0, xt, 6, fill="#72767d")
            lr.create_text(xt, 12, text=("0" if t == 0 else f"{t}"), fill="#9aa0a5", font=("TkDefaultFont", 8))

# ==================== Light audio tap for the meter ====================
class LevelTap:
    def __init__(self, get_capture_name, get_playback_name, get_samplerate, get_monitor_mode):
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate    = get_samplerate
        self.get_monitor_mode  = get_monitor_mode
        self.stream = None
        self.buffer = queue.Queue(maxsize=4)
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not _AUDIO_OK or self.running: return
        self.running = True
        self._open_stream()

    def stop(self):
        self.running = False
        with self.lock:
            try:
                if self.stream: self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _open_stream(self):
        try: sr   = int(self.get_samplerate() or 48000)
        except Exception: sr = 48000
        try: mode = (self.get_monitor_mode() or "input")
        except Exception: mode = "input"
        try: cap_name  = (self.get_capture_name() or "")
        except Exception: cap_name = ""
        try: play_name = (self.get_playback_name() or "")
        except Exception: play_name = ""

        dev_idx = None
        if mode == "input":
            dev_idx = _sd_find_device_index(cap_name, 'input')
        else:
            dev_idx = _sd_try_output_loopback(play_name)
            if dev_idx is None:
                tap_idx, _tap_name = _sd_find_monitor_tap(cap_name)
                dev_idx = tap_idx
            if dev_idx is None:
                dev_idx = _sd_find_device_index(cap_name, 'input')

        def cb(indata, frames, time_info, status):
            x = indata
            if x.ndim > 1: x = np.mean(x[:, :min(2, x.shape[1])], axis=1)
            try:
                self.buffer.put_nowait(x.astype(np.float64, copy=False))
            except queue.Full:
                try: _ = self.buffer.get_nowait()
                except Exception: pass
                try: self.buffer.put_nowait(x.astype(np.float64, copy=False))
                except Exception: pass

        try:
            if dev_idx is None: return
            try: ch = max(1, int(sd.query_devices(dev_idx).get('max_input_channels', 1)))
            except Exception: ch = 1
            ch = 2 if ch >= 2 else 1
            self.stream = sd.InputStream(device=dev_idx, channels=ch, samplerate=sr, blocksize=256, dtype='float32', callback=cb)
            self.stream.start()
        except Exception:
            self.stream = None

# ============================== YAML read ==============================
def _read_yaml_config(path: Path):
    if not _YAML_OK or not path.exists():
        return None, "PyYAML missing or config not found."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, None
    except Exception as e:
        return None, f"YAML read error: {e}"

# ================================ GUI =================================
class FIRFilterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FIRC")
       
        self.resizable(True, True)

        self.proc = None
        self.proc_thread = None
        self.proc_mode = None

        self.log_queue = queue.Queue()
        self._log_buffer = []

        self.ws = CamillaWS("ws://127.0.0.1:1234")
        if _WS_OK: self.ws.start()

        self._init_styles()
        self.create_widgets()

        # Meter loop independent of analyzer/matplotlib
        self.after(50, self._update_bottom_meter)
        self._sync_launch_btn()  # sets label + bypass indicator

        self.sr_status_var.set("Click Apply to set sample rate")
        self.sr_combo.bind('<<ComboboxSelected>>', lambda e: self.on_sr_change())

        def init():
            self.refresh_devices("all")
            self._ensure_bypass_config()
            try:
                self.level_tap = LevelTap(
                    get_capture_name=lambda: self.cap_var.get(),
                    get_playback_name=lambda: self.play_var.get(),
                    get_samplerate=lambda: self.sr_var.get(),
                    get_monitor_mode=lambda: self.monitor_sel.get()
                )
                self.level_tap.start()
            except Exception as e:
                self.append_log(f"[LevelTap] init warning: {e}\n")
            self._auto_start_bypass()
            self.append_log("Ready.\n")
            # Capture base window size (compact mode)
            self.update_idletasks()
            self._base_w = self.winfo_width()
            self._base_h = self.winfo_height()
            self._apply_window_mode("none")
        self.after(150, init)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- styles ----------
    def _init_styles(self):
        bg_main = "#2b2b2b"; bg_panel = "#313335"; bg_field = "#383a3c"
        fg_text = "#d2d4d6"; fg_dim = "#a0a2a5"; fg_invert = "#000000"
        btn_bg = "#3d4042"; btn_hover = "#4a4d50"; btn_border = "#505355"
        accent_bg = "#4b6ea9"; accent_hover = "#3d5c91"; accent_border = "#5779b8"; frame_border = "#414345"
        self.configure(bg=bg_main)
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
        style.configure(".", background=bg_main, foreground=fg_text, relief="flat")
        style.configure("TFrame", background=bg_panel)
        style.configure("TLabelFrame", background=bg_panel, foreground=fg_text, bordercolor=frame_border)
        style.configure("TLabel", background=bg_panel, foreground=fg_text)
        style.configure("TEntry", fieldbackground=bg_field, foreground=fg_text, insertcolor=fg_text, bordercolor=frame_border)
        style.configure("TCombobox", fieldbackground=bg_field, background=bg_field, foreground=fg_text, arrowcolor=fg_dim, bordercolor=frame_border)
        style.configure("TButton", background=btn_bg, foreground=fg_text, borderwidth=1, relief="flat", focusthickness=0, bordercolor=btn_border)
        style.map("TButton", background=[("active", btn_hover), ("pressed", btn_hover)], relief=[("pressed", "sunken")])
        style.configure("Accent.TButton", background=accent_bg, foreground=fg_invert, borderwidth=1, relief="flat", bordercolor=accent_border, focusthickness=0)
        style.map("Accent.TButton", background=[("active", accent_hover), ("pressed", accent_hover)], foreground=[("active", fg_invert)], relief=[("pressed", "sunken")])
        style.configure("TRadiobutton", background=bg_panel, foreground=fg_text)
        style.configure("TCheckbutton", background=bg_panel, foreground=fg_text)
        style.configure("Vertical.TScrollbar", background=bg_panel, troughcolor=bg_main, bordercolor=bg_main)
        style.configure("TLabelframe.Label", foreground=fg_dim, background=bg_panel)
        try: style.configure(".", font=("Helvetica Neue", 11))
        except Exception: pass

    # ---------- layout ----------
    def create_widgets(self):
        BW = 18
        self.grid_rowconfigure(0, weight=1); self.grid_columnconfigure(0, weight=1)
        frm = ttk.Frame(self); frm.grid(row=0, column=0, sticky="nsew", padx=10, pady=10); frm.grid_columnconfigure(0, weight=1)

        # Devices
        dev_frame = ttk.LabelFrame(frm, text="Devices")
        dev_frame.grid(row=0, column=0, sticky="ew")
        dev_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(dev_frame, text="Capture:").grid(row=0, column=0, sticky="w", padx=(6,5), pady=2)
        self.cap_var = tk.StringVar(); self.cap_combo = ttk.Combobox(dev_frame, textvariable=self.cap_var, width=50)
        self.cap_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(dev_frame, text="Playback:").grid(row=1, column=0, sticky="w", padx=(6,5), pady=2)
        self.play_var = tk.StringVar(); self.play_combo = ttk.Combobox(dev_frame, textvariable=self.play_var, width=50)
        self.play_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(dev_frame, text="Refresh Devices", width=BW, command=lambda: self.refresh_devices("all")).grid(row=2, column=1, sticky="e", padx=6, pady=(0,4))

        # FIR Configuration (separate L/R only)
        fir_frame = ttk.LabelFrame(frm, text="FIR Configuration")
        fir_frame.grid(row=1, column=0, sticky="ew", pady=(8,0))
        for c in range(5): fir_frame.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        sr_frame = ttk.Frame(fir_frame); sr_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(sr_frame, text="Samplerate:").pack(side=tk.LEFT)
        self.sr_var = tk.StringVar(value="48000")
        self.sr_combo = ttk.Combobox(sr_frame, textvariable=self.sr_var, values=["44100","48000"], width=10)
        self.sr_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(sr_frame, text="Apply", width=12, command=self.apply_sample_rate).pack(side=tk.LEFT)
        self.sr_status_var = tk.StringVar()
        ttk.Label(sr_frame, textvariable=self.sr_status_var, font=("",9,"italic")).pack(side=tk.LEFT, padx=5)

        mon_frame = ttk.Frame(fir_frame); mon_frame.grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=(20,5))
        ttk.Label(mon_frame, text="Meter Source:").pack(side=tk.LEFT)
        self.monitor_sel = tk.StringVar(value="input")
        ttk.Radiobutton(mon_frame, text="Input",  variable=self.monitor_sel, value="input",  command=lambda: self._on_meter_source()).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(mon_frame, text="Output", variable=self.monitor_sel, value="output", command=lambda: self._on_meter_source()).pack(side=tk.LEFT, padx=6)

        fir_lr = ttk.Frame(fir_frame); fir_lr.grid(row=2, column=0, columnspan=5, sticky="ew", pady=5)
        ttk.Label(fir_lr, text="Left FIR:").grid(row=0, column=0, sticky="w", padx=5)
        self.fir_left_var = tk.StringVar()
        ttk.Entry(fir_lr, textvariable=self.fir_left_var, width=60).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(fir_lr, text="Browse...", width=12, command=lambda: self.browse_fir_separate("left")).grid(row=0, column=2, padx=5)
        ttk.Label(fir_lr, text="Right FIR:").grid(row=1, column=0, sticky="w", padx=5)
        self.fir_right_var = tk.StringVar()
        ttk.Entry(fir_lr, textvariable=self.fir_right_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(fir_lr, text="Browse...", width=12, command=lambda: self.browse_fir_separate("right")).grid(row=1, column=2, padx=5)
        fir_lr.grid_columnconfigure(1, weight=1)

        # Action bar
        action = ttk.Frame(frm); action.grid(row=4, column=0, sticky="ew", pady=(8,0))
        for i in range(7): action.grid_columnconfigure(i, weight=1 if i == 3 else 0)
        ttk.Button(action, text="Show Config",  width=BW, command=self.open_config).grid(row=0, column=0, padx=(0,6), sticky="w")
        ttk.Button(action, text="Write Config", width=BW, command=self.write_to_config).grid(row=0, column=1, padx=(0,6), sticky="w")

        self.btn_toggle_vis  = ttk.Button(action, text="Visualizer", width=BW, command=self._toggle_visualizer_exclusive)
        self.btn_toggle_vis.grid(row=0, column=2, padx=(0,6), sticky="w")
        self.btn_toggle_logs = ttk.Button(action, text="Logs",       width=BW, command=self._toggle_logs_exclusive)
        self.btn_toggle_logs.grid(row=0, column=3, padx=(0,6), sticky="w")

        ttk.Button(action, text="Switch Output", width=BW, command=self.switch_output).grid(row=0, column=4, padx=(0,6), sticky="e")
        self.launch_btn = ttk.Button(action, text="Start correction", width=BW, style="Accent.TButton", command=self.toggle_launch)
        self.launch_btn.grid(row=0, column=6, padx=(6,0), sticky="e")

        # (bypassed) indicator below the Start/Stop button
        self.bypass_var = tk.StringVar(value="")
        ttk.Label(action, textvariable=self.bypass_var, font=("", 9, "italic")).grid(row=1, column=6, padx=(6,0), sticky="ne")

        # Shared container for Visualizer OR Logs (exclusive)
        self.an_container = ttk.Frame(frm)
        self.an_container.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        self.an_container.grid_columnconfigure(0, weight=1)
        # we don't set row weights because we show only one child at a time

        # Analyzer frame
        self.an_frame = ttk.LabelFrame(self.an_container, text="Analyzer")
        self.analyzer  = Analyzer(
            self.an_frame,
            get_capture_name=lambda: self.cap_var.get(),
            get_playback_name=lambda: self.play_var.get(),
            get_samplerate=lambda: self.sr_var.get(),
            get_monitor_mode=lambda: self.monitor_sel.get()
        )
        self.analyzer.pack(fill=tk.BOTH, expand=True)
        self._an_visible = False

        # Logs frame
        self.log_frame = ttk.LabelFrame(self.an_container, text="CamillaDSP logs")
        self.log_text  = scrolledtext.ScrolledText(self.log_frame, state=tk.DISABLED, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        try: self.log_text.configure(background="#2b2b2b", foreground="#d2d4d6", insertbackground="#d2d4d6")
        except Exception: pass
        self._logs_visible = False
        self._flush_log_buffer()

        # Bottom meter
        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10)); bottom.grid_columnconfigure(0, weight=1)
        self.output_meter = LogicMeter(bottom)
        self.output_meter.grid(row=0, column=0, sticky="ew")

        # start compact: hide both
        self.an_frame.grid_remove()
        self.log_frame.grid_remove()

    # ---------- exclusive view controls & window resizing ----------
    def _apply_window_mode(self, mode: str):
        """mode: 'none' | 'an' | 'logs' — shows exactly one (or none) and resizes window."""
        # hide both first
        self.an_frame.grid_remove(); self._an_visible = False
        self.log_frame.grid_remove(); self._logs_visible = False

        if mode == "an":
            self.an_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(6,0)); self._an_visible = True
            try:
                if _AUDIO_OK and _PLOT_OK and not self.analyzer.running: self.analyzer.start()
            except Exception: pass
        elif mode == "logs":
            self.log_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(6,0)); self._logs_visible = True
        # else 'none': nothing is visible

        # Resize window to fit requested content
        self.update_idletasks()
        if mode == "none" and hasattr(self, "_base_w") and hasattr(self, "_base_h"):
            self.geometry(f"{self._base_w}x{self._base_h}")
        else:
            w = self.winfo_reqwidth()
            h = self.winfo_reqheight()
            self.geometry(f"{w}x{h}")

    def _toggle_visualizer_exclusive(self):
        if self._an_visible:
            self._apply_window_mode("none")
        else:
            self._apply_window_mode("an")

    def _toggle_logs_exclusive(self):
        if self._logs_visible:
            self._apply_window_mode("none")
        else:
            self._apply_window_mode("logs")

    # ---------- meter source change ----------
    def _on_meter_source(self):
        try:
            if getattr(self, "level_tap", None):
                self.level_tap.stop(); self.level_tap.start()
        except Exception: pass
        try:
            if _AUDIO_OK and _PLOT_OK and self._an_visible:
                self.analyzer.stop(); self.analyzer.start()
        except Exception: pass

    # ---------- meter loop ----------
    def _update_bottom_meter(self):
        db_current = None; db_peak = None
        if _WS_OK and (self.ws.last_playback_rms_db is not None or self.ws.last_playback_peak_db is not None):
            db_current = float(self.ws.last_playback_rms_db) if self.ws.last_playback_rms_db is not None else float(self.ws.last_playback_peak_db)
            db_peak    = float(self.ws.last_playback_peak_db) if self.ws.last_playback_peak_db is not None else db_current
        else:
            src = None
            if getattr(self, "level_tap", None) and self.level_tap.buffer.qsize() > 0:
                src = self.level_tap.buffer
            if src is not None:
                try:
                    x = src.queue[-1] if hasattr(src, "queue") else src.get_nowait()
                    peak = float(np.max(np.abs(x)))
                    db_current = 20.0 * np.log10(max(peak, 1e-12))
                    db_peak = db_current
                except Exception:
                    pass
        if db_current is not None: db_current = max(-60.0, min(0.0, db_current))
        if db_peak    is not None: db_peak    = max(-60.0, min(0.0, db_peak))
        self.output_meter.draw_meter(db_current, db_peak)
        self.after(33, self._update_bottom_meter)

    # ---------- logs ----------
    def append_log(self, text: str):
        if hasattr(self, "log_text") and self.log_text:
            if getattr(self, "_log_buffer", None):
                self.log_text.configure(state=tk.NORMAL)
                for t in self._log_buffer: self.log_text.insert(tk.END, t)
                self._log_buffer.clear(); self.log_text.configure(state=tk.DISABLED)
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, text)
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        else:
            self._log_buffer.append(text)

    def _flush_log_buffer(self):
        if hasattr(self, "log_text") and self.log_text and getattr(self, "_log_buffer", None):
            self.log_text.configure(state=tk.NORMAL)
            for t in self._log_buffer: self.log_text.insert(tk.END, t)
            self._log_buffer.clear()
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)

    # ---------- Start/Stop button & bypass label ----------
    def _sync_launch_btn(self):
        label = "Stop correction" if self.proc_mode == 'correction' else "Start correction"
        try: self.launch_btn.config(text=label)
        except Exception: pass
        # bypass indicator
        self.bypass_var.set("(bypassed)" if self.proc_mode == 'bypass' else "")

    # ---------- config open/write ----------
    def open_config(self):
        try:
            cfg = CONFIG_INTERNAL_PATH
            if not cfg.exists():
                messagebox.showinfo("Not found", f"Bundled config does not exist: {cfg}"); return
            subprocess.run(["open", str(cfg)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open config: {e}")

    def write_to_config(self):
        left  = (self.fir_left_var.get().strip()  if hasattr(self, "fir_left_var")  and self.fir_left_var.get()  else "")
        right = (self.fir_right_var.get().strip() if hasattr(self, "fir_right_var") and self.fir_right_var.get() else "")
        if not left or not right:
            messagebox.showerror("Missing FIR", "Please select BOTH Left and Right FIR WAV files."); return

        sr = int(self.sr_var.get() or 48000)
        cap  = self.cap_var.get()  or "BlackHole 2ch"
        play = self.play_var.get() or "USB Audio CODEC "
        cap_esc  = cap.replace('"', '\\"')
        play_esc = play.replace("'", "''")
        left_esc  = left.replace('"', '\\"')
        right_esc = right.replace('"', '\\"')

        yaml_text = (
f"""devices:
  samplerate: {sr}
  chunksize: 1024
  capture:
    type: CoreAudio
    device: "{cap_esc}"
    channels: 2
  playback:
    type: CoreAudio
    device: '{play_esc}'
    channels: 2
pipeline:
- type: Filter
  channels:
  - 0
  names:
  - fir_L
- type: Filter
  channels:
  - 1
  names:
  - fir_R
filters:
  fir_L:
    type: Conv
    parameters:
      type: Wav
      filename: "{left_esc}"
      channel: 0
  fir_R:
    type: Conv
    parameters:
      type: Wav
      filename: "{right_esc}"
      channel: 0
"""
        )
        try:
            CONFIG_INTERNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_INTERNAL_PATH, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            messagebox.showinfo("Saved", f"Wrote settings into:\n{CONFIG_INTERNAL_PATH}")
        except Exception as e:
            messagebox.showerror("Write failed", str(e))

    # ---------- devices / SR ----------
    def refresh_devices(self, mode="all"):
        self.append_log(f"Refreshing {mode} devices...\n")
        cam_bin = which("camilladsp")
        if cam_bin is None:
            messagebox.showerror("Missing internal binary", "Bundled camilladsp not found."); return

        cap, play = [], []
        if mode in ["all", "input", "output"]:
            test_cfg = write_test_config()
            out_cam, _, rc_cam = run([cam_bin, test_cfg, "--list-devices"])
            if out_cam and rc_cam == 0:
                cap_list, play_list = parse_devices(out_cam)
                cap.extend(cap_list); play.extend(play_list)

        out_sp, _, _ = run(["system_profiler", "SPAudioDataType"])
        if out_sp:
            current_device, has_input, has_output = None, False, False
            for line in out_sp.splitlines():
                line = line.strip()
                if not line or line in ("Audio:", "Devices:"): continue
                if line.endswith(":"):
                    if current_device:
                        if has_input and mode in ["all","input"] and current_device not in cap: cap.append(current_device)
                        if has_output and mode in ["all","output"] and current_device not in play: play.append(current_device)
                    current_device, has_input, has_output = line[:-1].strip(), False, False
                    continue
                if "Input Channels:" in line:  has_input = True
                elif "Output Channels:" in line: has_output = True
            if current_device:
                if has_input and mode in ["all","input"] and current_device not in cap: cap.append(current_device)
                if has_output and mode in ["all","output"] and current_device not in play: play.append(current_device)

        if (not cap or not play) and mode != "none":
            sa = which("SwitchAudioSource")
            if sa:
                if not cap and mode in ["all","input"]:
                    out_in, _, _ = run([sa, "-a", "-t", "input"])
                    cap.extend(l.strip() for l in (out_in or "").splitlines() if l.strip())
                if not play and mode in ["all","output"]:
                    out_out, _, _ = run([sa, "-a", "-t", "output"])
                    play.extend(l.strip() for l in (out_out or "").splitlines() if l.strip())

        cap  = list(dict.fromkeys(cap))
        play = list(dict.fromkeys(play))

        if mode in ["all", "input"]:
            if not cap and mode != "none":
                messagebox.showwarning("No input devices", "No input devices found.")
            else:
                self.cap_combo['values'] = cap
                current = self.cap_var.get()
                if current in cap: self.cap_var.set(current)
                else:
                    try: idx = cap.index("BlackHole 2ch")
                    except Exception: idx = 0
                    if cap: self.cap_var.set(cap[idx])

        if mode in ["all", "output"]:
            if not play and mode != "none":
                messagebox.showwarning("No output devices", "No output devices found.")
            else:
                self.play_combo['values'] = play
                current = self.play_var.get()
                if current in play: self.play_var.set(current)
                else:
                    guess = None
                    for name in play:
                        if "USB" in name and ("DAC" in name or "CODEC" in name or "Codec" in name or "Audio" in name):
                            guess = name; break
                    try: idx = play.index(guess) if guess in play else 0
                    except Exception: idx = 0
                    if play: self.play_var.set(play[idx])

        cfg, _ = _read_yaml_config(CONFIG_INTERNAL_PATH)
        if cfg:
            try:
                devs = cfg.get("devices", {})
                sr = devs.get("samplerate")
                if isinstance(sr, int) and str(sr) in ("44100","48000"):
                    self.sr_var.set(str(sr)); self.sr_status_var.set(f"(loaded {sr}Hz from config)")
                cap_name  = devs.get("capture", {}).get("device")
                play_name = devs.get("playback", {}).get("device")
                if cap_name:  self.cap_var.set(cap_name)
                if play_name: self.play_var.set(play_name)
                filters = cfg.get("filters", {})
                fir_L = filters.get("fir_L", {}).get("parameters", {}).get("filename")
                fir_R = filters.get("fir_R", {}).get("parameters", {}).get("filename")
                if fir_L: self.fir_left_var.set(str(fir_L))
                if fir_R: self.fir_right_var.set(str(fir_R))
            except Exception:
                pass

    # ---------- devices SR helpers ----------
    def get_device_sample_rate(self, device_name):
        out_sp, _, _ = run(["system_profiler", "SPAudioDataType"])
        if out_sp:
            found = False
            for line in out_sp.splitlines():
                line = line.strip()
                if line.endswith(":"): found = (line[:-1].strip() == device_name)
                elif found and "Current SampleRate:" in line:
                    try: return int(line.split(":")[-1].strip())
                    except ValueError: pass
        return None

    def set_device_sample_rate(self, device_name, sample_rate):
        sa = which("SwitchAudioSource")
        if not sa: return False, "SwitchAudioSource not installed"
        max_attempts = 3; current_rate = None
        for attempt in range(max_attempts):
            _, _, rc = run([sa, "-r", str(sample_rate), "-n", device_name])
            if rc == 0:
                time.sleep(1.0)
                current_rate = self.get_device_sample_rate(device_name)
                if current_rate == sample_rate: return True, "OK"
            if attempt < max_attempts - 1:
                run([sa, "-t", "output", "-s", device_name]); time.sleep(0.5)
                run([sa, "-r", str(sample_rate), "-n", device_name]); time.sleep(1.0)
                current_rate = self.get_device_sample_rate(device_name)
                if current_rate == sample_rate: return True, "OK"
        return False, f"Failed after {max_attempts} attempts ({current_rate}Hz)"

    def apply_sample_rate(self):
        try: new_rate = int(self.sr_var.get())
        except ValueError:
            messagebox.showerror("Invalid Rate", "Please select a valid sample rate"); return
        was_running = self.proc is not None
        if was_running:
            self.append_log("Stopping CamillaDSP...\n")
            try:
                self.proc.terminate()
                try: self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired: self.proc.kill()
            except Exception as e:
                self.append_log(f"Stop error: {e}\n")
            self.proc = None; self.proc_mode = None; self._sync_launch_btn()

        self.sr_status_var.set("Applying...")
        self.update_idletasks()
        errs = []
        for dev_name, label in [(self.cap_var.get(), "Input"), (self.play_var.get(), "Output")]:
            if dev_name:
                ok, msg = self.set_device_sample_rate(dev_name, new_rate)
                if not ok: errs.append(f"{label}: {msg}")
        if errs:
            self.sr_status_var.set("Some failed!"); messagebox.showerror("Sample Rate", "\n".join(errs))
        else:
            self.sr_status_var.set(f"{new_rate}Hz set")
        if _AUDIO_OK and _PLOT_OK and self._an_visible:
            self.analyzer.stop(); self.analyzer.start()
        if was_running:
            self._auto_start_bypass()

    # ---------- FIR pickers ----------
    def browse_fir_separate(self, channel):
        path = filedialog.askopenfilename(title=f"Select {channel} channel FIR WAV", filetypes=[("WAV files", "*.wav"), ("All files", "*")])
        if path:
            if channel == "left": self.fir_left_var.set(path)
            else: self.fir_right_var.set(path)

    # ---------- routing / camilla ----------
    def switch_output(self):
        cap = self.cap_var.get()
        if not cap:
            messagebox.showerror("No capture device", "Select a capture device first"); return
        ok, msg = maybe_switch_output_to_blackhole(cap)
        if ok: messagebox.showinfo("Switched", msg)
        else:  messagebox.showwarning("Notice", msg)

    def _ensure_bypass_config(self):
        try: sr = int(self.sr_var.get() or 48000)
        except Exception: sr = 48000
        _ensure_impulse_wav(IMPULSE_L_PATH, sr)
        _ensure_impulse_wav(IMPULSE_R_PATH, sr)
        cap  = self.cap_var.get()  or "BlackHole 2ch"
        play = self.play_var.get() or "USB Audio CODEC "
        cap_esc  = cap.replace('"', '\\"'); play_esc = play.replace("'", "''")
        yaml_bypass = (
f"""devices:
  samplerate: {sr}
  chunksize: 1024
  capture:
    type: CoreAudio
    device: "{cap_esc}"
    channels: 2
  playback:
    type: CoreAudio
    device: '{play_esc}'
    channels: 2
pipeline:
- type: Filter
  channels:
  - 0
  names:
  - fir_L
- type: Filter
  channels:
  - 1
  names:
  - fir_R
filters:
  fir_L:
    type: Conv
    parameters:
      type: Wav
      filename: "{str(IMPULSE_L_PATH)}"
      channel: 0
  fir_R:
    type: Conv
    parameters:
      type: Wav
      filename: "{str(IMPULSE_R_PATH)}"
      channel: 0
"""
        )
        try:
            BYPASS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(BYPASS_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(yaml_bypass)
        except Exception as e:
            self.append_log(f"Bypass write error: {e}\n")

    def _auto_start_bypass(self):
        if self.proc is not None: return
        cam_bin = which("camilladsp")
        if cam_bin is None: return
        if not BYPASS_CONFIG_PATH.exists(): self._ensure_bypass_config()
        cmd = [cam_bin, str(BYPASS_CONFIG_PATH), "-v", "-p", "1234", "-w"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.append_log("Started CamillaDSP (bypass)\n")
            self.proc_mode = 'bypass'
            self._sync_launch_btn()
            self.proc_thread = threading.Thread(target=self._reader_thread, daemon=True); self.proc_thread.start()
            self.after(100, self._poll_log_queue)
        except Exception as e:
            self.append_log(f"Bypass start error: {e}\n")
            self.proc = None
            self.proc_mode = None
            self._sync_launch_btn()

    def toggle_launch(self):
        if self.proc_mode != 'correction':
            self._stop_if_running()
            cam_bin = which("camilladsp")
            if cam_bin is None:
                messagebox.showerror("Missing internal binary", "Bundled camilladsp not found."); return
            if not CONFIG_INTERNAL_PATH.exists():
                messagebox.showerror("Config missing", f"Config not found: {CONFIG_INTERNAL_PATH}"); return
            cmd = [cam_bin, str(CONFIG_INTERNAL_PATH), "-v", "-p", "1234", "-w"]
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.append_log("Started CamillaDSP (FIR correction)\n")
                self.proc_mode = 'correction'
                self._sync_launch_btn()
                self.proc_thread = threading.Thread(target=self._reader_thread, daemon=True); self.proc_thread.start()
                self.after(100, self._poll_log_queue)
            except Exception as e:
                messagebox.showerror("Start failed", str(e))
                self.proc = None
                self.proc_mode = None
                self._sync_launch_btn()
        else:
            self._stop_if_running()
            self.proc_mode = None
            self._sync_launch_btn()
            self._auto_start_bypass()

    def _stop_if_running(self):
        if self.proc is None: return
        try:
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self.proc.kill()
        except Exception as e:
            self.append_log(f"Stop error: {e}\n")
        finally:
            self.proc = None

    def _reader_thread(self):
        p = self.proc
        if p is None or p.stdout is None: return
        for line in iter(p.stdout.readline, ''):
            if not line: break
            self.log_queue.put(line)
        rc = p.poll(); self.log_queue.put(f"[process exited with code {rc}]\n")

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.append_log(line)
        except queue.Empty:
            pass
        if self.proc is not None:
            self.after(100, self._poll_log_queue)

    # ---------- misc ----------
    def on_sr_change(self):
        new_rate = self.sr_var.get()
        self.sr_status_var.set(f"Click Apply to set {new_rate}Hz")

    def on_close(self):
        try:
            if hasattr(self, "analyzer"): self.analyzer.stop()
            if hasattr(self, "level_tap"): self.level_tap.stop()
        except Exception: pass
        if _WS_OK: self.ws.stop()
        if self.proc is not None:
            if messagebox.askyesno("Quit", "CamillaDSP is running. Stop and quit?"):
                try:
                    self.proc.terminate()
                    try: self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired: self.proc.kill()
                except Exception:
                    pass
            else:
                return
        self.destroy()

# ================================ Main ================================
def main():
    app = FIRFilterGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
