import os
import sys
import re
import json
import math
import subprocess
import threading
import queue
import time
from pathlib import Path
import wave
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import scipy.signal

__version__ = "0.1.3"



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


# op deps
_AUDIO_OK = True
_PLOT_OK  = True

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


# ====================== Paths / resources ======================
def resource_path(rel):
    """Resolve a path relative to the bundle root (PyInstaller) or this file."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel
CONFIG_INTERNAL_PATH = resource_path('resources/config.yml')
BYPASS_CONFIG_PATH   = resource_path('resources/config_bypass.yml')
TEST_CONFIG_INTERNAL_PATH = resource_path('resources/test_config.yml')
IMPULSE_L_PATH = resource_path('resources/impulse_L.wav')
IMPULSE_R_PATH = resource_path('resources/impulse_R.wav')


# Subprocess helpers
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




# (bypass)
def _ensure_impulse_wav(path: Path, samplerate: int, n_samples: int = 2048):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        frames = [32767] + [0] * (max(1, n_samples) - 1)
        raw = b"".join(struct.pack("<h", s) for s in frames)
        wf.writeframes(raw)


def _db_to_linear(db_value):
    try:
        return float(10.0 ** (float(db_value) / 20.0))
    except Exception:
        return 1.0


def _read_wav_float(path_str):
    try:
        with wave.open(path_str, "rb") as wf:
            nchan = wf.getnchannels()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            raw = wf.readframes(n)
            if sw == 2:
                data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            elif sw == 3:
                a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                s = (
                    a[:, 0].astype(np.int32)
                    | (a[:, 1].astype(np.int32) << 8)
                    | (a[:, 2].astype(np.int32) << 16)
                )
                s[s >= 0x800000] -= 0x1000000
                data = s.astype(np.float64) / 8388608.0
            elif sw == 4:
                try:
                    data = np.frombuffer(raw, dtype="<f4").astype(np.float64)
                    if np.max(np.abs(data)) > 2.0:
                        raise ValueError()
                except Exception:
                    data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
            else:
                data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
                data = (data - 128.0) / 128.0
            if nchan > 1:
                data = data.reshape(-1, nchan)[:, 0]
            return data
    except Exception:
        return None



class Analyzer(ttk.LabelFrame):
    def __init__(self, master, get_capture_name, get_playback_name, get_samplerate, process_monitor_chunk=None):
        super().__init__(master)

        # getters
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate    = get_samplerate
        self.process_monitor_chunk = process_monitor_chunk

        # state
        self.running = False
        self.stream = None
        self.buffer = queue.Queue(maxsize=24)
        self.update_job = None
        self.loopback_unavailable = False
        self.using_fallback_input_for_output = False

        # Keep analyzer focused on the practical operating window.
        self.y_min_db = -72
        self.y_max_db = 0

        # last
        self._last_freqs = None
        self._smoothed_mag = None
        self._peak_hold = None
        self._peak_decay = 0.5  # dB per frame
        self._accum = []   # rolling sample accumulator for visualizer

        # opts
        # display scale (new!)
        self.display_gain_db = tk.DoubleVar(value=0.0)
        self._show_smooth_var = tk.BooleanVar(value=True)
        self._show_raw_var = tk.BooleanVar(value=True)

        # dsp knobs
        self._fft_size = 8192
        self._smooth_var_default = "1/16"
        self._avg_mode = "Medium"
        self._tilt_var = tk.StringVar(value="Off")
        self._tau_map = {
            "Realtime": 0.02,
            "Fast":     0.06,
            "Medium":   0.18,
            "Slow":     0.45,
        }

        self._power_ema = None
        self._last_update_t = None
        self._smooth_idx_l = None
        self._smooth_idx_r = None
        self._smooth_cached_fft = None
        self._smooth_cached_frac = None
        self._last_data_t = None

        # ui header
        status_row = ttk.Frame(self); status_row.pack(fill=tk.X, padx=6, pady=(6, 4))
        self._monitor_label_text = tk.StringVar(value="")
        ttk.Label(status_row, textvariable=self._monitor_label_text).pack(side=tk.LEFT)

        # controls
        controls = ttk.Frame(self); controls.pack(fill=tk.X, padx=6, pady=(0, 4))

        ttk.Label(controls, text="FFT").pack(side=tk.LEFT, padx=(0,4))
        self._fft_var = tk.StringVar(value=str(self._fft_size))
        fft_box = ttk.Combobox(controls, width=6, state="readonly",
                               values=("4096","8192","16384","32768"),
                               textvariable=self._fft_var)
        fft_box.pack(side=tk.LEFT, padx=(0,8))
        fft_box.bind("<<ComboboxSelected>>", lambda e: self._on_fft_change())

        ttk.Label(controls, text="Smoothing").pack(side=tk.LEFT, padx=(0,4))
        self._smooth_var = tk.StringVar(value=self._smooth_var_default)
        smooth_box = ttk.Combobox(controls, width=8, state="readonly",
                                  values=("None","1/128","1/32","1/16","1/8","1/6","1/3","1/2"),
                                  textvariable=self._smooth_var)
        smooth_box.pack(side=tk.LEFT, padx=(0,8))
        smooth_box.bind("<<ComboboxSelected>>", lambda e: self._on_smooth_change())

        ttk.Label(controls, text="Averaging").pack(side=tk.LEFT, padx=(0,4))
        self._avg_var = tk.StringVar(value=self._avg_mode)
        avg_box = ttk.Combobox(controls, width=10, state="readonly",
                               values=("Realtime","Fast","Medium","Slow"),
                               textvariable=self._avg_var)
        avg_box.pack(side=tk.LEFT, padx=(0,8))
        avg_box.bind("<<ComboboxSelected>>", lambda e: self._on_avg_change())

        ttk.Label(controls, text="Tilt").pack(side=tk.LEFT, padx=(0,4))
        tilt_box = ttk.Combobox(controls, width=8, state="readonly",
                                values=("Off","3 dB/oct","4.5 dB/oct","6 dB/oct"),
                                textvariable=self._tilt_var)
        tilt_box.pack(side=tk.LEFT, padx=(0,8))
        tilt_box.bind("<<ComboboxSelected>>", lambda e: self._on_tilt_change())


        
        ttk.Checkbutton(
            controls,
            text="Smoothed",
            variable=self._show_smooth_var
        ).pack(side=tk.LEFT, padx=(0,8))

        ttk.Checkbutton(
            controls,
            text="Raw FFT",
            variable=self._show_raw_var
        ).pack(side=tk.LEFT, padx=(0,8))

        self._freeze = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            controls,
            text="Freeze",
            variable=self._freeze
        ).pack(side=tk.LEFT, padx=(0,8))

        ttk.Label(controls, text="Window").pack(side=tk.LEFT, padx=(0,4))

        self._window_var = tk.StringVar(value="Hann")
        window_box = ttk.Combobox(
            controls,
            width=10,
            state="readonly",
            values=("Hann", "Blackman-Harris", "Flat-top", "Rect"),
            textvariable=self._window_var
        )
        window_box.pack(side=tk.LEFT, padx=(0,8))
        

        # Display Gain slider (kept for debug, but hidden from UI)

        # plot area
        body = ttk.Frame(self); body.pack(fill=tk.BOTH, expand=True)
        if _AUDIO_OK and _PLOT_OK:
            self.fig = Figure(figsize=(8.5, 3.2), dpi=100, facecolor="#1a1a1a")
            self.fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.12)
            self.ax  = self.fig.add_subplot(111, facecolor="#1a1a1a")
            self.ax.set_xscale('log'); self._set_axes_limits_once()
            self.ax.grid(True, which='both', alpha=0.28, color="#4d4d4d")
            self.ax.tick_params(labelbottom=True, labelleft=True, colors="#f0f0f0", labelsize=10)
            yt = [-72, -60, -48, -36, -24, -12, 0]
            self.ax.set_yticks(yt); self.ax.set_yticklabels([str(v) for v in yt])
            for spine in self.ax.spines.values(): spine.set_color("#7a7a7a")
            xticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            self.ax.set_xticks(xticks)
            self.ax.get_xaxis().set_major_formatter(lambda x, pos: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}")
            self._add_band_spans()
            (self.line,) = self.ax.plot([], [], linewidth=1.7, color="#42b8ff")
            (self.raw_line,) = self.ax.plot([], [], linewidth=1.0, alpha=0.5, color="#4fef0a")
            (self.ref_line,) = self.ax.plot([], [], linewidth=1.3, alpha=0.85, color="#ffc857")
            (self.peak_line,) = self.ax.plot([], [], linewidth=1.2, color="#ff3b3b", alpha=0.9)
            self.fill_poly = None
            self.canvas = FigureCanvasTkAgg(self.fig, master=body)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(6, 3), pady=(0, 6))
            self.overlay = tk.Label(self.canvas_widget, text="Output spectrum unavailable", fg="#f0f0f0", bg="#1a1a1a")
            self.overlay.place_forget()
        else:
            ttk.Label(self, text="Install: pip install numpy sounddevice matplotlib", foreground="orange").pack(fill=tk.X, padx=6, pady=8)

        self._fir_freqs = None
        self._fir_magdb = None

    def _add_band_spans(self):
        bands = [
            (20,   60,  (0.78, 0.35, 0.95, 0.12)),
            (60,  250,  (0.10, 0.82, 0.48, 0.12)),
            (250, 500,  (0.98, 0.70, 0.18, 0.12)),
            (500, 2000, (0.98, 0.34, 0.34, 0.10)),
            (2000,6000, (0.36, 0.62, 0.98, 0.10)),
            (6000,20000,(0.15, 0.88, 0.98, 0.10)),
        ]
        for a,b,c in bands:
            self.ax.axvspan(a, b, color=c, linewidth=0)

    def start(self):
        if self.running or not _AUDIO_OK or not _PLOT_OK:
            return
        sr = self._safe_sr()
        if not sr:
            return
    
        self._smoothed_mag = None
        self._last_freqs = None
        self._power_ema = None
        self._last_update_t = None
        self._accum = []
        self.loopback_unavailable = False
        self.using_fallback_input_for_output = False
        self._last_data_t = time.monotonic()
    
        cap_name = ""
        try:
            cap_name = (self.get_capture_name() or "")
        except Exception:
            cap_name = ""

        dev_idx = _sd_find_device_index(cap_name, 'input')
        if dev_idx is None:
            self._monitor_label_text.set("Camilla output analysis: (no valid capture input)")
            if _PLOT_OK:
                self.overlay.config(text="No valid capture input selected")
                self.overlay.place(relx=0.5, rely=0.5, anchor="center")
            return

        self._monitor_label_text.set(f"Metering from: {cap_name}")
        self._update_overlay()
    
        def cb(indata, frames, time_info, status):
            x = indata
            if x.ndim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[:, :min(2, x.shape[1])]
            if self.process_monitor_chunk is not None:
                try:
                    x = self.process_monitor_chunk(x)
                except Exception:
                    pass
            if x.ndim > 1:
                x = np.mean(x[:, :min(2, x.shape[1])], axis=1)
            self._last_data_t = time.monotonic()
            try:
                self.buffer.put_nowait(x.astype(np.float64, copy=False))
            except queue.Full:
                pass
    
        try:
            try:
                max_ch = max(1, int(sd.query_devices(dev_idx).get('max_input_channels', 1)))
            except Exception:
                max_ch = 1
            stream_channels = 2 if max_ch >= 2 else 1
            self.stream = sd.InputStream(
                device=dev_idx,
                channels=stream_channels,
                samplerate=sr,
                blocksize=256,
                dtype='float32',
                callback=cb
            )
            self.stream.start()
        except Exception as e:
            self.stream = None
            if _PLOT_OK:
                self.overlay.config(text=f"Failed to open input: {e}")
                self.overlay.place(relx=0.5, rely=0.5, anchor="center")
            return
    
        self.running = True
        self._schedule_update(self._fft_size, sr, period_ms=16)
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
            self.raw_line.set_data([], [])
            self.ref_line.set_data([], [])
            if self.fill_poly is not None: self.fill_poly.remove(); self.fill_poly = None
            self.canvas.draw_idle()
        self._update_overlay()

    def _restart_stream(self):
        if not self.running:
            return
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None
        self.running = False
        self.start()

    def _fft_dbfs(self, x, sr, N=None):
        if x is None or len(x) == 0:
            freqs = np.array([20.0])
            db = np.array([self.y_min_db])
            return freqs, db

        N = self._fft_size

        x = x[:N] if len(x) >= N else np.pad(x, (0, N - len(x)))

        x = x - np.mean(x)

        win_type = self._window_var.get()

        if win_type == "Hann":
            w = np.hanning(N)
        elif win_type == "Blackman-Harris":
            w = np.blackman(N)
        elif win_type == "Flat-top":
            w = scipy.signal.windows.flattop(N)
        else:  # Rect
            w = np.ones(N)
        xw = x * w

        X = np.fft.rfft(xw)
        freqs = np.fft.rfftfreq(N, 1.0 / sr)

        #TRUE amplitude correction for Hann window
        window_gain = np.sum(w) / N

        mag = np.abs(X) / (N * window_gain / 2.0)

        mag = np.maximum(mag, 1e-12)

        db = 20.0 * np.log10(mag / 1.0)  # 1.0 = full scale reference

        if np.max(np.abs(x)) < 1e-5:
            db[:] = self.y_min_db

        return freqs, db

    def _prepare_smoothing(self, freqs, fraction):
        if (self._smooth_cached_fft == len(freqs)) and (self._smooth_cached_frac == fraction):
            return
        f = np.clip(freqs, 20.0, None)
        logf = np.log(f)
        half = 0.5 * math.log(2.0) / fraction
        left = logf - half
        right = logf + half
        l_idx = np.searchsorted(logf, left, side='left')
        r_idx = np.searchsorted(logf, right, side='right')
        l_idx = np.clip(l_idx, 0, len(freqs)-1)
        r_idx = np.clip(r_idx, 1, len(freqs))
        self._smooth_idx_l = l_idx.astype(np.int32)
        self._smooth_idx_r = r_idx.astype(np.int32)
        self._smooth_cached_fft = len(freqs)
        self._smooth_cached_frac = fraction

    def _fractional_octave_power_smooth_fast(self, freqs, db, fraction):
        if not fraction or fraction == 0:
            return db
        self._prepare_smoothing(freqs, fraction)
        p = 10**(db/10.0)
        csum = np.concatenate(([0.0], np.cumsum(p)))
        l = self._smooth_idx_l
        r = self._smooth_idx_r
        count = (r - l).astype(np.float64)
        count[count <= 0] = 1.0
        p_mean = (csum[r] - csum[l]) / count
        return 10.0 * np.log10(np.maximum(p_mean, 1e-24))

    def _apply_tilt(self, freqs, db):
        tilt_map = {
            "Off": 0.0,
            "3 dB/oct": 3.0,
            "4.5 dB/oct": 4.5,
            "6 dB/oct": 6.0,
        }
        tilt_db_per_oct = tilt_map.get(self._tilt_var.get(), 0.0)
        if tilt_db_per_oct == 0.0:
            return db
        octaves = np.log2(np.maximum(freqs, 20.0) / 1000.0)
        return db + (octaves * tilt_db_per_oct)

    def _hearing_weighting(self, freqs, phon_level=70):
        f = np.maximum(freqs, 20.0)
        a = np.array([20, 50, 100, 200, 400, 800, 1600, 3150, 6300, 12500])
        w70 = np.array([-31, -23, -17, -10, -6, -3, -1, 0, -3, -8])
        return np.interp(np.log10(f), np.log10(a), w70, left=w70[0], right=w70[-1])

    def _safe_sr(self):
        try: return int(self.get_samplerate())
        except Exception: return None

    def _schedule_update(self, fft_size, sr, period_ms=16):
        if not self.running: return
        if self._last_data_t is not None and (time.monotonic() - self._last_data_t) > 1.5:
            self._restart_stream()
            return
        now = time.monotonic()
        if self._last_update_t is None:
            self._last_update_t = now
        dt = now - self._last_update_t
        self._last_update_t = now
        self._update_plot(fft_size, sr, dt)
        self.update_job = self.after(period_ms, lambda: self._schedule_update(fft_size, sr, period_ms))

    def _set_axes_limits_once(self):
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(self.y_min_db, self.y_max_db)

    def _update_fill(self, freqs, y):
        verts = np.column_stack([freqs, y]); bottom = self.y_min_db
        poly = np.vstack([[freqs[0], bottom], verts, [freqs[-1], bottom]])
        if self.fill_poly is None:
            self.fill_poly = PolyCollection([poly], facecolor=(0.26, 0.58, 1.0, 0.18), edgecolor='none')
            self.ax.add_collection(self.fill_poly)
        else:
            self.fill_poly.set_verts([poly])

    def _update_plot(self, fft_size, sr, dt):
        if self._freeze.get():
            return

        if not _PLOT_OK:
            return

        # Drain all available blocks into the rolling accumulator
        while not self.buffer.empty():
            try:
                chunk = self.buffer.get_nowait()
                self._accum.extend(chunk.tolist())
            except Exception:
                break

        # Keep only what we need (4x fft_size rolling window)
        max_keep = self._fft_size * 4
        if len(self._accum) > max_keep:
            del self._accum[:len(self._accum) - max_keep]

        hop = self._fft_size // 4  # 75% overlap

        if not hasattr(self, "_fft_cursor"):
            self._fft_cursor = 0

        if len(self._accum) < self._fft_size:
            return

        # advance cursor
        if self._fft_cursor + self._fft_size > len(self._accum):
            self._fft_cursor = max(0, len(self._accum) - self._fft_size)

        x = np.array(
            self._accum[self._fft_cursor:self._fft_cursor + self._fft_size],
            dtype=np.float64
        )

        self._fft_cursor += hop
        
        result = self._fft_dbfs(x, sr)
        if result is None:
            return
        freqs, mag_db_raw = result


        

        # --- smoothing ---
        smooth_map = {"None":0,"1/128":128,"1/32":32,"1/16":16,"1/8":8,"1/6":6,"1/3":3,"1/2":2}
        frac = smooth_map.get(self._smooth_var.get(), 0)
        mag_db = self._fractional_octave_power_smooth_fast(freqs, mag_db_raw, fraction=frac)

        # --- averaging ---
        tau = self._tau_map.get(self._avg_mode, 0.02)
        alpha = 1.0 - math.exp(-max(dt, 1e-3) / max(tau, 1e-3))

        if self._smoothed_mag is None or self._smoothed_mag.shape != mag_db.shape:
            self._smoothed_mag = mag_db.copy()
        else:
            self._smoothed_mag = alpha * mag_db + (1.0 - alpha) * self._smoothed_mag

        mag_db = self._smoothed_mag

        

        if self._peak_hold is None or self._peak_hold.shape != mag_db.shape:
            self._peak_hold = mag_db.copy()
        else:
            self._peak_hold = np.maximum(self._peak_hold - self._peak_decay, mag_db)

        mag_db_raw_disp = np.clip(self._apply_tilt(freqs, mag_db_raw), self.y_min_db, self.y_max_db)
        mag_db_disp = np.clip(self._apply_tilt(freqs, mag_db), self.y_min_db, self.y_max_db)

        # --- toggle behavior ---
        if self._show_smooth_var.get():
            self.line.set_data(freqs, mag_db_disp)
        else:
            self.line.set_data([], [])

        if self._show_raw_var.get():
            self.raw_line.set_data(freqs, mag_db_raw_disp)
        else:
            self.raw_line.set_data([], [])

        # --- draw ---
        self._last_freqs = freqs
        self._smoothed_mag = mag_db

        if self._show_smooth_var.get():
            self._update_fill(freqs, mag_db_disp)
        elif self.fill_poly is not None:
            self.fill_poly.remove()
            self.fill_poly = None
        self._draw_reference_curve(freqs)
        self.canvas.draw_idle()

    def _update_overlay(self):
        if not _PLOT_OK:
            return
        if not self.running or (self.stream is None):
            self.overlay.config(text="Analyzer idle")
            self.overlay.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.overlay.place_forget()

    def _on_fft_change(self):
        try:
            self._fft_size = int(self._fft_var.get())
            self._smooth_cached_fft = None
            self._smoothed_mag = None
            self._last_freqs = None
            self._accum = []
        except Exception:
            pass

    def _on_smooth_change(self):
        self._smooth_cached_frac = None
        self._smoothed_mag = None

    def _on_avg_change(self):
        self._avg_mode = self._avg_var.get()
        self._smoothed_mag = None

    def _on_tilt_change(self):
        try:
            self._smoothed_mag = None
            self.canvas.draw_idle()
        except Exception:
            pass

    

    def set_reference_curve(self, freqs, mag_db):
        try:
            self._fir_freqs = np.asarray(freqs)
            self._fir_magdb = np.asarray(mag_db)
        except Exception:
            pass

    def _draw_reference_curve(self, freqs_display):
        if self._fir_freqs is None or self._fir_magdb is None or freqs_display is None:
            self.ref_line.set_data([], [])
            return
        try:
            # Interpolate the stored FIR curve onto current analyzer frequencies
            f = np.clip(freqs_display, 20.0, self._fir_freqs[-1] if self._fir_freqs.size else 20000.0)
            y = np.interp(np.log10(f), np.log10(np.maximum(self._fir_freqs, 20.0)), self._fir_magdb)
            self.ref_line.set_data(freqs_display, np.clip(self._apply_tilt(freqs_display, y), self.y_min_db, self.y_max_db))
        except Exception:
            self.ref_line.set_data([], [])

    def clear_reference_curve(self):
        self._fir_freqs = None
        self._fir_magdb = None
        try:
            self.ref_line.set_data([], [])
            self.canvas.draw_idle()
        except Exception:
            pass

class Meter(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.db_min = -72.0
        self.db_max = 0.0
        self.warn_yellow_db = -6.0
        self.warn_red_db = -3.0
        self.peak_hold = None
        self.peak_hold_time = time.time()
        self.peak_decay_rate = 20.0
        self.peak_hold_duration = 2.0
        self.canvas = tk.Canvas(self, height=24, highlightthickness=0, bg="#F20E0E")
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
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
        w = max(self.winfo_width(), 620)
        h = 24
        c = self.canvas
        c.configure(height=h)
        c.delete("all")

        # background
        c.create_rectangle(0, 0, w, h, fill="#181818", width=0)

        # safety defaults
        xcur = None

        # draw level bar
        if db_current is not None:
            db_current = max(self.db_min, min(self.db_max, float(db_current)))
            xcur = self._db_to_x(db_current, w)

            warn_yellow = self._db_to_x(self.warn_yellow_db, w)
            warn_red = self._db_to_x(self.warn_red_db, w)

            # GREEN
            c.create_rectangle(0, 0, min(xcur, warn_yellow), h, fill="#2ecc71", width=0)

            # YELLOW
            if xcur > warn_yellow:
                c.create_rectangle(warn_yellow, 0, min(xcur, warn_red), h, fill="#f1c40f", width=0)

            # RED
            if xcur > warn_red:
                c.create_rectangle(warn_red, 0, xcur, h, fill="#e74c3c", width=0)

        # peak logic (unchanged but safe)
        if self.peak_hold is not None:
            now = time.time()
            elapsed = now - self.peak_hold_time
            if elapsed > self.peak_hold_duration:
                self.peak_hold -= self.peak_decay_rate * 2 * (elapsed - self.peak_hold_duration)
                self.peak_hold_time = now
                if self.peak_hold < self.db_min:
                    self.peak_hold = None

        if self.peak_hold is not None:
            xpk = self._db_to_x(self.peak_hold, w)
            c.create_line(xpk, 0, xpk, h, fill="#00ffcc", width=2)

        for t in [-72, -60, -48, -36, -24, -12, 0]:
            xt = self._db_to_x(t, w)
            c.create_line(xt, h - 5, xt, h, fill="#444")


class VerticalMeter(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.db_min = -72.0
        self.db_max = 0.0
        self.warn_yellow_db = -6.0
        self.warn_red_db = -3.0
        self.peak_hold = None
        self.peak_hold_time = time.time()
        self.peak_decay_rate = 20.0
        self.peak_hold_duration = 2.0
        self.canvas = tk.Canvas(self, width=20, height=110, highlightthickness=0, bg="#181818")
        self.canvas.grid(row=0, column=0, sticky="ns")
        self.label_row = tk.Canvas(self, width=34, height=110, highlightthickness=0, bg="#181818")
        self.label_row.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        self.canvas.bind("<Button-1>", lambda e: self._reset_peak())
        self.bind("<Configure>", lambda e: self._redraw())

    def draw_meter(self, db_current, db_peak):
        now = time.time()
        if db_peak is not None:
            db_peak = max(self.db_min, min(self.db_max, float(db_peak)))
            if self.peak_hold is None or db_peak > self.peak_hold:
                self.peak_hold = db_peak
                self.peak_hold_time = now
        if db_current is not None:
            db_current = max(self.db_min, min(self.db_max, float(db_current)))
        self._redraw(db_current)

    def _reset_peak(self):
        self.peak_hold = None
        self.peak_hold_time = time.time()
        self._redraw()

    def _db_to_y(self, db, h):
        db = max(self.db_min, min(self.db_max, db))
        frac = (db - self.db_min) / (self.db_max - self.db_min)
        return int(h - (frac * h))

    def _redraw(self, db_current=None):
        self.update_idletasks()
        w = 20
        h = 110
        c = self.canvas
        c.configure(width=w, height=h)
        c.delete("all")
        c.create_rectangle(0, 0, w, h, fill="#181818", width=0)

        if db_current is not None:
            ycur = self._db_to_y(db_current, h)
            yy = self._db_to_y(self.warn_yellow_db, h)
            yr = self._db_to_y(self.warn_red_db, h)
            c.create_rectangle(0, ycur, w, h, fill="#2ecc71", width=0)
            if ycur < yy:
                c.create_rectangle(0, ycur, w, yy, fill="#f1c40f", width=0)
            if ycur < yr:
                c.create_rectangle(0, ycur, w, yr, fill="#e74c3c", width=0)

        if self.peak_hold is not None:
            now = time.time()
            elapsed = now - self.peak_hold_time
            if elapsed > self.peak_hold_duration:
                self.peak_hold -= self.peak_decay_rate * 2 * (elapsed - self.peak_hold_duration)
                self.peak_hold_time = now
                if self.peak_hold < self.db_min:
                    self.peak_hold = None

        if self.peak_hold is not None:
            ypk = self._db_to_y(self.peak_hold, h)
            c.create_line(0, ypk, w, ypk, fill="#6f9a18", width=2)

        lr = self.label_row
        lr.configure(width=34, height=h)
        lr.delete("all")
        for t in [0, -6, -12, -18, -24, -36, -48, -60, -72]:
            yt = self._db_to_y(t, h)
            lr.create_line(0, yt, 7, yt, fill="#444")
            lr.create_text(10, yt, text=("0" if t == 0 else str(t)), anchor="w", fill="#b0b0b0", font=("TkDefaultFont", 7, "bold"))
class LevelTap:
    def __init__(self, get_capture_name, get_playback_name, get_samplerate):
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate    = get_samplerate
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
        try: cap_name  = (self.get_capture_name() or "")
        except Exception: cap_name = ""
        try: play_name = (self.get_playback_name() or "")
        except Exception: play_name = ""

        dev_idx = _sd_find_device_index(cap_name, 'input')

        def cb(indata, frames, time_info, status):
            x = indata
            if x.ndim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[:, :min(2, x.shape[1])]
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

# YAML read
def _read_yaml_config(path: Path):
    if not _YAML_OK or not path.exists():
        return None, "PyYAML missing or config not found."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, None
    except Exception as e:
        return None, f"YAML read error: {e}"

# GUI
class FIRFilterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"FIRC v{__version__}")
        self.resizable(True, True)

        self.proc = None
        self.proc_thread = None
        self.proc_mode = None
        self._pending_gain_job = None
        self._meter_ir_cache = {"mode": None, "left": None, "right": None, "sig": None}
        self._meter_conv_state = {"left": None, "right": None, "sig": None}
        self._analyzer_conv_state = {"left": None, "right": None, "sig": None}

        self.log_queue = queue.Queue()
        self._log_buffer = []

        self._init_styles()
        self.create_widgets()

        # meter loop
        self.after(50, self._update_bottom_meter)
        self._sync_launch_btn()

        self.sr_status_var.set("Click Apply to set sample rate")
        self.sr_combo.bind('<<ComboboxSelected>>', lambda e: self.on_sr_change())

        # init async
        def init():
            self.refresh_devices("all")
            self._ensure_bypass_config()
            try:
                self.level_tap = LevelTap(
                    get_capture_name=lambda: self.cap_var.get(),
                    get_playback_name=lambda: self.play_var.get(),
                    get_samplerate=lambda: self.sr_var.get()
                )
                self.level_tap.start()
            except Exception as e:
                self.append_log(f"[LevelTap] init warning: {e}\n")
            self._auto_start_bypass()
            self.append_log("Ready.\n")
            # size
            self.update_idletasks()
            self._base_w = self.winfo_width()
            self._base_h = self.winfo_height()
            self._apply_window_mode("none")
        self.after(150, init)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # styles
    def _init_styles(self):
        bg_main = "#1a1a1a"; bg_panel = "#202225"; bg_field = "#2a2d31"
        fg_text = "#f0f0f0"; fg_dim = "#a0a2a5"; fg_invert = "#000000"
        btn_bg = "#353a3f"; btn_hover = "#41464c"; btn_border = "#242525"
        accent_bg = "#42b8ff"; accent_hover = "#2fa6ef"; accent_border = "#5bc1ff"; frame_border = "#3a3d41"
        self.configure(bg=bg_main)
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("Running.TButton", background="#2ecc71", foreground="#000000")
        style.configure("Bypass.TButton", background="#7f8c8d", foreground="#000000")
        style.configure(".", background=bg_main, foreground=fg_text, relief="flat")
        style.configure("TFrame", background=bg_panel)
        style.configure("TLabelFrame", background=bg_panel, foreground=fg_text, bordercolor=frame_border)
        style.configure("TLabel", background=bg_panel, foreground=fg_text)
        style.configure("TEntry", fieldbackground=bg_field, foreground=fg_text, insertcolor=fg_text, bordercolor=frame_border)
        style.configure("TCombobox", fieldbackground=bg_field, background=bg_field, foreground=fg_text, arrowcolor=fg_text, bordercolor=frame_border)
        style.map("TCombobox",
                  fieldbackground=[("readonly", bg_field), ("!disabled", bg_field), ("focus", bg_field)],
                  foreground=[("disabled", fg_dim), ("!disabled", fg_text)],
                  arrowcolor=[("active", fg_text), ("!active", fg_text)],
                  bordercolor=[("focus", accent_border), ("!focus", frame_border)])
        # dropdown list
        self.option_add("*TCombobox*Listbox.background", bg_field)
        self.option_add("*TCombobox*Listbox.foreground", fg_text)
        self.option_add("*TCombobox*Listbox.selectBackground", accent_bg)
        self.option_add("*TCombobox*Listbox.selectForeground", fg_invert)
        self.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.option_add("*TCombobox*Listbox.highlightThickness", 0)
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

    # layout
    def create_widgets(self):
        BW = 18
        self.grid_rowconfigure(0, weight=1); self.grid_columnconfigure(0, weight=1)
        frm = ttk.Frame(self); frm.grid(row=0, column=0, sticky="nsew", padx=10, pady=10); frm.grid_columnconfigure(0, weight=1)

        # devices
        dev_frame = ttk.LabelFrame(frm, text="Devices")
        dev_frame.grid(row=0, column=0, sticky="ew")
        dev_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(dev_frame, text="Capture:").grid(row=0, column=0, sticky="w", padx=(6,5), pady=2)
        self.cap_var = tk.StringVar(); self.cap_combo = ttk.Combobox(dev_frame, textvariable=self.cap_var, width=50, state="readonly")
        self.cap_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Label(dev_frame, text="Playback:").grid(row=1, column=0, sticky="w", padx=(6,5), pady=2)
        self.play_var = tk.StringVar(); self.play_combo = ttk.Combobox(dev_frame, textvariable=self.play_var, width=50, state="readonly")
        self.play_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(dev_frame, text="Refresh Devices", width=BW, command=lambda: self.refresh_devices("all")).grid(row=2, column=1, sticky="e", padx=6, pady=(0,4))

        # FIR
        fir_frame = ttk.LabelFrame(frm, text="FIR Configuration")
        fir_frame.grid(row=1, column=0, sticky="ew", pady=(8,0))
        for c in range(5): fir_frame.grid_columnconfigure(c, weight=1 if c == 1 else 0)

        sr_frame = ttk.Frame(fir_frame); sr_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(sr_frame, text="Samplerate:").pack(side=tk.LEFT)
        self.sr_var = tk.StringVar(value="48000")
        self.sr_combo = ttk.Combobox(sr_frame, textvariable=self.sr_var, values=["44100","48000"], width=10, state="readonly")
        self.sr_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(sr_frame, text="Apply", width=12, command=self.apply_sample_rate).pack(side=tk.LEFT)
        self.sr_status_var = tk.StringVar()
        ttk.Label(sr_frame, textvariable=self.sr_status_var, font=("",9,"italic")).pack(side=tk.LEFT, padx=5)

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

        # action
        action = ttk.Frame(frm); action.grid(row=4, column=0, sticky="ew", pady=(8,0))
        for i in range(7): action.grid_columnconfigure(i, weight=1)
        self.btn_toggle_io   = ttk.Button(action, text="Input Level", command=self._toggle_gain_section)
        self.btn_toggle_io.grid(row=0, column=0, padx=(0,3), sticky="ew")
        ttk.Button(action, text="Show Config",  command=self._toggle_config_exclusive).grid(row=0, column=1, padx=(0,3), sticky="ew")
        ttk.Button(action, text="Write Config", command=self.write_to_config).grid(row=0, column=2, padx=(0,3), sticky="ew")

        self.btn_toggle_vis  = ttk.Button(action, text="Visualizer", command=self._toggle_visualizer_exclusive)
        self.btn_toggle_vis.grid(row=0, column=3, padx=(0,3), sticky="ew")
        self.btn_toggle_logs = ttk.Button(action, text="Logs",       command=self._toggle_logs_exclusive)
        self.btn_toggle_logs.grid(row=0, column=4, padx=(0,3), sticky="ew")

        self.launch_btn = ttk.Button(action, text="Start correction", style="Accent.TButton", command=self.toggle_launch)
        self.launch_btn.grid(row=0, column=6, padx=(3,0), sticky="ew")

        # bypass label
        self.bypass_var = tk.StringVar(value="")
        ttk.Label(action, textvariable=self.bypass_var, font=("", 9, "italic")).grid(row=1, column=6, padx=(6,0), sticky="ne")

        # container
        self.an_container = ttk.Frame(frm)
        self.an_container.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        self.an_container.grid_columnconfigure(0, weight=1)

        # analyzer
        self.an_frame = ttk.LabelFrame(self.an_container)
        self.analyzer  = Analyzer(
            self.an_frame,
            get_capture_name=lambda: self.cap_var.get(),
            get_playback_name=lambda: self.play_var.get(),
            get_samplerate=lambda: self.sr_var.get(),
            process_monitor_chunk=self._analyzer_monitor_chunk
        )
        self.analyzer.pack(fill=tk.BOTH, expand=True)
        self._an_visible = False

        # logs
        self.log_frame = ttk.LabelFrame(self.an_container, text="CamillaDSP logs")
        self.log_text  = scrolledtext.ScrolledText(self.log_frame, state=tk.DISABLED, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        try: self.log_text.configure(background="#1a1a1a", foreground="#f0f0f0", insertbackground="#f0f0f0")
        except Exception: pass
        self._logs_visible = False
        self._flush_log_buffer()

        self.cfg_frame = ttk.LabelFrame(self.an_container, text="Current Config")
        self.cfg_text  = scrolledtext.ScrolledText(self.cfg_frame, state=tk.DISABLED, height=12, wrap=tk.NONE)
        self.cfg_text.pack(fill=tk.BOTH, expand=True)
        try: self.cfg_text.configure(background="#1a1a1a", foreground="#f0f0f0", insertbackground="#f0f0f0")
        except Exception: pass
        self._cfg_visible = False

        # bottom meter
        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=0)
        bottom.grid_columnconfigure(2, weight=1)
        meter_label = ttk.Label(bottom, text="Output Level", font=("TkDefaultFont", 9))
        meter_label.grid(row=0, column=1, sticky="w", padx=(0, 0), pady=(0, 2))
        self.output_meter_wrap = ttk.Frame(bottom, width=720, height=28)
        self.output_meter_wrap.grid(row=1, column=1, sticky="")
        self.output_meter_wrap.grid_propagate(False)
        self.output_meter_wrap.grid_columnconfigure(0, weight=1)
        self.output_meter_wrap.grid_rowconfigure(0, weight=1)
        self.output_meter = Meter(self.output_meter_wrap)
        self.output_meter.grid(row=0, column=0, sticky="nsew")

        # --- Input / Output setup panel ---
        self._gain_visible = False
        self._gain_in_db  = tk.DoubleVar(value=0.0)
        self._gain_out_db = tk.DoubleVar(value=0.0)

        self.gain_section = ttk.LabelFrame(self.an_container, text="Input / Output Setup")

        gs = self.gain_section
        gs.grid_columnconfigure(0, weight=1)
        gs.grid_columnconfigure(1, weight=1)

        def _build_strip(parent, title, meter_attr, slider_attr, label_attr, clip_attr, variable, column):
            strip = ttk.Frame(parent)
            strip.grid(row=0, column=column, sticky="n", padx=18, pady=(8, 6))
            ttk.Label(strip, text=title).grid(row=0, column=0, columnspan=2, pady=(0, 6))
            meter = VerticalMeter(strip)
            meter.grid(row=1, column=0, rowspan=2, sticky="ns", padx=(0, 8))
            setattr(self, meter_attr, meter)
            slider = ttk.Scale(strip, from_=6, to=-18, orient=tk.VERTICAL, length=150,
                               variable=variable, command=self._on_gain_change)
            slider.grid(row=1, column=1, sticky="ns")
            setattr(self, slider_attr, slider)
            label = ttk.Label(strip, text="+0.0 dB", width=8, anchor="center")
            label.grid(row=3, column=0, columnspan=2, pady=(6, 0))
            setattr(self, label_attr, label)
            clip_label = tk.Label(strip, text="", width=8, anchor="center", font=("TkDefaultFont", 10, "bold"), fg="#ff3b3b", bg="#1a1a1a")
            clip_label.grid(row=4, column=0, columnspan=2, pady=(2, 0))
            setattr(self, clip_attr, clip_label)

        _build_strip(gs, "Into CamillaDSP", "pre_meter", "_gain_in_slider", "_gain_in_label", "_clip_in_label", self._gain_in_db, 0)
        _build_strip(gs, "Out To Playback", "post_meter", "_gain_out_slider", "_gain_out_label", "_clip_out_label", self._gain_out_db, 1)

        ttk.Label(gs, text="Peak dBFS into convolution and out to playback. Click a meter to reset peak.",
                  font=("", 9, "italic"), foreground="#a0a2a5").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0,8))

        # start compact
        self.an_frame.grid_remove()
        self.log_frame.grid_remove()
        self.cfg_frame.grid_remove()
        self.gain_section.grid_remove()
        self._on_gain_change()

    # exclusive view
    def _apply_window_mode(self, mode: str):
        self.an_frame.grid_remove(); self._an_visible = False
        self.log_frame.grid_remove(); self._logs_visible = False
        self.cfg_frame.grid_remove(); self._cfg_visible = False
        self.gain_section.grid_remove(); self._gain_visible = False

        if mode == "an":
            self.an_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(6,0)); self._an_visible = True
            try:
                if _AUDIO_OK and _PLOT_OK and not self.analyzer.running: self.analyzer.start()
            except Exception: pass
        elif mode == "logs":
            self.log_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(6,0)); self._logs_visible = True
        elif mode == "cfg":
            self._load_config_into_view()
            self.cfg_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(6,0)); self._cfg_visible = True
        elif mode == "io":
            self.gain_section.grid(row=0, column=0, sticky="ew", padx=0, pady=(6,0)); self._gain_visible = True

        self.update_idletasks()
        if mode == "none" and hasattr(self, "_base_w") and hasattr(self, "_base_h"):
            self.geometry(f"{self._base_w}x{self._base_h}")
        else:
            w = self.winfo_reqwidth()
            h = self.winfo_reqheight()
            self.geometry(f"{w}x{h}")

    # toggles
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

    def _toggle_config_exclusive(self):
        if self._cfg_visible:
            self._apply_window_mode("none")
        else:
            self._apply_window_mode("cfg")

    # meter loop
    def _update_bottom_meter(self):
        pre_db = None
        post_db = None

        if getattr(self, "level_tap", None) and self.level_tap.buffer.qsize() > 0:
            try:
                x = self.level_tap.buffer.queue[-1] if hasattr(self.level_tap.buffer, "queue") else self.level_tap.buffer.get_nowait()
                pre_db, post_db = self._measure_live_levels(x)
            except Exception:
                pass

        self.pre_meter.draw_meter(pre_db, pre_db)
        self.post_meter.draw_meter(post_db, post_db)
        self.output_meter.draw_meter(post_db, post_db)
        
        # Update clipping indicators
        if hasattr(self, "_clip_in_label"):
            self._clip_in_label.config(text="CLIP!" if pre_db is not None and pre_db >= -1.0 else "")
        if hasattr(self, "_clip_out_label"):
            self._clip_out_label.config(text="CLIP!" if post_db is not None and post_db >= -1.0 else "")
        
        self.after(33, self._update_bottom_meter)

    def _active_meter_paths(self):
        if self.proc_mode == "correction":
            return self.fir_left_var.get().strip(), self.fir_right_var.get().strip()
        return str(IMPULSE_L_PATH), str(IMPULSE_R_PATH)

    def _get_active_irs(self):
        left_path, right_path = self._active_meter_paths()
        sig = (self.proc_mode, left_path, right_path)
        if self._meter_ir_cache.get("sig") == sig:
            return self._meter_ir_cache["left"], self._meter_ir_cache["right"]

        ir_left = _read_wav_float(left_path) if left_path else None
        ir_right = _read_wav_float(right_path) if right_path else None
        if ir_left is None or len(ir_left) == 0:
            ir_left = np.array([1.0], dtype=np.float64)
        if ir_right is None or len(ir_right) == 0:
            ir_right = np.array([1.0], dtype=np.float64)

        self._meter_ir_cache = {
            "mode": self.proc_mode,
            "left": np.asarray(ir_left, dtype=np.float64),
            "right": np.asarray(ir_right, dtype=np.float64),
            "sig": sig,
        }
        self._meter_conv_state = {"left": None, "right": None, "sig": None}
        return self._meter_ir_cache["left"], self._meter_ir_cache["right"]

    def _process_signal_block(self, x, conv_state):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)

        gain_in = _db_to_linear(self._gain_in_db.get())
        gain_out = _db_to_linear(self._gain_out_db.get())

        pre = arr[:, :2] * gain_in

        ir_left, ir_right = self._get_active_irs()
        sig = (len(ir_left), len(ir_right))
        if conv_state.get("sig") != sig:
            conv_state["left"] = np.zeros(max(len(ir_left) - 1, 0), dtype=np.float64)
            conv_state["right"] = np.zeros(max(len(ir_right) - 1, 0), dtype=np.float64)
            conv_state["sig"] = sig

        left_in = pre[:, 0]
        right_in = pre[:, 1]
        if len(ir_left) > 1:
            left_out, conv_state["left"] = scipy.signal.lfilter(
                ir_left, [1.0], left_in, zi=conv_state["left"]
            )
        else:
            left_out = left_in * float(ir_left[0])
        if len(ir_right) > 1:
            right_out, conv_state["right"] = scipy.signal.lfilter(
                ir_right, [1.0], right_in, zi=conv_state["right"]
            )
        else:
            right_out = right_in * float(ir_right[0])

        post = np.column_stack((left_out, right_out)) * gain_out
        return pre, post

    def _measure_live_levels(self, x):
        pre, post = self._process_signal_block(x, self._meter_conv_state)
        pre_peak = float(np.max(np.abs(pre))) if pre.size else 0.0
        post_peak = float(np.max(np.abs(post))) if post.size else 0.0

        def _peak_to_db(peak):
            if peak <= 0.0:
                return None
            return max(-60.0, min(0.0, 20.0 * np.log10(max(peak, 1e-12))))

        return _peak_to_db(pre_peak), _peak_to_db(post_peak)

    def _analyzer_monitor_chunk(self, x):
        _pre, post = self._process_signal_block(x, self._analyzer_conv_state)
        return post

    # logs
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

    # start/stop btn
    def _sync_launch_btn(self):
        if self.proc_mode == 'correction':
            self.launch_btn.config(
                text="Active",
                style="Running.TButton"
            )
        elif self.proc_mode == 'bypass':
            self.launch_btn.config(
                text="Bypassed",
                style="Bypass.TButton"
            )
        else:
            self.launch_btn.config(
                text="Start",
                style="TButton"
            )

    # config open/write
    def open_config(self):
        self._toggle_config_exclusive()

    def _load_config_into_view(self):
        try:
            if not CONFIG_INTERNAL_PATH.exists():
                text = f"Bundled config does not exist:\n{CONFIG_INTERNAL_PATH}"
            else:
                with open(CONFIG_INTERNAL_PATH, "r", encoding="utf-8") as f:
                    text = f.read()
        except Exception as e:
            text = f"Could not load config:\n{e}"

        self.cfg_text.configure(state=tk.NORMAL)
        self.cfg_text.delete("1.0", tk.END)
        self.cfg_text.insert("1.0", text)
        self.cfg_text.configure(state=tk.DISABLED)

    def write_to_config(self, show_message=True):
        left  = (self.fir_left_var.get().strip()  if hasattr(self, "fir_left_var")  and self.fir_left_var.get()  else "")
        right = (self.fir_right_var.get().strip() if hasattr(self, "fir_right_var") and self.fir_right_var.get() else "")
        if not left or not right:
            if show_message:
                messagebox.showerror("Missing FIR", "Please select BOTH Left and Right FIR WAV files.")
            return False

        sr = int(self.sr_var.get() or 48000)
        cap  = self.cap_var.get()
        play = self.play_var.get()
        if not cap or not play:
            if show_message:
                messagebox.showerror("Missing Devices", "Please select capture and playback devices.")
            return False
        cap_esc  = cap.replace('"', '\\"')
        play_esc = play.replace("'", "''")
        left_esc  = left.replace('"', '\\"')
        right_esc = right.replace('"', '\\"')

        gi = 0.0; go = 0.0
        try: gi = float(self._gain_in_db.get())
        except Exception: pass
        try: go = float(self._gain_out_db.get())
        except Exception: pass

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
  - 1
  names:
  - gain_in
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
- type: Filter
  channels:
  - 0
  - 1
  names:
  - gain_out
filters:
  gain_in:
    type: Gain
    parameters:
      gain: {gi:.2f}
      scale: dB
      inverted: false
  gain_out:
    type: Gain
    parameters:
      gain: {go:.2f}
      scale: dB
      inverted: false
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
            if show_message:
                messagebox.showinfo("Saved", f"Wrote settings into:\n{CONFIG_INTERNAL_PATH}")
            return True
        except Exception as e:
            if show_message:
                messagebox.showerror("Write failed", str(e))
            else:
                self.append_log(f"Write failed: {e}\n")
            return False

    # devices / SR
    def refresh_devices(self, mode="all"):
        self.append_log(f"Refreshing {mode} devices...\n")
        cam_bin = which("camilladsp")
        if cam_bin is None:
            messagebox.showerror("Missing internal binary", "Bundled camilladsp not found."); return

        cap, play = [], []
        if mode in ["all", "input", "output"]:
            test_cfg = TEST_CONFIG_INTERNAL_PATH
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
                gain_in = filters.get("gain_in", {}).get("parameters", {}).get("gain")
                gain_out = filters.get("gain_out", {}).get("parameters", {}).get("gain")
                if gain_in is not None:
                    self._gain_in_db.set(float(gain_in))
                if gain_out is not None:
                    self._gain_out_db.set(float(gain_out))
                self._on_gain_change()
            except Exception:
                pass

    # SR helpers
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

    # FIR pickers
    def browse_fir_separate(self, channel):
        path = filedialog.askopenfilename(title=f"Select {channel} channel FIR WAV", filetypes=[("WAV files", "*.wav"), ("All files", "*")])
        if path:
            if channel == "left": self.fir_left_var.set(path)
            else: self.fir_right_var.set(path)

    # routing / camilla
    def switch_output(self):
        cap = self.cap_var.get()
        if not cap:
            messagebox.showerror("No capture device", "Select a capture device first"); return
        ok, msg = maybe_switch_output_to_blackhole(cap)
        if ok: messagebox.showinfo("Switched", msg)
        else:  messagebox.showwarning("Notice", msg)

    # bypass cfg
    def _ensure_bypass_config(self):
        try: sr = int(self.sr_var.get() or 48000)
        except Exception: sr = 48000
        _ensure_impulse_wav(IMPULSE_L_PATH, sr)
        _ensure_impulse_wav(IMPULSE_R_PATH, sr)
        cap  = self.cap_var.get()
        play = self.play_var.get()
        if not cap or not play:
            self.append_log("Bypass config skipped: no capture/playback device selected.\n")
            return
        try: gi = float(self._gain_in_db.get())
        except Exception: gi = 0.0
        try: go = float(self._gain_out_db.get())
        except Exception: go = 0.0
        cap_esc  = cap.replace('"', '\\"')
        play_esc = play.replace("'", "''")
        impulse_l = str(IMPULSE_L_PATH).replace('"', '\\"')
        impulse_r = str(IMPULSE_R_PATH).replace('"', '\\"')
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
  - 1
  names:
  - gain_in
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
- type: Filter
  channels:
  - 0
  - 1
  names:
  - gain_out
filters:
  gain_in:
    type: Gain
    parameters:
      gain: {gi:.2f}
      scale: dB
      inverted: false
  gain_out:
    type: Gain
    parameters:
      gain: {go:.2f}
      scale: dB
      inverted: false
  fir_L:
    type: Conv
    parameters:
      type: Wav
      filename: "{impulse_l}"
      channel: 0
  fir_R:
    type: Conv
    parameters:
      type: Wav
      filename: "{impulse_r}"
      channel: 0
"""
        )
        try:
            BYPASS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(BYPASS_CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(yaml_bypass)
        except Exception as e:
            self.append_log(f"Bypass write error: {e}\n")

    # FIR curve compute (EQ shown)
    def _compute_fir_curve_for_display(self):
        # sr
        try:
            sr = int(self.sr_var.get() or 48000)
        except Exception:
            sr = 48000
        # fft len → analyzer size
        nfft = getattr(self.analyzer, "_fft_size", 16384)

        # load L/R
        paths = [self.fir_left_var.get().strip(), self.fir_right_var.get().strip()]
        H_list = []
        for p in paths:
            if not p:
                continue
            x = _read_wav_float(p)
            if x is None or x.size == 0:
                self.append_log(f"[FIR] read error {p}\n")
                continue
            # light window to reduce ripple
            winN = min(len(x), 8192)
            win = np.hanning(winN)
            xw = x.copy()
            xw[:winN] *= win
            N = int(nfft)
            if xw.size < N:
                h = np.pad(xw, (0, N - xw.size))
            else:
                h = xw[:N]
            H = np.fft.rfft(h, n=N)
            H_list.append(H)

        if not H_list:
            return None, None

        # average transfer
        Havg = np.mean(np.vstack(H_list), axis=0)
        freqs = np.fft.rfftfreq(nfft, 1.0/sr)
        mag = np.abs(Havg)
        mag = np.maximum(mag, 1e-12)
        mag_db = 20.0 * np.log10(mag)

        # normalize at 1 kHz (0 dB ref)
        try:
            k = int(np.argmin(np.abs(freqs - 1000.0)))
            mag_db = mag_db - mag_db[k]
        except Exception:
            pass

        # light smooth (1/6)
        try:
            mag_db = self.analyzer._fractional_octave_power_smooth_fast(freqs, mag_db, fraction=6)
        except Exception:
            pass

        return freqs, mag_db

    def _auto_start_bypass(self):
        if self.proc is not None:
            return
        cam_bin = which("camilladsp")
        if cam_bin is None:
            return
        self._ensure_bypass_config()
        cmd = [cam_bin, str(BYPASS_CONFIG_PATH), "-v", "-p", "1234", "-w"]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.proc_mode = 'bypass'
            self._sync_launch_btn()
            self.append_log("Started CamillaDSP (bypass)\n")
            self.proc_thread = threading.Thread(target=self._reader_thread, daemon=True)
            self.proc_thread.start()
            self.after(100, self._poll_log_queue)
        except Exception as e:
            self.append_log(f"Bypass start error: {e}\n")
            self.proc = None
            self.proc_mode = None
            self._sync_launch_btn()
    
        

    # toggle correction
    def toggle_launch(self):
        if self.proc_mode != 'correction':
            if not self.write_to_config(show_message=False):
                messagebox.showerror("Missing FIR", "Please select BOTH Left and Right FIR WAV files.")
                return
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
                try:
                    freqs, mag_db = self._compute_fir_curve_for_display()
                    if freqs is not None and mag_db is not None and hasattr(self, "analyzer"):
                        self.analyzer.set_reference_curve(freqs, mag_db)
                    else:
                        self.append_log("[FIR] No valid FIR WAV(s) selected; overlay skipped.\n")
                except Exception as e:
                    self.append_log(f"[FIR] curve error: {e}\n")
            except Exception as e:
                messagebox.showerror("Start failed", str(e))
                self.proc = None
                self.proc_mode = None
                self._sync_launch_btn()
        else:
            self._stop_if_running()
            self.proc_mode = None
            self._sync_launch_btn()
            # clear FIR
            try: self.analyzer.clear_reference_curve()
            except Exception: pass
            self._auto_start_bypass()

    # stop camilla
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

    # reader
    def _reader_thread(self):
        p = self.proc
        if p is None or p.stdout is None: return
        for line in iter(p.stdout.readline, ''):
            if not line: break
            self.log_queue.put(line)
        rc = p.poll(); self.log_queue.put(f"[process exited with code {rc}]\n")

    # poll logs
    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.append_log(line)
        except queue.Empty:
            pass
        if self.proc is not None:
            self.after(100, self._poll_log_queue)

    # misc
    def _toggle_gain_section(self):
        if self._gain_visible:
            self._apply_window_mode("none")
        else:
            self._apply_window_mode("io")

    def _restart_camilla_for_current_mode(self):
        active_mode = self.proc_mode
        if active_mode not in ("correction", "bypass"):
            return
        self._stop_if_running()
        self.proc_mode = None
        self._sync_launch_btn()
        if active_mode == "correction":
            self.toggle_launch()
        else:
            self._auto_start_bypass()

    def _apply_gain_update(self):
        self._pending_gain_job = None
        config_ok = self.write_to_config(show_message=False)
        self._ensure_bypass_config()
        self._meter_ir_cache["sig"] = None
        self._meter_conv_state = {"left": None, "right": None, "sig": None}
        self._analyzer_conv_state = {"left": None, "right": None, "sig": None}
        if self.proc_mode == "correction":
            if config_ok:
                self._restart_camilla_for_current_mode()
        elif self.proc_mode == "bypass":
            self._restart_camilla_for_current_mode()

    def _on_gain_change(self, _=None):
        gi = self._gain_in_db.get()
        go = self._gain_out_db.get()
        self._gain_in_label.config(text=f"{gi:+.1f} dB")
        self._gain_out_label.config(text=f"{go:+.1f} dB")
        if self._pending_gain_job is not None:
            self.after_cancel(self._pending_gain_job)
        self._pending_gain_job = self.after(250, self._apply_gain_update)

    def on_sr_change(self):
        new_rate = self.sr_var.get()
        self.sr_status_var.set(f"Click Apply to set {new_rate}Hz")

    def on_close(self):
        try:
            if hasattr(self, "analyzer"): self.analyzer.stop()
            if hasattr(self, "level_tap"): self.level_tap.stop()
        except Exception: pass
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

# Main

def main():
    import traceback
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            app = FIRFilterGUI()
            app.mainloop()
            break
        except Exception as e:
            err_msg = f"[FIRC] Crash on startup (attempt {attempt}): {e}\n"
            tb = traceback.format_exc()
            with open("firc_crashlog.txt", "a", encoding="utf-8") as f:
                f.write(err_msg)
                f.write(tb)
            if attempt == max_attempts:
                import tkinter.messagebox as mb
                mb.showerror("FIRC Startup Error", f"FIRC failed to start after {max_attempts} attempts.\nSee firc_crashlog.txt for details.\nError: {e}")
                sys.exit(1)
            time.sleep(1)

if __name__ == "__main__":
    main()
  