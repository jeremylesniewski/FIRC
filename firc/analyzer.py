import math
import queue
import time
import tkinter as tk
from tkinter import ttk

import scipy.signal

from .deps import _AUDIO_OK, _PLOT_OK, Figure, FigureCanvasTkAgg, PolyCollection, np, sd
from .platform_utils import _sd_find_device_index


class Analyzer(ttk.LabelFrame):
    def __init__(
        self,
        master,
        get_capture_name,
        get_playback_name,
        get_samplerate,
        process_monitor_chunk=None,
        get_audio_chunk=None,
    ):
        super().__init__(master)

        # getters
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate = get_samplerate
        self.process_monitor_chunk = process_monitor_chunk
        self.get_audio_chunk = get_audio_chunk

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
            self.fig = Figure(figsize=(9.2, 3.2), dpi=100, facecolor="#1a1a1a")
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

    def _ingest_audio_block(self, x):
        if x is None:
            return
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        else:
            x = x[:, : min(2, x.shape[1])]
        if self.process_monitor_chunk is not None:
            try:
                x = self.process_monitor_chunk(x)
            except Exception:
                pass
        if x.ndim > 1:
            x = np.mean(x[:, : min(2, x.shape[1])], axis=1)
        self._last_data_t = time.monotonic()
        try:
            self.buffer.put_nowait(x.astype(np.float64, copy=False))
        except queue.Full:
            pass

    def _pull_shared_audio(self):
        if self.get_audio_chunk is None:
            return
        try:
            self._ingest_audio_block(self.get_audio_chunk())
        except Exception:
            pass

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

        if self.get_audio_chunk is not None:
            self._monitor_label_text.set(f"Metering from: {cap_name or 'shared tap'}")
            self._update_overlay()
            self.running = True
            self._schedule_update(self._fft_size, sr, period_ms=16)
            return

        dev_idx = _sd_find_device_index(cap_name, "input")
        if dev_idx is None:
            self._monitor_label_text.set("Camilla output analysis: (no valid capture input)")
            if _PLOT_OK:
                self.overlay.config(text="No valid capture input selected")
                self.overlay.place(relx=0.5, rely=0.5, anchor="center")
            return

        self._monitor_label_text.set(f"Metering from: {cap_name}")
        self._update_overlay()

        def cb(indata, frames, time_info, status):
            self._ingest_audio_block(indata)

        try:
            try:
                max_ch = max(1, int(sd.query_devices(dev_idx).get("max_input_channels", 1)))
            except Exception:
                max_ch = 1
            stream_channels = 2 if max_ch >= 2 else 1
            self.stream = sd.InputStream(
                device=dev_idx,
                channels=stream_channels,
                samplerate=sr,
                blocksize=256,
                dtype="float32",
                callback=cb,
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
        if self.get_audio_chunk is not None:
            self._last_data_t = time.monotonic()
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
        if not self.running:
            return
        if self.get_audio_chunk is not None:
            self._pull_shared_audio()
        elif self._last_data_t is not None and (time.monotonic() - self._last_data_t) > 1.5:
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
        active = self.running and (self.stream is not None or self.get_audio_chunk is not None)
        
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

