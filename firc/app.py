import queue
import subprocess
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import scipy.signal

from . import __version__
from .analyzer import Analyzer
from .audio_utils import _db_to_linear, _read_wav_float
from .config import _build_yaml_config, _read_yaml_config
from .deps import _AUDIO_OK, _PLOT_OK, np
from .level_tap import LevelTap
from .passthrough import PassthroughEngine
from .meters import Meter, VerticalMeter
from .platform_utils import (
    _canonical_device_name,
    _matching_device_name,
    _sd_list_devices,
    maybe_switch_output_to_blackhole,
    run,
    summarize_camilla_output,
    which,
)
from .runtime import (
    CONFIG_INTERNAL_PATH,
    IMPULSE_L_PATH,
    IMPULSE_R_PATH,
    ensure_runtime_files,
)


# GUI
class FIRFilterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"FIRC v{__version__}")
        self.resizable(True, True)

        self.proc = None
        self.proc_thread = None
        self.proc_mode = None
        self.proc_start_time = None
        self._proc_generation = 0
        self._polling_logs = False
        self._monitoring_process = False
        self._ready_for_gain_updates = False
        self._refresh_in_progress = False
        self._camilla_starting = False
        self._sr_apply_in_progress = False
        self._error_dialog_open = False
        self._pending_gain_job = None
        self.passthrough = None
        self.level_tap = None
        self._audio_watchdog_active = False
        self._last_audio_recover = 0.0
        self._recovering_audio = False
        self._audio_stream_lock = threading.Lock()
        self._meter_ir_cache = {"sig": None, "left": None, "right": None, "comp": 1.0}
        self._meter_conv_state    = {"left": None, "right": None, "sig": None}
        self._analyzer_conv_state = {"left": None, "right": None, "sig": None}
        self._compensation_enabled = tk.BooleanVar(value=True)

        self.log_queue = queue.Queue()
        self._log_buffer = []

        ensure_runtime_files()
        self._init_styles()
        self.create_widgets()

        # meter loop
        self.after(50, self._update_bottom_meter)
        self._sync_launch_btn()

        self.sr_status_var.set("Click Apply to set sample rate")
        self.sr_combo.bind('<<ComboboxSelected>>', lambda e: self.on_sr_change())

        # init async (device enumeration can block; keep it off the UI thread)
        self.after(150, lambda: self._refresh_devices_async("all", on_done=self._finish_startup))

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _show_dialog(self, kind, title, message, *, camilla_log=False):
        """Show a modal dialog attached to the main window (macOS-safe)."""
        if self._error_dialog_open and kind in ("error", "warning"):
            self.append_log(f"[{title}] {message}\n")
            return None
        text = summarize_camilla_output(message) if camilla_log else str(message)
        if len(text) > 1200:
            text = text[:1200] + "\n...(see Logs for full output)"
        self._error_dialog_open = kind in ("error", "warning")
        try:
            self.update_idletasks()
            self.lift()
            self.focus_force()
            opts = {"parent": self}
            if kind == "error":
                return messagebox.showerror(title, text, **opts)
            if kind == "warning":
                return messagebox.showwarning(title, text, **opts)
            if kind == "info":
                return messagebox.showinfo(title, text, **opts)
            if kind == "yesno":
                return messagebox.askyesno(title, text, **opts)
            return messagebox.showinfo(title, text, **opts)
        finally:
            self._error_dialog_open = False

    def _resolve_selected_devices(self):
        cap = _canonical_device_name(self.cap_var.get(), "input")
        play = _canonical_device_name(self.play_var.get(), "output")
        if cap and cap != self.cap_var.get():
            self.cap_var.set(cap)
        if play and play != self.play_var.get():
            self.play_var.set(play)
        return cap, play

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
        for i in range(6): action.grid_columnconfigure(i, weight=1)
        ttk.Button(action, text="Show Config",  command=self._toggle_config_exclusive).grid(row=0, column=0, padx=(0,3), sticky="ew")
        ttk.Button(action, text="Write Config", command=self.write_to_config).grid(row=0, column=1, padx=(0,3), sticky="ew")

        self.btn_toggle_vis  = ttk.Button(action, text="Visualizer", command=self._toggle_visualizer_exclusive)
        self.btn_toggle_vis.grid(row=0, column=2, padx=(0,3), sticky="ew")
        self.btn_toggle_logs = ttk.Button(action, text="Logs",       command=self._toggle_logs_exclusive)
        self.btn_toggle_logs.grid(row=0, column=3, padx=(0,3), sticky="ew")

        # IR compensation toggle
        ttk.Checkbutton(action, text="IR Compensation", variable=self._compensation_enabled,
                        command=self._on_compensation_toggle).grid(row=0, column=4, padx=(3,0), sticky="w")

        self.launch_btn = ttk.Button(action, text="Start correction", style="Accent.TButton", command=self.toggle_launch)
        self.launch_btn.grid(row=0, column=5, padx=(3,0), sticky="ew")

        # bypass label
        self.bypass_var = tk.StringVar(value="")
        ttk.Label(action, textvariable=self.bypass_var, font=("", 9, "italic")).grid(row=1, column=5, padx=(6,0), sticky="ne")

        # container
        self.an_container = ttk.Frame(frm)
        self.an_container.grid(row=5, column=0, sticky="nsew", pady=(10, 0))
        self.an_container.grid_columnconfigure(0, weight=1)

        # analyzer
        self.an_frame = ttk.LabelFrame(self.an_container)
        self.analyzer = Analyzer(
            self.an_frame,
            get_capture_name=lambda: self.cap_var.get(),
            get_playback_name=lambda: self.play_var.get(),
            get_samplerate=lambda: self.sr_var.get(),
            process_monitor_chunk=self._analyzer_monitor_chunk,
            get_audio_chunk=self._get_monitor_audio_chunk,
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
        self.post_meter = self.output_meter

        self._gain_in_db  = tk.DoubleVar(value=0.0)
        self._gain_out_db = tk.DoubleVar(value=-12.0)

        self._gain_out_label = None
        self._clip_out_label = None
        self._gain_out_slider = None

        # start compact
        self.an_frame.grid_remove()
        self.log_frame.grid_remove()
        self.cfg_frame.grid_remove()

    # exclusive view
    def _apply_window_mode(self, mode: str):
        self.an_frame.grid_remove(); self._an_visible = False
        self.log_frame.grid_remove(); self._logs_visible = False
        self.cfg_frame.grid_remove(); self._cfg_visible = False

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
        post_db = None

        try:
            x = self._get_monitor_audio_chunk()
            if x is not None:
                _, post_db = self._measure_live_levels(x)
        except Exception:
            pass

        self.post_meter.draw_meter(post_db, post_db)
        self.output_meter.draw_meter(post_db, post_db)

        if self._clip_out_label is not None:
            self._clip_out_label.config(text="CLIP!" if post_db is not None and post_db >= -1.0 else "")

        self.after(33, self._update_bottom_meter)

    def _get_monitor_audio_chunk(self):
        if self.proc_mode == "correction":
            if getattr(self, "level_tap", None):
                return self.level_tap.get_latest()
            return None
        if getattr(self, "passthrough", None):
            return self.passthrough.get_latest()
        return None

    def _active_meter_paths(self):
        if self.proc_mode == "correction":
            return self.fir_left_var.get().strip(), self.fir_right_var.get().strip()
        return "", ""

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

        ir_left  = np.asarray(ir_left,  dtype=np.float64)
        ir_right = np.asarray(ir_right, dtype=np.float64)

        gain_L = float(np.sqrt(np.sum(ir_left  ** 2))) or 1.0
        gain_R = float(np.sqrt(np.sum(ir_right ** 2))) or 1.0
        avg_gain = (gain_L + gain_R) / 2.0
        comp = max(0.1, min(10.0, 1.0 / avg_gain)) if avg_gain > 1e-6 else 1.0

        self._meter_ir_cache = {
            "sig":   sig,
            "left":  ir_left,
            "right": ir_right,
            "comp":  comp,
        }
        self._meter_conv_state    = {"left": None, "right": None, "sig": None}
        self._analyzer_conv_state = {"left": None, "right": None, "sig": None}
        return ir_left, ir_right

    def _process_signal_block(self, x, conv_state):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)

        pre = arr[:, :2]

        ir_left, ir_right = self._get_active_irs()
        sig = (len(ir_left), len(ir_right))
        if conv_state.get("sig") != sig:
            conv_state["left"]  = np.zeros(max(len(ir_left)  - 1, 0), dtype=np.float64)
            conv_state["right"] = np.zeros(max(len(ir_right) - 1, 0), dtype=np.float64)
            conv_state["sig"]   = sig

        left_in  = pre[:, 0]
        right_in = pre[:, 1]

        if len(ir_left) > 1:
            left_out,  conv_state["left"]  = scipy.signal.lfilter(ir_left,  [1.0], left_in,  zi=conv_state["left"])
        else:
            left_out  = left_in  * float(ir_left[0])

        if len(ir_right) > 1:
            right_out, conv_state["right"] = scipy.signal.lfilter(ir_right, [1.0], right_in, zi=conv_state["right"])
        else:
            right_out = right_in * float(ir_right[0])

        if self._compensation_enabled.get():
            comp = self._meter_ir_cache.get("comp", 1.0) or 1.0
            left_out  = left_out  * comp
            right_out = right_out * comp

        gain_out = _db_to_linear(self._gain_out_db.get())
        post = np.column_stack((left_out, right_out)) * gain_out
        return pre, post

    def _measure_live_levels(self, x):
        pre, post = self._process_signal_block(x, self._meter_conv_state)
        pre_peak  = float(np.max(np.abs(pre)))  if pre.size  else 0.0
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
        elif getattr(self, "passthrough", None) and self.passthrough.is_active():
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
                self._show_dialog("error", "Missing FIR", "Please select BOTH Left and Right FIR WAV files.")
            return False
        missing = [path for path in (left, right) if not Path(path).exists()]
        if missing:
            msg = "FIR WAV file not found:\n" + "\n".join(missing)
            if show_message:
                self._show_dialog("error", "Missing FIR file", msg)
            else:
                self.append_log(msg + "\n")
            return False

        sr = int(self.sr_var.get() or 48000)
        cap, play = self._resolve_selected_devices()
        if not cap or not play:
            if show_message:
                self._show_dialog("error", "Missing Devices", "Please select capture and playback devices.")
            return False
        go = 0.0
        try: go = float(self._gain_out_db.get())
        except Exception: pass

        try:
            yaml_text = _build_yaml_config(sr, cap, play, left, right, go)
            CONFIG_INTERNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_INTERNAL_PATH, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            if show_message:
                self._show_dialog("info", "Saved", f"Wrote settings into:\n{CONFIG_INTERNAL_PATH}")
            return True
        except Exception as e:
            if show_message:
                self._show_dialog("error", "Write failed", str(e))
            else:
                self.append_log(f"Write failed: {e}\n")
            return False

    def _finish_startup(self):
        try:
            self._init_audio_engines()
            self._ensure_passthrough_audio(log=True)
            self._schedule_audio_watchdog()
            self.append_log("Ready.\n")
            self._ready_for_gain_updates = True
            self.update_idletasks()
            self._base_w = self.winfo_width()
            self._base_h = self.winfo_height()
            self._apply_window_mode("none")
        except Exception as e:
            self.append_log(f"[Startup error] {e}\n")
            self._ready_for_gain_updates = True

    def _init_audio_engines(self):
        try:
            self.passthrough = PassthroughEngine(
                get_capture_name=lambda: self.cap_var.get(),
                get_playback_name=lambda: self.play_var.get(),
                get_samplerate=lambda: self.sr_var.get(),
                get_gain_in_db=lambda: self._gain_in_db.get(),
                get_gain_out_db=lambda: self._gain_out_db.get(),
            )
        except Exception as e:
            self.append_log(f"[PassthroughEngine] init failed: {e}\n")
            self.passthrough = None

        try:
            self.level_tap = LevelTap(
                get_capture_name=lambda: self.cap_var.get(),
                get_playback_name=lambda: self.play_var.get(),
                get_samplerate=lambda: self.sr_var.get(),
            )
        except Exception as e:
            self.append_log(f"[LevelTap] init failed: {e}\n")
            self.level_tap = None

    def _stop_passthrough(self):
        if getattr(self, "passthrough", None):
            self.passthrough.stop()

    def _stop_meter_tap(self):
        if getattr(self, "level_tap", None):
            self.level_tap.stop()

    def _start_meter_tap(self):
        if getattr(self, "level_tap", None) is None:
            return
        try:
            self.level_tap.stop()
            self.level_tap.start()
        except Exception as e:
            self.append_log(f"[Meter tap] restart failed: {e}\n")

    def _ensure_passthrough_audio(self, log=False):
        """Direct capture→playback (no CamillaDSP). Default whenever not in correction."""
        if self.proc_mode == "correction":
            return False

        with self._audio_stream_lock:
            self._stop_if_running()
            self._stop_meter_tap()
            cap, play = self._resolve_selected_devices()
            if not cap or not play:
                self.bypass_var.set("Select capture + playback")
                if log:
                    self.append_log("[Passthrough] waiting for devices\n")
                self._sync_launch_btn()
                return False
            try:
                if getattr(self, "passthrough", None) is None:
                    if log:
                        self.append_log("[Passthrough] engine not initialized\n")
                    return False

                ok = False
                if self.passthrough.is_active():
                    ok = True
                elif self.passthrough.running:
                    ok = self.passthrough.restart()
                else:
                    ok = self.passthrough.start()

                if ok:
                    self.proc_mode = None
                    self.bypass_var.set("Direct passthrough")
                    if log:
                        self.append_log("[Passthrough] capture → playback active\n")
                else:
                    self.bypass_var.set("Passthrough failed")
                    if log:
                        self.append_log("[Passthrough] could not open audio stream\n")
            except Exception as e:
                self.bypass_var.set(f"Passthrough error: {e}")
                if log:
                    self.append_log(f"[Passthrough] error: {e}\n")
                ok = False
            finally:
                self._sync_launch_btn()
            return ok

    def _schedule_audio_watchdog(self):
        if self._audio_watchdog_active:
            return
        self._audio_watchdog_active = True
        self.after(2000, self._audio_watchdog_tick)

    def _audio_watchdog_tick(self):
        self._audio_watchdog_active = False
        try:
            if self.winfo_exists():
                self._maintain_audio_path()
        except Exception as e:
            self.append_log(f"[Audio watchdog] error: {e}\n")
        finally:
            if self.winfo_exists():
                self._schedule_audio_watchdog()

    def _maintain_audio_path(self):
        now = time.monotonic()
        if self._recovering_audio or self._camilla_starting or (now - self._last_audio_recover) < 3.0:
            return

        try:
            if self.proc_mode == "correction":
                if self.proc is not None and self.proc.poll() is None:
                    return
                self._recovering_audio = True
                self.append_log("[Watchdog] correction offline — restarting CamillaDSP…\n")
                if self.write_to_config(show_message=False):
                    self._stop_passthrough()
                    self._start_camilla(CONFIG_INTERNAL_PATH, "correction", show_errors=False)
                else:
                    self.proc_mode = None
                    self._ensure_passthrough_audio()
                self._last_audio_recover = now
                self._recovering_audio = False
                return

            if self.proc is not None:
                self._stop_if_running()

            passthrough_ok = (
                getattr(self, "passthrough", None) is not None
                and self.passthrough.is_active()
                and self.passthrough.seconds_since_audio() <= 4.0
            )
            if not passthrough_ok:
                self._recovering_audio = True
                self.append_log("[Watchdog] passthrough offline — restarting…\n")
                self._ensure_passthrough_audio(log=False)
                self._last_audio_recover = now
                self._recovering_audio = False
        except Exception as e:
            self.append_log(f"[Watchdog error] {e}\n")
            self._recovering_audio = False

    def _restart_audio_after_device_change(self, force_correction=False):
        if force_correction or self.proc_mode == "correction":
            with self._audio_stream_lock:
                self._stop_if_running()
            if self.write_to_config(show_message=False):
                self._stop_passthrough()
                self._start_camilla(CONFIG_INTERNAL_PATH, "correction", show_errors=False)
        else:
            self._ensure_passthrough_audio()

    # devices / SR
    def refresh_devices(self, mode="all"):
        self._refresh_devices_async(mode)

    def _refresh_devices_async(self, mode="all", on_done=None):
        if self._refresh_in_progress:
            self.append_log("Device refresh already in progress...\n")
            return
        self._refresh_in_progress = True
        self.append_log(f"Refreshing {mode} devices...\n")
        threading.Thread(
            target=self._refresh_devices_worker,
            args=(mode, on_done),
            daemon=True,
            name="RefreshDevices",
        ).start()

    def _refresh_devices_worker(self, mode, on_done):
        missing_camilla = which("camilladsp") is None
        cap, play = [], []
        if mode in ["all", "input", "output"] and _AUDIO_OK:
            cap_list, play_list = _sd_list_devices()
            cap.extend(cap_list)
            play.extend(play_list)

        cfg, _ = _read_yaml_config(CONFIG_INTERNAL_PATH)

        out_sp, _, _ = run(["system_profiler", "SPAudioDataType"], timeout=20)
        if out_sp:
            current_device, has_input, has_output = None, False, False
            for line in out_sp.splitlines():
                line = line.strip()
                if not line or line in ("Audio:", "Devices:"):
                    continue
                if line.endswith(":"):
                    if current_device:
                        if has_input and mode in ["all", "input"] and current_device not in cap:
                            cap.append(current_device)
                        if has_output and mode in ["all", "output"] and current_device not in play:
                            play.append(current_device)
                    current_device, has_input, has_output = line[:-1].strip(), False, False
                    continue
                if "Input Channels:" in line:
                    has_input = True
                elif "Output Channels:" in line:
                    has_output = True
            if current_device:
                if has_input and mode in ["all", "input"] and current_device not in cap:
                    cap.append(current_device)
                if has_output and mode in ["all", "output"] and current_device not in play:
                    play.append(current_device)

        if (not cap or not play) and mode != "none":
            sa = which("SwitchAudioSource")
            if sa:
                if not cap and mode in ["all", "input"]:
                    out_in, _, _ = run([sa, "-a", "-t", "input"], timeout=10)
                    cap.extend(l.strip() for l in (out_in or "").splitlines() if l.strip())
                if not play and mode in ["all", "output"]:
                    out_out, _, _ = run([sa, "-a", "-t", "output"], timeout=10)
                    play.extend(l.strip() for l in (out_out or "").splitlines() if l.strip())

        cap = list(dict.fromkeys(cap))
        play = list(dict.fromkeys(play))
        self.after(0, lambda: self._apply_refresh_devices(mode, cap, play, cfg, missing_camilla, on_done))

    def _apply_refresh_devices(self, mode, cap, play, cfg, missing_camilla, on_done):
        self._refresh_in_progress = False
        if missing_camilla:
            self._show_dialog("error", "Missing internal binary", "Bundled camilladsp not found.")
            if on_done:
                on_done()
            return

        if mode in ["all", "input"]:
            if not cap and mode != "none":
                self._show_dialog("warning", "No input devices", "No input devices found.")
            else:
                self.cap_combo["values"] = cap
                current = self.cap_var.get()
                if current in cap:
                    self.cap_var.set(current)
                else:
                    try:
                        idx = cap.index("BlackHole 2ch")
                    except Exception:
                        idx = 0
                    if cap:
                        self.cap_var.set(cap[idx])

        if mode in ["all", "output"]:
            if not play and mode != "none":
                self._show_dialog("warning", "No output devices", "No output devices found.")
            else:
                self.play_combo["values"] = play
                current = self.play_var.get()
                if current in play:
                    self.play_var.set(current)
                else:
                    guess = None
                    for name in play:
                        if "USB" in name and ("DAC" in name or "CODEC" in name or "Codec" in name or "Audio" in name):
                            guess = name
                            break
                    try:
                        idx = play.index(guess) if guess in play else 0
                    except Exception:
                        idx = 0
                    if play:
                        self.play_var.set(play[idx])

        if cfg:
            try:
                devs = cfg.get("devices", {})
                sr = devs.get("samplerate")
                if isinstance(sr, int) and str(sr) in ("44100", "48000"):
                    self.sr_var.set(str(sr))
                    self.sr_status_var.set(f"(loaded {sr}Hz from config)")
                cap_name = devs.get("capture", {}).get("device")
                play_name = devs.get("playback", {}).get("device")
                if cap_name:
                    match = _matching_device_name(cap_name, cap)
                    if not cap:
                        self.cap_var.set(str(cap_name).strip())
                    elif match:
                        self.cap_var.set(match)
                    else:
                        self.append_log(f"[Config] Capture device not available: {cap_name}\n")
                if play_name:
                    match = _matching_device_name(play_name, play)
                    if not play:
                        self.play_var.set(str(play_name).strip())
                    elif match:
                        self.play_var.set(match)
                    else:
                        self.append_log(f"[Config] Playback device not available: {play_name}\n")
                filters = cfg.get("filters", {})
                fir_L = filters.get("fir_L", {}).get("parameters", {}).get("filename")
                fir_R = filters.get("fir_R", {}).get("parameters", {}).get("filename")
                if fir_L:
                    self.fir_left_var.set(str(fir_L))
                if fir_R:
                    self.fir_right_var.set(str(fir_R))
                # Note: gain_in is no longer used (always 0dB). Only restore gain_out.
                gain_out = filters.get("gain_out", {}).get("parameters", {}).get("gain")
                if gain_out is not None:
                    self._gain_out_db.set(float(gain_out))
                self._on_gain_change()
            except Exception:
                pass

        self._resolve_selected_devices()
        self._restart_audio_after_device_change()
        if on_done:
            on_done()

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
        try:
            new_rate = int(self.sr_var.get())
        except ValueError:
            self._show_dialog("error", "Invalid Rate", "Please select a valid sample rate")
            return
        if self._sr_apply_in_progress:
            return
        was_correction = self.proc_mode == "correction"
        if self.proc is not None:
            self._stop_if_running()
        self._stop_passthrough()

        self._sr_apply_in_progress = True
        self.sr_status_var.set("Applying...")
        cap = self.cap_var.get()
        play = self.play_var.get()
        threading.Thread(
            target=self._apply_sample_rate_worker,
            args=(new_rate, cap, play, was_correction),
            daemon=True,
            name="ApplySampleRate",
        ).start()

    def _apply_sample_rate_worker(self, new_rate, cap_name, play_name, was_correction):
        errs = []
        for dev_name, label in ((cap_name, "Input"), (play_name, "Output")):
            if dev_name:
                ok, msg = self.set_device_sample_rate(dev_name, new_rate)
                if not ok:
                    errs.append(f"{label}: {msg}")
        self.after(0, lambda: self._finish_apply_sample_rate(new_rate, errs, was_correction))

    def _finish_apply_sample_rate(self, new_rate, errs, was_correction):
        self._sr_apply_in_progress = False
        if errs:
            self.sr_status_var.set("Some failed!")
            self._show_dialog("error", "Sample Rate", "\n".join(errs))
        else:
            self.sr_status_var.set(f"{new_rate}Hz set")
        if was_correction:
            self._restart_audio_after_device_change(force_correction=True)
        else:
            self._ensure_passthrough_audio()
        if _AUDIO_OK and _PLOT_OK and self._an_visible:
            self.analyzer.stop()
            self.analyzer.start()

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
            self._show_dialog("error", "No capture device", "Select a capture device first")
            return
        ok, msg = maybe_switch_output_to_blackhole(cap)
        if ok:
            self._show_dialog("info", "Switched", msg)
        else:
            self._show_dialog("warning", "Notice", msg)

    # FIR curve compute (EQ shown)
    def _compute_fir_curve_for_display(self):
        try:
            sr = int(self.sr_var.get() or 48000)
        except Exception:
            sr = 48000
        nfft = getattr(self.analyzer, "_fft_size", 16384)

        paths = [self.fir_left_var.get().strip(), self.fir_right_var.get().strip()]
        H_list = []
        for p in paths:
            if not p:
                continue
            x = _read_wav_float(p)
            if x is None or x.size == 0:
                self.append_log(f"[FIR] read error {p}\n")
                continue
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

        Havg = np.mean(np.vstack(H_list), axis=0)
        freqs = np.fft.rfftfreq(nfft, 1.0/sr)
        mag = np.abs(Havg)
        mag = np.maximum(mag, 1e-12)
        mag_db = 20.0 * np.log10(mag)

        try:
            k = int(np.argmin(np.abs(freqs - 1000.0)))
            mag_db = mag_db - mag_db[k]
        except Exception:
            pass

        try:
            mag_db = self.analyzer._fractional_octave_power_smooth_fast(freqs, mag_db, fraction=6)
        except Exception:
            pass

        return freqs, mag_db

    def _start_camilla(self, config_path, mode="correction", show_errors=True):
        if self._camilla_starting:
            self.append_log(f"CamillaDSP start already in progress ({mode}).\n")
            return False
        # Don't restart if already running fine
        if self.proc_mode == "correction" and self.proc is not None and self.proc.poll() is None:
            return False
        cam_bin = which("camilladsp")
        if cam_bin is None:
            msg = "Bundled camilladsp not found."
            if show_errors:
                self._show_dialog("error", "Missing internal binary", msg)
            else:
                self.append_log(f"{mode.capitalize()} not started: {msg}\n")
            return False
        if not Path(config_path).exists():
            msg = f"Config not found: {config_path}"
            if show_errors:
                self._show_dialog("error", "Config missing", msg)
            else:
                self.append_log(f"{mode.capitalize()} not started: {msg}\n")
            return False

        if mode != "correction":
            return False

        self._resolve_selected_devices()
        if not self.write_to_config(show_message=False):
            if show_errors:
                self._show_dialog("error", "Missing FIR", "Please select BOTH Left and Right FIR WAV files.")
            return False

        self._stop_passthrough()
        self._stop_meter_tap()
        self._camilla_starting = True
        threading.Thread(
            target=self._start_camilla_worker,
            args=(cam_bin, config_path, mode, show_errors),
            daemon=True,
            name=f"StartCamilla-{mode}",
        ).start()
        return True

    def _start_camilla_worker(self, cam_bin, config_path, mode, show_errors):
        check = [cam_bin, str(config_path), "--check"]
        out, err, rc = run(check, timeout=20)
        self.after(0, lambda: self._finish_start_camilla(cam_bin, config_path, mode, show_errors, rc, out, err))

    def _finish_start_camilla(self, cam_bin, config_path, mode, show_errors, rc, out, err):
        if rc != 0:
            self._camilla_starting = False
            msg = (out or err or "Unknown config validation error").strip()
            self.append_log(f"[CamillaDSP config check failed]\n{msg}\n")
            if show_errors:
                self._show_dialog("error", "Config check failed", msg, camilla_log=True)
            return False

        with self._audio_stream_lock:
            self._stop_if_running()
            cmd = [cam_bin, str(config_path), "-v"]
            self._proc_generation += 1
            generation = self._proc_generation
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            except Exception as e:
                self._camilla_starting = False
                if show_errors:
                    self._show_dialog("error", "Start failed", str(e))
                else:
                    self.append_log(f"{mode.capitalize()} start error: {e}\n")
                self.proc = None
                self.proc_mode = None
                self._sync_launch_btn()
                return False

        def verify_start():
            with self._audio_stream_lock:
                if generation != self._proc_generation:
                    return

                if proc.poll() is not None:
                    try:
                        output, _ = proc.communicate(timeout=1)
                    except Exception:
                        output = ""
                    msg = output.strip() or f"CamillaDSP exited immediately with code {proc.returncode}."
                    self.append_log(f"[CamillaDSP failed to start: {mode}]\n{msg}\n")
                    if show_errors:
                        self._show_dialog("error", "Start failed", msg, camilla_log=True)
                    self.proc = None
                    self.proc_mode = None
                    self._sync_launch_btn()
                    self._ensure_passthrough_audio()
                    self._camilla_starting = False
                    return

                self.proc = proc
                self.proc_mode = "correction"
                self.proc_start_time = time.time()
                self._sync_launch_btn()
                self.bypass_var.set("CamillaDSP correction")
                self.append_log("Started CamillaDSP (correction)\n")
                self._start_meter_tap()
                self.proc_thread = threading.Thread(
                    target=self._reader_thread,
                    args=(proc, generation),
                    daemon=True,
                    name=f"CamillaReader-{mode}",
                )
                self.proc_thread.start()
                self._schedule_log_poll()
                self._schedule_process_monitor()
                self._camilla_starting = False

        self.after(350, verify_start)
        return True

    # toggle correction
    def toggle_launch(self):
        if self.proc_mode != 'correction':
            if self._camilla_starting:
                return  # already in progress, ignore click
            if not self.write_to_config(show_message=False):
                self._show_dialog("error", "Missing FIR", "Please select BOTH Left and Right FIR WAV files.")
                return
            self._stop_passthrough()
            if self._start_camilla(CONFIG_INTERNAL_PATH, "correction", show_errors=True):
                try:
                    freqs, mag_db = self._compute_fir_curve_for_display()
                    if freqs is not None and mag_db is not None and hasattr(self, "analyzer"):
                        self.analyzer.set_reference_curve(freqs, mag_db)
                    else:
                        self.append_log("[FIR] No valid FIR WAV(s) selected; overlay skipped.\n")
                except Exception as e:
                    self.append_log(f"[FIR] curve error: {e}\n")
        else:
            if self._camilla_starting:
                return
            with self._audio_stream_lock:
                self._stop_if_running()
                self.proc_mode = None
                self._sync_launch_btn()
            try:
                self.analyzer.clear_reference_curve()
            except Exception:
                pass
            self._stop_meter_tap()
            self._ensure_passthrough_audio()

    # stop camilla
    def _stop_if_running(self):
        if self.proc is None: return
        proc = self.proc
        self._proc_generation += 1
        try:
            self.append_log("Stopping CamillaDSP...\n")
            proc.terminate()
            try:
                proc.wait(timeout=3)
                self.append_log("[CamillaDSP terminated gracefully]\n")
            except subprocess.TimeoutExpired:
                self.append_log("[CamillaDSP did not respond to SIGTERM; forcing kill]\n")
                proc.kill()
                proc.wait(timeout=1)
        except Exception as e:
            self.append_log(f"Stop error: {e}\n")
        finally:
            if self.proc is proc:
                self.proc = None
            self.proc_start_time = None

    # process health monitoring
    def _schedule_process_monitor(self):
        if self._monitoring_process:
            return
        self._monitoring_process = True
        self.after(1000, self._monitor_process_health)

    def _monitor_process_health(self):
        """Detect if CamillaDSP process is hung or dead."""
        self._monitoring_process = False
        if self.proc is None or self.proc_mode is None:
            return

        rc = self.proc.poll()
        if rc is not None:
            with self._audio_stream_lock:
                self.append_log(f"[CamillaDSP exited with code {rc}] — recovering passthrough…\n")
                self.proc = None
                self.proc_mode = None
                self._sync_launch_btn()
                self._stop_meter_tap()
            self._ensure_passthrough_audio()
            return

        if self.proc_mode == "correction":
            self._schedule_process_monitor()

    # reader
    def _reader_thread(self, proc, generation):
        """Read output from CamillaDSP subprocess."""
        if proc is None or proc.stdout is None:
            return

        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                self.log_queue.put((generation, line))
            rc = proc.poll()
            self.log_queue.put((generation, f"[process stream closed, exit code: {rc}]\n"))
        except Exception as e:
            self.log_queue.put((generation, f"[reader thread error: {e}]\n"))

    # poll logs
    def _schedule_log_poll(self):
        if self._polling_logs:
            return
        self._polling_logs = True
        self.after(100, self._poll_log_queue)

    def _poll_log_queue(self):
        self._polling_logs = False
        try:
            while True:
                generation, line = self.log_queue.get_nowait()
                if generation == self._proc_generation:
                    self.append_log(line)
        except queue.Empty:
            pass
        if self.proc is not None or not self.log_queue.empty():
            self._schedule_log_poll()

    # misc
    def _apply_gain_update(self):
        self._pending_gain_job = None
        try:
            config_ok = self.write_to_config(show_message=False)
            self._meter_ir_cache["sig"] = None
            self._meter_conv_state = {"left": None, "right": None, "sig": None}
            self._analyzer_conv_state = {"left": None, "right": None, "sig": None}
            if self.proc_mode == "correction":
                if config_ok and not self._camilla_starting:
                    self._stop_if_running()
                    self._stop_passthrough()
                    self._start_camilla(CONFIG_INTERNAL_PATH, "correction", show_errors=False)
            else:
                self._ensure_passthrough_audio()
        except Exception as e:
            self.append_log(f"[Gain update error] {e}\n")

    def _on_gain_change(self, _=None):
        try:
            go = self._gain_out_db.get()
            # FIX: use None check instead of hasattr, since the stub sets these to None
            if self._gain_out_label is not None:
                self._gain_out_label.config(text=f"{go:.1f} dB")
            if getattr(self, "passthrough", None):
                self.passthrough.update_gains(0.0, go)  # Input always 0dB, only output gain
        except Exception:
            pass
        if not self._ready_for_gain_updates:
            return
        if self._pending_gain_job is not None:
            self.after_cancel(self._pending_gain_job)
        self._pending_gain_job = self.after(500, self._apply_gain_update)

    def _on_compensation_toggle(self):
        """Handle convolution compensation toggle change."""
        try:
            if self._compensation_enabled.get():
                self.append_log("[Compensation] Auto-compensation enabled\n")
            else:
                self.append_log("[Compensation] Auto-compensation disabled\n")
        except Exception as e:
            self.append_log(f"[Compensation error] {e}\n")

    def on_sr_change(self):
        new_rate = self.sr_var.get()
        self.sr_status_var.set(f"Click Apply to set {new_rate}Hz")

    def on_close(self):
        try:
            if hasattr(self, "analyzer") and self.analyzer:
                self.analyzer.stop()
            self._stop_meter_tap()
            self._stop_passthrough()
        except Exception as e:
            self.append_log(f"[Cleanup] {e}\n")

        if self.proc is not None:
            if self._show_dialog("yesno", "Quit", "CamillaDSP is running. Stop and quit?"):
                self._stop_if_running()
            else:
                return

        try:
            self.destroy()
        except Exception:
            pass