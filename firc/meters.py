import time
import tkinter as tk
from tkinter import ttk


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
