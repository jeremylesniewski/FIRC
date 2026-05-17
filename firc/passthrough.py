import threading
import time

from .audio_utils import _db_to_linear
from .deps import _AUDIO_OK, np, sd
from .platform_utils import _sd_find_device_index


class PassthroughEngine:
    """Capture → gain → playback in Python (bypass, no CamillaDSP)."""

    def __init__(self, get_capture_name, get_playback_name, get_samplerate, get_gain_in_db, get_gain_out_db):
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate = get_samplerate
        self.get_gain_in_db = get_gain_in_db
        self.get_gain_out_db = get_gain_out_db
        self.stream = None
        self.running = False
        self.lock = threading.Lock()
        self._latest = None
        self._last_ok = 0.0

    def get_latest(self):
        with self.lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def is_active(self):
        try:
            return self.running and self.stream is not None and self.stream.active
        except Exception:
            return False

    def seconds_since_audio(self):
        if self._last_ok <= 0:
            return float("inf")
        return time.monotonic() - self._last_ok

    def start(self):
        if not _AUDIO_OK or self.running:
            return False
        self.running = True
        return self._open_stream()

    def stop(self):
        self.running = False
        with self.lock:
            self._latest = None
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass
            self.stream = None

    def restart(self):
        self.stop()
        self.running = True
        return self._open_stream()

    def _open_stream(self):
        try:
            sr = int(self.get_samplerate() or 48000)
        except (ValueError, TypeError):
            sr = 48000

        cap_name = ""
        play_name = ""
        try:
            cap_name = (self.get_capture_name() or "").strip()
            play_name = (self.get_playback_name() or "").strip()
        except Exception:
            pass

        cap_idx = _sd_find_device_index(cap_name, "input")
        play_idx = _sd_find_device_index(play_name, "output")
        if cap_idx is None or play_idx is None:
            self.stream = None
            return False

        channels = 2

        def callback(indata, outdata, frames, time_info, status):
            try:
                x = np.asarray(indata, dtype=np.float64)
                if x.ndim == 1:
                    x = x[:, np.newaxis]
                if x.shape[1] == 1:
                    x = np.repeat(x, 2, axis=1)
                x = x[:, :2]

                gain_in = _db_to_linear(self.get_gain_in_db())
                gain_out = _db_to_linear(self.get_gain_out_db())
                pre = x * gain_in
                post = pre * gain_out

                with self.lock:
                    self._latest = pre.copy()
                    self._last_ok = time.monotonic()

                n = min(frames, post.shape[0])
                out = post[:n].astype(np.float32, copy=False)
                if outdata.shape[1] == 1:
                    outdata[:n, 0] = out.mean(axis=1)
                else:
                    outdata[:n, : out.shape[1]] = out[:, : outdata.shape[1]]
                if n < frames:
                    outdata[n:] = 0
            except Exception:
                outdata.fill(0)

        try:
            self.stream = sd.Stream(
                device=(cap_idx, play_idx),
                channels=channels,
                samplerate=sr,
                blocksize=256,
                dtype="float32",
                callback=callback,
            )
            self.stream.start()
            self._last_ok = time.monotonic()
            return True
        except Exception as e:
            self.stream = None
            return False
