import queue
import threading

from .deps import _AUDIO_OK, np, sd
from .platform_utils import _sd_find_device_index


class LevelTap:
    def __init__(self, get_capture_name, get_playback_name, get_samplerate):
        self.get_capture_name = get_capture_name
        self.get_playback_name = get_playback_name
        self.get_samplerate = get_samplerate
        self.stream = None
        self.buffer = queue.Queue(maxsize=4)
        self.running = False
        self.lock = threading.Lock()
        self._latest = None

    def get_latest(self):
        with self.lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def start(self):
        if not _AUDIO_OK or self.running:
            return
        self.running = True
        self._open_stream()

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
        if not self.running:
            return
        with self.lock:
            try:
                if self.stream:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass
            self.stream = None
        self._open_stream()

    def _open_stream(self):
        try:
            sr = int(self.get_samplerate() or 48000)
        except Exception:
            sr = 48000
        try:
            cap_name = (self.get_capture_name() or "")
        except Exception:
            cap_name = ""

        dev_idx = _sd_find_device_index(cap_name, "input")

        def cb(indata, frames, time_info, status):
            x = indata
            if x.ndim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[:, : min(2, x.shape[1])]
            block = x.astype(np.float64, copy=False)
            with self.lock:
                self._latest = block
            try:
                self.buffer.put_nowait(block)
            except queue.Full:
                try:
                    _ = self.buffer.get_nowait()
                except Exception:
                    pass
                try:
                    self.buffer.put_nowait(block)
                except Exception:
                    pass

        try:
            if dev_idx is None:
                return
            try:
                ch = max(1, int(sd.query_devices(dev_idx).get("max_input_channels", 1)))
            except Exception:
                ch = 1
            ch = 2 if ch >= 2 else 1
            self.stream = sd.InputStream(
                device=dev_idx,
                channels=ch,
                samplerate=sr,
                blocksize=256,
                dtype="float32",
                callback=cb,
            )
            self.stream.start()
        except Exception:
            self.stream = None
