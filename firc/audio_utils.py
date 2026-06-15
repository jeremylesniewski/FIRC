import struct
import wave
from pathlib import Path

import numpy as np


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


def _linear_to_dbfs(value):
    try:
        value = float(value)
    except Exception:
        return None
    if value <= 0.0:
        return None
    return max(-120.0, min(0.0, 20.0 * np.log10(max(value, 1e-12))))


def _peak_dbfs(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return None
    return _linear_to_dbfs(float(np.max(np.abs(arr))))


def _rms_dbfs(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        rms = float(np.sqrt(np.mean(arr * arr)))
    else:
        rms = float(np.max(np.sqrt(np.mean(arr * arr, axis=0))))
    return _linear_to_dbfs(rms)


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
