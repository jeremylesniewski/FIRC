import os
from pathlib import Path


def ensure_mplconfigdir():
    try:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        mpldir = Path(os.environ.get("MPLCONFIGDIR", base / "matplotlib"))
        mpldir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpldir)
    except Exception:
        pass


ensure_mplconfigdir()

_AUDIO_OK = True
_PLOT_OK = True

try:
    import numpy as np
    import sounddevice as sd
except Exception:
    _AUDIO_OK = False
    np = None
    sd = None

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.collections import PolyCollection
except Exception:
    _PLOT_OK = False
    Figure = None
    FigureCanvasTkAgg = None
    PolyCollection = None
