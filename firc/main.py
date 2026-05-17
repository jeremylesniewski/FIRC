import sys
import traceback

from .app import FIRFilterGUI
from .runtime import (
    BYPASS_CONFIG_PATH,
    CONFIG_INTERNAL_PATH,
    IMPULSE_L_PATH,
    IMPULSE_R_PATH,
    ensure_runtime_files,
)


def main():
    if "--smoke-test" in sys.argv:
        ensure_runtime_files()
        print(f"config={CONFIG_INTERNAL_PATH}")
        print(f"bypass={BYPASS_CONFIG_PATH}")
        print(f"impulse_l={IMPULSE_L_PATH}")
        print(f"impulse_r={IMPULSE_R_PATH}")
        return
    try:
        app = FIRFilterGUI()
        app.mainloop()
    except Exception as e:
        err_msg = f"[FIRC] Startup error: {e}\n"
        tb = traceback.format_exc()
        with open("firc_crashlog.txt", "a", encoding="utf-8") as f:
            f.write(err_msg)
            f.write(tb)
        import tkinter.messagebox as mb
        mb.showerror("FIRC Startup Error", f"FIRC failed to start.\nSee firc_crashlog.txt for details.\nError: {e}")
        sys.exit(1)
