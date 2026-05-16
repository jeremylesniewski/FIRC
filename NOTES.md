# FIRC — personal notes (v0.1.4)

Stuff I actually touch when setting up a machine or chasing weird audio behaviour. Not user docs — just my cheat sheet.

## Version

- Single source of truth: `firc/__init__.py` → `__version__`
- Window title and macOS bundle version follow that (build script reads it too)

## Before you build / ship

```bash
source .venv/bin/activate
pytest tests/
./build_mac_app.sh
open dist/FIRC.app
```

Optional sign (only when I care about Gatekeeper on other Macs):

```bash
export CODESIGN_IDENTITY="Developer ID Application: …"
./build_mac_app.sh
```

## Where configs really live

Bundled templates sit in `resources/` — **don’t edit those expecting the app to use them at runtime**.

Live configs the app reads/writes:

`~/Library/Application Support/FIRC/`

- `config.yml` — correction (your REW FIR paths end up here after **Write Config**)
- `config_bypass.yml` — passthrough / impulse
- `impulse_L.wav`, `impulse_R.wav` — generated on first run if missing

Override dir for testing: `FIRC_CONFIG_DIR=/tmp/firc-test`

## Things I tweak per setup

### Playback device name (the trailing-space trap)

CoreAudio often reports **`USB Audio CODEC `** with a **trailing space**. CamillaDSP is picky — `USB Audio CODEC` without the space fails with “Could not find playback device” even though `--check` can still say valid.

- **Refresh Devices** in the app (lists exact names from sounddevice)
- **Write Config** before **Start correction**
- If it still acts up: open `config.yml` in Application Support and match the name to what Camilla logs under “Available playback devices”

### Sample rate

REW exports: my HS5 set is **44.1 kHz** → set combobox to **44100** and hit **Apply** on both capture + playback before starting.

Mismatch = crackle, wrong filter length feel, or Camilla refusing to run happily.

### Routing (typical desk)

1. System output → **BlackHole 2ch** (or SwitchAudioSource / FIRC’s switch helper)
2. FIRC capture: **BlackHole 2ch**
3. FIRC playback: **USB DAC** (the `USB Audio CODEC ` one)
4. FIR files: whatever REW last exported into `~/Documents/REW/...`

### Default gains in bundled YAML

`resources/config*.yml` in the repo are neutral **0 dB** templates. My old personal gains (6 dB in / 1.5 dB out) were machine-specific — live gains are whatever I last wrote to Application Support.

### CamillaDSP version

Build script pins **3.0.0** into `vendor/bin/camilladsp`. Bump `CAMILLADSP_VERSION` in `build_mac_app.sh` if I upgrade.

## When the app “just quits” or audio is stuck

1. **Kill stray camilladsp** from old launches (they hold devices):

   ```bash
   pkill -f camilladsp
   ```

2. Launch from Terminal once to see stderr:

   ```bash
   dist/FIRC.app/Contents/MacOS/FIRC
   ```

3. Check `firc_crashlog.txt` in the cwd if Python died during startup

4. After a bad run: **Refresh Devices** → **Write Config** → try bypass first (starts automatically), then correction

## Code layout (post-refactor)

| Module | What I mess with |
|--------|------------------|
| `firc/app.py` | GUI, launch/stop, device refresh, dialogs |
| `firc/config.py` | YAML shape for Camilla |
| `firc/runtime.py` | App Support paths, bundle resource lookup |
| `firc/platform_utils.py` | `run()`, device canonical names, error text cleanup |
| `firc/analyzer.py` | Spectrum UI — FFT / smoothing defaults |
| `build_mac_app.sh` | Bundle contents, codesign, Camilla download |

`FIRC.py` is only the launcher — don’t grow it again.

## Tests I actually run

```bash
pytest tests/test_config_generation.py -q
python FIRC.py --smoke-test
```

## Git / release

Tag when happy: `v0.1.4` — see `CHANGELOG.md` for the GitHub release blurb.
