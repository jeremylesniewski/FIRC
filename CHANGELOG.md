# Changelog

## [0.1.4] — 2026-05-16

Maintenance release: same app, better structure and fewer macOS papercuts. Not a feature milestone — just stuff that needed fixing before I touch a C++ rewrite.

### Added

- **`firc/` package** — split out of the monolithic `FIRC.py` (launcher stays thin)
- **Per-user config** in `~/Library/Application Support/FIRC/`
- **`python FIRC.py --smoke-test`**
- **Config generation tests** (`tests/`)
- **Build** bundles CamillaDSP 3.0.0, bypass YAML, impulses; optional `CODESIGN_IDENTITY`
- **NOTES.md** — personal setup cheat sheet

### Changed

- Exact CoreAudio device names (trailing-space playback devices)
- Device refresh / sample rate / Camilla start off the UI thread
- Shared capture tap for meters + visualizer
- Clearer error dialogs on macOS

### Fixed

- UI freezes on Refresh / Apply / Start
- CamillaDSP “playback device not found”
- Stuck error popups; bypass fallback when correction dies

---

## [0.1.3] — 2026-04-29

- README image update

## [0.1.2] — earlier

- Initial public macOS builds
