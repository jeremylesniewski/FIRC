from pathlib import Path

try:
    import yaml
    _YAML_OK = True
except Exception:
    _YAML_OK = False
    yaml = None


# YAML helpers
def _build_yaml_config(
    sr: int,
    cap_device: str,
    play_device: str,
    left_fir: str,
    right_fir: str,
    gain_in_db: float = 0.0,
    gain_out_db: float = -6.0,
) -> str:
    if not _YAML_OK:
        raise RuntimeError("PyYAML not available; cannot generate config.")

    config = {
        'devices': {
            'samplerate': sr,
            'chunksize': 1024,
            'capture': {
                'type': 'CoreAudio',
                'device': cap_device,
                'channels': 2
            },
            'playback': {
                'type': 'CoreAudio',
                'device': play_device,
                'channels': 2
            }
        },
        'pipeline': [
            {'type': 'Filter', 'channels': [0, 1], 'names': ['gain_in']},
            {'type': 'Filter', 'channels': [0], 'names': ['fir_L']},
            {'type': 'Filter', 'channels': [1], 'names': ['fir_R']},
            {'type': 'Filter', 'channels': [0, 1], 'names': ['gain_out']}
        ],
        'filters': {
            'gain_in': {
                'type': 'Gain',
                'parameters': {
                    'gain': round(gain_in_db, 2),
                    'scale': 'dB',
                    'inverted': False
                }
            },
            'gain_out': {
                'type': 'Gain',
                'parameters': {
                    'gain': round(gain_out_db, 2),
                    'scale': 'dB',
                    'inverted': False
                }
            },
            'fir_L': {
                'type': 'Conv',
                'parameters': {
                    'type': 'Wav',
                    'filename': left_fir,
                    'channel': 0
                }
            },
            'fir_R': {
                'type': 'Conv',
                'parameters': {
                    'type': 'Wav',
                    'filename': right_fir,
                    'channel': 0
                }
            }
        }
    }
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def _read_yaml_config(path: Path):
    if not _YAML_OK or not path.exists():
        return None, "PyYAML missing or config not found."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}, None
    except Exception as e:
        return None, f"YAML read error: {e}"
