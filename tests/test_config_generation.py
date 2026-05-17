from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FIRC import _build_yaml_config


def test_build_yaml_config_escapes_device_names_and_paths():
    text = _build_yaml_config(
        48000,
        'BlackHole "2ch"',
        "USB Audio CODEC 'Main'",
        '/tmp/left "quoted".wav',
        "/tmp/right's.wav",
        1.234,
        -5.678,
    )

    config = yaml.safe_load(text)

    assert config["devices"]["samplerate"] == 48000
    assert config["devices"]["capture"]["device"] == 'BlackHole "2ch"'
    assert config["devices"]["playback"]["device"] == "USB Audio CODEC 'Main'"
    assert config["filters"]["fir_L"]["parameters"]["filename"] == '/tmp/left "quoted".wav'
    assert config["filters"]["fir_R"]["parameters"]["filename"] == "/tmp/right's.wav"
    assert config["filters"]["gain_in"]["parameters"]["gain"] == 1.23
    assert config["filters"]["gain_out"]["parameters"]["gain"] == -5.68


def test_default_resource_configs_use_relative_impulses():
    root = Path(__file__).resolve().parents[1]

    for name in ("config.yml", "config_bypass.yml"):
        config = yaml.safe_load((root / "resources" / name).read_text(encoding="utf-8"))
        assert config["filters"]["fir_L"]["parameters"]["filename"] == "resources/impulse_L.wav"
        assert config["filters"]["fir_R"]["parameters"]["filename"] == "resources/impulse_R.wav"
