#!/usr/bin/env python3
"""Compatibility launcher for FIRC."""

from firc.app import FIRFilterGUI
from firc.config import _build_yaml_config, _read_yaml_config
from firc.main import main

__all__ = ["FIRFilterGUI", "_build_yaml_config", "_read_yaml_config", "main"]


if __name__ == "__main__":
    main()
