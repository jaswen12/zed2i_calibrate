"""
Configuration loader.

All scripts call load_config() to get the merged config dict.
The config file path defaults to <repo_root>/config/calibration.yaml.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo root is two levels above this file: src/zed2i_calibrate/config.py
_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "calibration.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """
    Load calibration config from YAML.

    Args:
        path: Path to YAML file. Defaults to config/calibration.yaml in repo root.

    Returns:
        Parsed config dict.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def repo_root() -> Path:
    """Absolute path to repo root."""
    return _REPO_ROOT


def resolve_path(cfg: dict, key: str) -> Path:
    """
    Resolve a path from config['paths'] relative to repo root.

    Example:
        resolve_path(cfg, "stereo_samples")  # -> /abs/path/data/stereo_samples
    """
    rel = cfg["paths"][key]
    return (_REPO_ROOT / rel).resolve()


def resolve_output(cfg: dict, key: str) -> Path:
    """
    Resolve an output file path from config['output'] relative to repo root.

    Example:
        resolve_output(cfg, "zed_intrinsics")  # -> /abs/path/results/zed_intrinsics.yaml
    """
    rel = cfg["output"][key]
    return (_REPO_ROOT / rel).resolve()
