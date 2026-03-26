#!/usr/bin/env python3
"""
Script 01: Export ZED2i factory intrinsics to OpenCV YAML.

Reads built-in (factory) intrinsic parameters from the ZED SDK and saves them
as a baseline. On macOS (no ZED SDK), exports mock intrinsics for development.

Usage:
    python scripts/01_export_zed_intrinsics.py
    python scripts/01_export_zed_intrinsics.py --config config/calibration.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zed2i_calibrate.camera import open_camera
from zed2i_calibrate.config import load_config, resolve_output
from zed2i_calibrate.io import save_intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ZED2i factory intrinsics.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to calibration.yaml (default: config/calibration.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = resolve_output(cfg, "zed_intrinsics")

    with open_camera(cfg) as cam:
        intrinsics = cam.get_intrinsics()

    print(f"\n{intrinsics}")
    print(f"  Left  K:\n{intrinsics.K_left}")
    print(f"  Right K:\n{intrinsics.K_right}")
    print(f"  Baseline: {abs(intrinsics.T[0]) * 1000:.1f} mm")

    save_intrinsics(output_path, intrinsics)
    print(f"\nDone. Factory intrinsics saved to: {output_path}")


if __name__ == "__main__":
    main()
