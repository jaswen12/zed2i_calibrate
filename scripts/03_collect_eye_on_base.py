#!/usr/bin/env python3
"""
Script 03: Collect hand-eye samples — Eye-on-Base configuration.

Setup:
  - Camera is FIXED (tripod / table mount), does NOT move.
  - ChArUco board is attached to the robot flange.
  - Robot moves the board to various poses; each pose is captured.

Usage:
    python scripts/03_collect_eye_on_base.py --live
    python scripts/03_collect_eye_on_base.py --live --mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent))

from zed2i_calibrate.board import CharucoBoard
from zed2i_calibrate.config import load_config
from zed2i_calibrate.handeye_collect import HandEyeCollector

import _shared_handeye as shared


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect eye-on-base hand-eye samples.")
    parser.add_argument("--live", action="store_true", required=True,
                        help="Interactive collection mode.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock robot bridge (no real robot).")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    board = CharucoBoard.from_config(cfg)
    K, D = shared.load_camera_matrix(cfg)

    cfg["handeye"]["mode"] = "eye_on_base"
    collector = HandEyeCollector(board, cfg, camera_matrix=K, dist_coeffs=D)

    shared.run_live(
        cfg=cfg,
        board=board,
        collector=collector,
        use_mock=args.mock,
        data_key="eye_on_base_samples",
        window_title="Eye-on-Base",
    )


if __name__ == "__main__":
    main()
