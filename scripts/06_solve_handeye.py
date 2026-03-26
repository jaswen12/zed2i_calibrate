#!/usr/bin/env python3
"""
Script 06: Solve hand-eye calibration.

Loads collected samples (from script 03 or 04), runs all 5 OpenCV methods,
compares results, and saves the best transform.

Usage:
    # Eye-on-base (default from config)
    python scripts/06_solve_handeye.py

    # Override mode via CLI
    python scripts/06_solve_handeye.py --mode eye_on_hand

    # Specify a single method
    python scripts/06_solve_handeye.py --method TSAI
"""

from __future__ import annotations

import argparse
import sys

from zed2i_calibrate.config import load_config, resolve_output, resolve_path
from zed2i_calibrate.handeye_collect import HandEyeCollector
from zed2i_calibrate.handeye_solve import (
    HANDEYE_METHODS,
    pick_best,
    solve_all_methods,
    solve_handeye,
)
from zed2i_calibrate.validate import compare_methods
from zed2i_calibrate.io import save_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve hand-eye calibration")
    parser.add_argument(
        "--mode",
        choices=["eye_on_base", "eye_on_hand"],
        default=None,
        help="Override mode (default: from config)",
    )
    parser.add_argument(
        "--method",
        choices=list(HANDEYE_METHODS.keys()),
        default=None,
        help="Run only this method (default: all)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to calibration.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("handeye", {}).get("mode", "eye_on_base")

    # Determine data directory
    if mode == "eye_on_base":
        data_key = "eye_on_base_samples"
        output_key = "eye_on_base"
    else:
        data_key = "eye_on_hand_samples"
        output_key = "eye_on_hand"

    data_dir = resolve_path(cfg, data_key)
    output_path = resolve_output(cfg, output_key)

    print(f"Mode:      {mode}")
    print(f"Data:      {data_dir}")
    print(f"Output:    {output_path}")
    print()

    # Load samples
    T_bt, T_cb, meta = HandEyeCollector.load_samples(data_dir)
    n = len(T_bt)
    print(f"Loaded {n} samples from {data_dir}")
    print()

    if args.method:
        # Single method
        method_enum = HANDEYE_METHODS[args.method]
        result = solve_handeye(T_bt, T_cb, mode=mode, method=method_enum, method_name=args.method)
        print(result.summary())
        results = [result]
    else:
        # All methods
        results = solve_all_methods(T_bt, T_cb, mode=mode)
        print(compare_methods(results))
        print()

    if not results:
        print("ERROR: No valid results. Check your data.")
        sys.exit(1)

    best = pick_best(results)
    print(f"\nSaving best result ({best.method_name}) → {output_path}")

    save_transform(
        output_path,
        best.T,
        label="T",
        metadata={
            "method": best.method_name,
            "mode": mode,
            "n_samples": best.n_samples,
            "loo_rotation_error_deg": best.rotation_error_deg,
            "loo_translation_error_mm": best.translation_error_mm,
        },
    )

    print("\nDone. Run script 07 to validate.")


if __name__ == "__main__":
    main()
