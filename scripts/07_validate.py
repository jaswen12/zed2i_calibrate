#!/usr/bin/env python3
"""
Script 07: Validate calibration results.

Runs quality checks on hand-eye (and optionally stereo) calibration:
  - Board-in-base consistency (should converge to a single position)
  - Leave-one-out stability
  - Cross-method comparison

Usage:
    python scripts/07_validate.py
    python scripts/07_validate.py --mode eye_on_hand
"""

from __future__ import annotations

import argparse

import numpy as np

from zed2i_calibrate.config import load_config, resolve_output, resolve_path
from zed2i_calibrate.handeye_collect import HandEyeCollector
from zed2i_calibrate.handeye_solve import solve_all_methods
from zed2i_calibrate.io import load_transform
from zed2i_calibrate.validate import compare_methods, validate_handeye


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate calibration results")
    parser.add_argument(
        "--mode",
        choices=["eye_on_base", "eye_on_hand"],
        default=None,
        help="Override mode (default: from config)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to calibration.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.get("handeye", {}).get("mode", "eye_on_base")

    if mode == "eye_on_base":
        data_key = "eye_on_base_samples"
        output_key = "eye_on_base"
    else:
        data_key = "eye_on_hand_samples"
        output_key = "eye_on_hand"

    data_dir = resolve_path(cfg, data_key)
    output_path = resolve_output(cfg, output_key)

    # Load samples
    T_bt, T_cb, meta = HandEyeCollector.load_samples(data_dir)
    n = len(T_bt)
    print(f"Loaded {n} samples ({mode}) from {data_dir}\n")

    # Load solved transform
    T_result = load_transform(output_path, label="T")
    print(f"Loaded result from {output_path}")
    print(f"  T =\n{np.array2string(T_result, precision=6, suppress_small=True)}\n")

    # Validate
    val = validate_handeye(
        T_result=T_result,
        T_base_tcp_list=T_bt,
        T_cam_board_list=T_cb,
        mode=mode,
        method_name="saved",
    )
    print(val.summary())

    # Cross-method comparison
    print("\n--- Cross-method comparison ---\n")
    results = solve_all_methods(T_bt, T_cb, mode=mode)
    print(compare_methods(results))

    # Per-sample board-in-base breakdown
    if val.board_positions_base is not None:
        print("\n--- Per-sample board offset consistency ---")
        print(f"{'Sample':>6} {'Offset X':>10} {'Y':>10} {'Z':>10} {'Norm':>10}")
        print("-" * 52)

        if mode == "eye_on_base":
            T_base_cam = np.linalg.inv(T_result)
            offsets = []
            for i, (T_bt_i, T_cb_i) in enumerate(zip(T_bt, T_cb)):
                T_base_board = T_base_cam @ T_cb_i
                T_tcp_board = np.linalg.inv(T_bt_i) @ T_base_board
                offsets.append(T_tcp_board[:3, 3])
            offsets = np.array(offsets)
        else:
            T_gripper_cam = np.linalg.inv(T_result)
            positions = []
            for T_bt_i, T_cb_i in zip(T_bt, T_cb):
                T_base_board = T_bt_i @ T_gripper_cam @ T_cb_i
                positions.append(T_base_board[:3, 3])
            offsets = np.array(positions)

        mean_offset = offsets.mean(axis=0)
        for i, o in enumerate(offsets):
            dev = o - mean_offset
            norm = np.linalg.norm(dev) * 1000
            print(
                f"{i:6d} "
                f"{dev[0]*1000:+10.2f} {dev[1]*1000:+10.2f} {dev[2]*1000:+10.2f} "
                f"{norm:10.2f} mm"
            )

    # Quality summary
    print("\n--- Quality Summary ---")
    if val.board_position_std_mm < 5.0:
        print(f"  Board consistency: GOOD (std = {val.board_position_std_mm:.2f} mm)")
    elif val.board_position_std_mm < 15.0:
        print(f"  Board consistency: ACCEPTABLE (std = {val.board_position_std_mm:.2f} mm)")
    else:
        print(f"  Board consistency: POOR (std = {val.board_position_std_mm:.2f} mm)")
        print("  Consider: more samples, better board coverage, or check robot accuracy")

    if val.board_position_max_mm > 20.0:
        print(f"  WARNING: max deviation = {val.board_position_max_mm:.2f} mm (outlier likely)")
        print("  Consider removing worst sample and re-solving")

    print("\nDone.")


if __name__ == "__main__":
    main()
