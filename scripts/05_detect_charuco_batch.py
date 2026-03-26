#!/usr/bin/env python3
"""
Script 05: Batch ChArUco detection on collected images.

Runs detection on all images in a samples directory and reports:
  - Which images succeeded / failed
  - Per-image corner count and reprojection error
  - Summary statistics

This is useful for:
  1. Quality-checking collected samples before running hand-eye solve
  2. Identifying bad images to re-capture
  3. Offline re-detection after collecting (e.g., with different parameters)

Usage:
    python scripts/05_detect_charuco_batch.py data/eye_on_base/images
    python scripts/05_detect_charuco_batch.py data/eye_on_hand/images --min-corners 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from zed2i_calibrate.board import CharucoBoard
from zed2i_calibrate.config import load_config, resolve_output
from zed2i_calibrate.io import load_intrinsics, load_stereo_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ChArUco detection.")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing images (*.png / *.jpg).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--min-corners", type=int, default=6,
                        help="Minimum corners for valid detection (default: 6).")
    parser.add_argument("--show", action="store_true",
                        help="Display each detection result in a window.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    board = CharucoBoard.from_config(cfg)

    # Load intrinsics
    stereo_path = resolve_output(cfg, "stereo_extrinsics")
    factory_path = resolve_output(cfg, "zed_intrinsics")

    if stereo_path.exists():
        cal = load_stereo_calibration(stereo_path)
        K, D = cal["K_left"], cal["D_left"]
        print(f"[init] Intrinsics from: {stereo_path}")
    elif factory_path.exists():
        intr = load_intrinsics(factory_path)
        K, D = intr.K_left, intr.D_left
        print(f"[init] Intrinsics from: {factory_path}")
    else:
        print("[WARN] No intrinsics found. Running detection without pose estimation.")
        K, D = None, None

    # Find images
    img_dir = Path(args.image_dir)
    if not img_dir.exists():
        print(f"[ERROR] Directory not found: {img_dir}")
        return

    image_files = sorted(
        list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    )
    if not image_files:
        print(f"[ERROR] No images found in {img_dir}")
        return

    print(f"\n=== Batch Detection: {len(image_files)} images ===\n")

    successes = []
    failures = []
    errors = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] Cannot read: {img_path.name}")
            failures.append(img_path.name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = board.detect(gray, K, D, min_corners=args.min_corners)

        if result.valid:
            reproj = result.reprojection_error
            reproj_str = f"{reproj:.3f}px" if reproj is not None else "N/A"
            print(f"  [+] {img_path.name}: {result.n_corners} corners, err={reproj_str}")
            successes.append(img_path.name)
            if reproj is not None:
                errors.append(reproj)

            if args.show:
                vis = board.draw_detected(img, result,
                                          camera_matrix=K, dist_coeffs=D)
                cv2.imshow(f"Detection: {img_path.name}", vis)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                if key == ord("q"):
                    break
        else:
            print(f"  [-] {img_path.name}: FAILED (corners={result.n_corners})")
            failures.append(img_path.name)

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total images:  {len(image_files)}")
    print(f"  Succeeded:     {len(successes)}")
    print(f"  Failed:        {len(failures)}")

    if errors:
        print(f"  Reprojection error:")
        print(f"    Mean:  {np.mean(errors):.4f} px")
        print(f"    Std:   {np.std(errors):.4f} px")
        print(f"    Max:   {np.max(errors):.4f} px")
        print(f"    Min:   {np.min(errors):.4f} px")

    if failures:
        print(f"\n  Failed images:")
        for f in failures:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
