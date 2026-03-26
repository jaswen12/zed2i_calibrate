#!/usr/bin/env python3
"""
Script 02: Stereo camera calibration.

Two modes:
  --live     Capture images from ZED (or mock) interactively.
             Press SPACE to capture, Q to finish and calibrate.

  --offline  Load previously saved image pairs from data/stereo_samples/.
             Expects left_0000.png / right_0000.png naming convention.

Outputs:
  results/stereo_extrinsics.yaml   (K_left, D_left, K_right, D_right, R, T)

Usage:
    python scripts/02_stereo_calibrate.py --live
    python scripts/02_stereo_calibrate.py --offline
    python scripts/02_stereo_calibrate.py --offline --samples-dir data/stereo_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from zed2i_calibrate.board import CharucoBoard
from zed2i_calibrate.camera import open_camera
from zed2i_calibrate.config import load_config, resolve_output, resolve_path
from zed2i_calibrate.io import load_intrinsics, save_stereo_calibration
from zed2i_calibrate.stereo_calibrate import StereoCalibrator


def run_live(cfg: dict, board: CharucoBoard, calibrator: StereoCalibrator) -> None:
    """Capture stereo pairs interactively from camera."""
    sample_dir = resolve_path(cfg, "stereo_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)

    left_images = []
    right_images = []

    with open_camera(cfg) as cam:
        intrinsics = cam.get_intrinsics()

        print("\n=== Live Stereo Calibration ===")
        print("  SPACE  = capture current frame")
        print("  Q      = finish and calibrate")
        print(f"  Target: {calibrator.min_samples}–{calibrator.max_samples} samples\n")

        while not calibrator.is_full:
            frame = cam.grab()
            gray_l = cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame.right, cv2.COLOR_BGR2GRAY)

            # Preview with detection overlay
            result_l = board.detect(gray_l)
            result_r = board.detect(gray_r)

            vis_l = board.draw_detected(frame.left, result_l, draw_axes=False)
            vis_r = board.draw_detected(frame.right, result_r, draw_axes=False)

            status = f"Samples: {calibrator.n_samples}/{calibrator.max_samples}"
            if result_l.valid and result_r.valid:
                status += f" | L:{result_l.n_corners} R:{result_r.n_corners} corners"
            else:
                status += " | Board not detected in both views"

            cv2.putText(vis_l, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            combined = np.hstack([vis_l, vis_r])
            # Resize for display if too wide
            max_w = 1920
            if combined.shape[1] > max_w:
                scale = max_w / combined.shape[1]
                combined = cv2.resize(combined, None, fx=scale, fy=scale)

            cv2.imshow("Stereo Calibration (SPACE=capture, Q=quit)", combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                sample = calibrator.add_sample(gray_l, gray_r)
                if sample is not None:
                    left_images.append(frame.left)
                    right_images.append(frame.right)
                    print(f"  [+] Sample {calibrator.n_samples}: "
                          f"{sample.n_corners} corners matched")
                else:
                    print("  [-] Failed: not enough matching corners in both views")

    cv2.destroyAllWindows()

    if left_images:
        calibrator.save_sample_images(sample_dir, left_images, right_images)


def run_offline(
    cfg: dict,
    board: CharucoBoard,
    calibrator: StereoCalibrator,
    samples_dir: Path,
) -> None:
    """Load image pairs from disk and detect ChArUco."""
    print(f"\n=== Offline Stereo Calibration ===")
    print(f"  Loading from: {samples_dir}\n")

    left_files = sorted(samples_dir.glob("left_*.png"))
    right_files = sorted(samples_dir.glob("right_*.png"))

    if len(left_files) == 0:
        print("No image pairs found. Expected files: left_0000.png, right_0000.png, ...")
        return

    if len(left_files) != len(right_files):
        print(f"Mismatch: {len(left_files)} left, {len(right_files)} right images.")
        return

    for lf, rf in zip(left_files, right_files):
        gray_l = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        gray_r = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)

        if gray_l is None or gray_r is None:
            print(f"  [-] Failed to read: {lf.name} / {rf.name}")
            continue

        sample = calibrator.add_sample(gray_l, gray_r)
        if sample is not None:
            print(f"  [+] {lf.name}: {sample.n_corners} corners")
        else:
            print(f"  [-] {lf.name}: detection failed")

        if calibrator.is_full:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo camera calibration.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Capture from camera.")
    mode.add_argument("--offline", action="store_true", help="Load images from disk.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Override samples directory (offline mode).")
    parser.add_argument("--factory-intrinsics", type=str, default=None,
                        help="Path to factory intrinsics YAML for initial guess.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    board = CharucoBoard.from_config(cfg)
    calibrator = StereoCalibrator(board, cfg)

    # Optionally load factory intrinsics as initial guess
    K_l_init = K_r_init = D_l_init = D_r_init = None
    factory_path = args.factory_intrinsics
    if factory_path is None:
        default_factory = resolve_output(cfg, "zed_intrinsics")
        if default_factory.exists():
            factory_path = str(default_factory)

    if factory_path:
        print(f"Using factory intrinsics as initial guess: {factory_path}")
        intr = load_intrinsics(factory_path)
        K_l_init, D_l_init = intr.K_left, intr.D_left
        K_r_init, D_r_init = intr.K_right, intr.D_right

    # Collect samples
    if args.live:
        run_live(cfg, board, calibrator)
    else:
        samples_dir = Path(args.samples_dir) if args.samples_dir else resolve_path(cfg, "stereo_samples")
        run_offline(cfg, board, calibrator, samples_dir)

    # Calibrate
    if not calibrator.is_ready:
        print(f"\nNot enough samples ({calibrator.n_samples}/{calibrator.min_samples}). "
              f"Collect more image pairs.")
        return

    print(f"\nCalibrating with {calibrator.n_samples} samples...")
    result = calibrator.calibrate(K_l_init, D_l_init, K_r_init, D_r_init)
    print(f"\n{result.summary()}")

    # Save
    output_path = resolve_output(cfg, "stereo_extrinsics")
    save_stereo_calibration(
        output_path,
        result.K_left, result.D_left,
        result.K_right, result.D_right,
        result.R, result.T,
        result.image_size,
        result.rms_error,
        result.n_samples,
    )
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
