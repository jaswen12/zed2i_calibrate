#!/usr/bin/env python3
"""
Script 01b: Standalone stereo capture via UVC.

Captures ZED2i stereo image pairs using only cv2.VideoCapture — no ZED SDK,
no pyzed, no NVIDIA GPU required. The ZED2i exposes itself as a UVC device
streaming a side-by-side image; this script splits the frame and saves
left_NNNN.png / right_NNNN.png pairs that script 02 can consume offline.

Use this when:
  - The lab machine has no NVIDIA GPU (ZED SDK won't install)
  - You want to capture data on a different computer than where you calibrate
  - You can't or don't want to `pip install -e .` on the capture machine

Dependencies: only opencv-contrib-python and numpy (stdlib otherwise).

Usage:
    python scripts/01b_capture_uvc_only.py
    python scripts/01b_capture_uvc_only.py --resolution HD720
    python scripts/01b_capture_uvc_only.py --device 2 --output /tmp/captures
    python scripts/01b_capture_uvc_only.py --no-detect    # skip ChArUco preview

After capturing, run script 02 in offline mode:
    python scripts/02_stereo_calibrate.py --offline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# Side-by-side UVC modes for ZED2i: (combined_width, height, max_fps).
RESOLUTION_MAP: dict[str, Tuple[int, int, int]] = {
    "HD2K":   (4416, 1242, 15),
    "HD1080": (3840, 1080, 30),
    "HD720":  (2560, 720, 60),
    "VGA":    (1344, 376, 100),
}

# OpenCV ArUco dictionary names → constants (lazy-resolved to handle older OpenCV).
def _aruco_dict(name: str) -> int:
    table = {
        "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250":  cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100":  cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250":  cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_250":  cv2.aruco.DICT_6X6_250,
        "DICT_7X7_250":  cv2.aruco.DICT_7X7_250,
    }
    if name not in table:
        raise SystemExit(f"Unknown ArUco dictionary: {name}. Valid: {list(table)}")
    return table[name]


def load_config_if_present(config_path: Optional[Path]) -> Optional[dict]:
    """Try to load YAML config; return None if PyYAML or file is unavailable."""
    if config_path is None or not config_path.exists():
        return None
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        print("[capture] PyYAML not installed — running without config defaults")
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def open_uvc(device_index: int, resolution: str) -> Tuple["cv2.VideoCapture", Tuple[int, int]]:
    """Open a UVC device and request the right side-by-side resolution."""
    if resolution not in RESOLUTION_MAP:
        raise SystemExit(
            f"Bad resolution {resolution!r}. Choose from {list(RESOLUTION_MAP)}"
        )
    w_total, h, max_fps = RESOLUTION_MAP[resolution]

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open device index {device_index}.\n"
            f"  Linux:  v4l2-ctl --list-devices  (try a different index)\n"
            f"  macOS:  System Settings → Privacy → Camera permission\n"
            f"  lsusb | grep -i 'stereolabs\\|zed'  (verify ZED is connected)"
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_total)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, max_fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"[capture] Device {device_index}: requested {w_total}x{h}, "
        f"got {actual_w}x{actual_h} @ {actual_fps:.0f}fps"
    )
    if (actual_w, actual_h) != (w_total, h):
        print(
            "[capture] WARNING: requested resolution not honored. The device "
            "may not be a ZED2i, or the OS rejected the request. The split into "
            "left/right halves assumes a stereo side-by-side stream."
        )

    return cap, (actual_w, actual_h)


def split_stereo(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame into (left, right)."""
    w = frame.shape[1]
    mid = w // 2
    return frame[:, :mid].copy(), frame[:, mid:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone UVC stereo capture (no ZED SDK required)."
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="UVC device index (default: from config, else 0).",
    )
    parser.add_argument(
        "--resolution", type=str, default=None,
        choices=list(RESOLUTION_MAP.keys()),
        help="Stereo resolution (default: from config, else HD1080).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: data/stereo_samples relative to repo).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to calibration.yaml (default: config/calibration.yaml).",
    )
    parser.add_argument(
        "--no-detect", action="store_true",
        help="Skip live ChArUco detection overlay (lower latency preview).",
    )
    args = parser.parse_args()

    # Locate repo root and defaults
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    default_config = repo_root / "config" / "calibration.yaml"
    config_path = Path(args.config) if args.config else default_config
    cfg = load_config_if_present(config_path)

    # Resolve params with this precedence: CLI > config > built-in default
    cam_cfg = (cfg or {}).get("camera", {})
    board_cfg = (cfg or {}).get("board", {})

    device = args.device if args.device is not None else int(cam_cfg.get("uvc_device_index", 0))
    resolution = args.resolution or cam_cfg.get("resolution", "HD1080")

    if args.output:
        output_dir = Path(args.output)
    else:
        paths = (cfg or {}).get("paths", {})
        rel = paths.get("stereo_samples", "data/stereo_samples")
        output_dir = repo_root / rel
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    board = None
    if not args.no_detect and board_cfg:
        try:
            dict_const = _aruco_dict(board_cfg.get("aruco_dict", "DICT_5X5_1000"))
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)
            board = cv2.aruco.CharucoBoard(
                (int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
                float(board_cfg["square_length_mm"]) / 1000.0,
                float(board_cfg["marker_length_mm"]) / 1000.0,
                aruco_dict,
            )
            detector = cv2.aruco.CharucoDetector(board)
            print(
                f"[capture] ChArUco preview ON ({board_cfg['squares_x']}x"
                f"{board_cfg['squares_y']}, {board_cfg['aruco_dict']})"
            )
        except (KeyError, ValueError) as e:
            print(f"[capture] ChArUco preview disabled (config error: {e})")

    print(f"[capture] Output: {output_dir}")
    print(f"[capture] Resolution: {resolution}, device: {device}")

    cap, (actual_w, actual_h) = open_uvc(device, resolution)
    eye_w = actual_w // 2

    # Determine starting index so we don't overwrite previous captures.
    existing = sorted(output_dir.glob("left_*.png"))
    next_idx = 0
    if existing:
        try:
            last = int(existing[-1].stem.split("_")[-1])
            next_idx = last + 1
            print(f"[capture] Resuming from index {next_idx} "
                  f"({len(existing)} existing pairs in directory)")
        except ValueError:
            pass

    print()
    print("Controls:")
    print("  SPACE  capture a stereo pair")
    print("  Q      quit")
    print("  R      reset counter (overwrite from 0; will prompt)")
    print()

    captured = 0
    window_name = "UVC Stereo Capture (SPACE=capture, Q=quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[capture] grab failed, retrying...")
                continue

            left, right = split_stereo(frame)

            # Live ChArUco detection overlay (left eye only, for speed)
            corners_l = ids_l = None
            corners_r = ids_r = None
            if detector is not None:
                gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                corners_l, ids_l, _, _ = detector.detectBoard(gray_l)
                corners_r, ids_r, _, _ = detector.detectBoard(gray_r)

            vis_l = left.copy()
            vis_r = right.copy()
            if corners_l is not None and ids_l is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis_l, corners_l, ids_l)
            if corners_r is not None and ids_r is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis_r, corners_r, ids_r)

            n_l = 0 if corners_l is None else len(corners_l)
            n_r = 0 if corners_r is None else len(corners_r)
            both_ok = n_l > 0 and n_r > 0
            status_color = (0, 255, 0) if both_ok else (0, 165, 255)
            status = (
                f"Captured: {captured}  next idx: {next_idx}  "
                f"L:{n_l} R:{n_r} corners  ({eye_w}x{actual_h} per eye)"
            )
            if detector is None:
                status = f"Captured: {captured}  next idx: {next_idx}  ({eye_w}x{actual_h} per eye)"

            preview = np.hstack([vis_l, vis_r])
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_color, 2)

            # Downscale preview to fit screen
            max_display_w = 1800
            if preview.shape[1] > max_display_w:
                scale = max_display_w / preview.shape[1]
                preview = cv2.resize(preview, None, fx=scale, fy=scale)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                print("[capture] Reset requested. Type 'yes' to confirm overwriting "
                      "from 0 (current counter will reset, files NOT deleted): ", end="")
                ans = input().strip().lower()
                if ans == "yes":
                    next_idx = 0
                    captured = 0
                    print("[capture] Counter reset to 0.")
                else:
                    print("[capture] Reset cancelled.")
            elif key == ord(" "):
                # When ChArUco is on, only capture if both views see the board.
                if detector is not None and not both_ok:
                    print(
                        f"  [skip] L:{n_l} R:{n_r} corners — need both views detecting. "
                        f"Move the board so both eyes see it."
                    )
                    continue

                left_path = output_dir / f"left_{next_idx:04d}.png"
                right_path = output_dir / f"right_{next_idx:04d}.png"
                cv2.imwrite(str(left_path), left)
                cv2.imwrite(str(right_path), right)
                print(f"  [+] {left_path.name} / {right_path.name}  (L:{n_l} R:{n_r} corners)")
                next_idx += 1
                captured += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print()
    print(f"[capture] Saved {captured} pair(s) to {output_dir}")
    print()
    if captured > 0:
        print("Next step (on a machine with the package installed):")
        print(f"    python scripts/02_stereo_calibrate.py --offline "
              f"--samples-dir {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[capture] Interrupted.")
        sys.exit(0)
