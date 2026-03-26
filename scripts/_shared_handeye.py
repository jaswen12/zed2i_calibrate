"""
Shared utilities for eye-on-base (script 03) and eye-on-hand (script 04).

NOT a user-facing script. Imported by 03 and 04 to avoid code duplication.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from zed2i_calibrate.board import CharucoBoard
from zed2i_calibrate.camera import open_camera
from zed2i_calibrate.config import resolve_output, resolve_path
from zed2i_calibrate.handeye_collect import HandEyeCollector
from zed2i_calibrate.io import load_intrinsics, load_stereo_calibration


def load_camera_matrix(cfg: dict) -> tuple:
    """Load camera matrix from best available source (stereo > factory)."""
    stereo_path = resolve_output(cfg, "stereo_extrinsics")
    factory_path = resolve_output(cfg, "zed_intrinsics")

    if stereo_path.exists():
        print(f"[init] Using stereo calibration intrinsics: {stereo_path}")
        cal = load_stereo_calibration(stereo_path)
        return cal["K_left"], cal["D_left"]
    elif factory_path.exists():
        print(f"[init] Using factory intrinsics: {factory_path}")
        intr = load_intrinsics(factory_path)
        return intr.K_left, intr.D_left
    else:
        raise FileNotFoundError(
            "No intrinsics found. Run script 01 or 02 first.\n"
            f"  Expected: {stereo_path} or {factory_path}"
        )


def open_robot_bridge(cfg: dict, use_mock: bool):
    """
    Open robot bridge.

    Mock mode priority:
      1. HERMES MockRobotBridge (if hermes is installed)
      2. Local LocalMockRobotBridge (no HERMES dependency)

    Real mode:
      Requires HERMES with FlexivRobotBridge.
    """
    if use_mock:
        try:
            from hermes.bridge.mock import MockRobotBridge
            bridge = MockRobotBridge()
        except ImportError:
            from zed2i_calibrate.robot_mock import LocalMockRobotBridge
            bridge = LocalMockRobotBridge()
    else:
        from hermes.bridge.flexiv import FlexivRobotBridge
        from hermes.bridge.base import BridgeConfig
        robot_cfg = cfg.get("handeye", {}).get("robot", {})
        bc = BridgeConfig(
            robot_sn=robot_cfg.get("serial_number", "Rizon4s-062626"),
            local_ip=robot_cfg.get("local_ip") or "",
        )
        bridge = FlexivRobotBridge(bc)
    return bridge


def run_live(
    cfg: dict,
    board: CharucoBoard,
    collector: HandEyeCollector,
    use_mock: bool,
    data_key: str = "eye_on_base_samples",
    window_title: str = "Eye-on-Base",
) -> None:
    """Interactive collection: move robot, press SPACE to capture."""
    settle_time = cfg.get("handeye", {}).get("settle_time_s", 1.5)
    output_dir = resolve_path(cfg, data_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge = open_robot_bridge(cfg, use_mock)
    ok = bridge.connect()
    if not ok:
        print("[ERROR] Failed to connect to robot bridge.")
        return

    print(f"\n=== {window_title} Collection ===")
    print(f"  Robot: {bridge.name}")
    print(f"  Mode: {collector.mode}")
    print(f"  Target: {collector.min_samples}–{collector.max_samples} samples")
    print(f"  Settle time: {settle_time}s")
    print(f"  SPACE = capture | Q = finish\n")

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    try:
        with open_camera(cfg) as cam:
            while not collector.is_full:
                frame = cam.grab()
                gray = cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY)

                state = bridge.read_state()
                if state is None:
                    continue

                tcp = np.array(state.tcp_pose)

                # Preview
                result = board.detect(gray, collector.camera_matrix, collector.dist_coeffs)
                vis = board.draw_detected(
                    frame.left, result,
                    camera_matrix=collector.camera_matrix,
                    dist_coeffs=collector.dist_coeffs,
                )

                status = f"Samples: {collector.n_samples}/{collector.max_samples}"
                if result.valid:
                    status += f" | {result.n_corners} corners"
                    if result.reprojection_error is not None:
                        status += f" | err={result.reprojection_error:.2f}px"
                else:
                    status += " | Board NOT detected"

                cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

                tcp_str = (f"TCP: x={tcp[0]:.3f} y={tcp[1]:.3f} z={tcp[2]:.3f}")
                cv2.putText(vis, tcp_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 200, 0), 1)

                cv2.imshow(f"{window_title} (SPACE=capture, Q=quit)", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord(" "):
                    print(f"  Waiting {settle_time}s for settle...")
                    time.sleep(settle_time)

                    frame = cam.grab()
                    gray = cv2.cvtColor(frame.left, cv2.COLOR_BGR2GRAY)
                    state = bridge.read_state()
                    if state is None:
                        print("  [-] Robot state unavailable")
                        continue

                    tcp = np.array(state.tcp_pose)
                    q = np.array(state.q)

                    sample = collector.add_sample(gray, tcp, q)
                    if sample is not None:
                        cv2.imwrite(
                            str(images_dir / f"{sample.index:04d}.png"),
                            frame.left,
                        )
                        print(
                            f"  [+] Sample {collector.n_samples}: "
                            f"{sample.n_corners} corners, "
                            f"err={sample.reprojection_error:.2f}px"
                        )
                    else:
                        print("  [-] Failed: detection or quality check failed")
    finally:
        cv2.destroyAllWindows()
        bridge.disconnect()

    if collector.n_samples > 0:
        collector.save(output_dir)
        print(f"\nSaved {collector.n_samples} samples → {output_dir}")
    else:
        print("\nNo samples collected.")
