"""
Hand-eye calibration solver.

Wraps cv2.calibrateHandEye with support for:
  - eye_on_base: camera fixed externally, board on robot gripper
  - eye_on_hand: camera on robot flange, board fixed in world

Runs multiple OpenCV methods and compares results so the user can pick
the most consistent solution.

Usage:
    from zed2i_calibrate.handeye_solve import solve_handeye, solve_all_methods
    from zed2i_calibrate.handeye_collect import HandEyeCollector

    T_bt, T_cb, meta = HandEyeCollector.load_samples("data/eye_on_base")
    result = solve_handeye(T_bt, T_cb, mode="eye_on_base", method=cv2.HAND_EYE_TSAI)
    all_results = solve_all_methods(T_bt, T_cb, mode="eye_on_base")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# OpenCV method map
# ---------------------------------------------------------------------------

HANDEYE_METHODS: Dict[str, int] = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class HandEyeResult:
    """Result of a single hand-eye solve."""

    T: np.ndarray             # (4, 4) solved transform
    method_name: str          # e.g. "TSAI"
    mode: str                 # "eye_on_base" or "eye_on_hand"
    n_samples: int
    rotation_error_deg: float  # leave-one-out consistency (populated by validate)
    translation_error_mm: float

    @property
    def t_mm(self) -> np.ndarray:
        """Translation in millimetres."""
        return self.T[:3, 3] * 1000.0

    @property
    def rvec(self) -> np.ndarray:
        """Rodrigues rotation vector."""
        r, _ = cv2.Rodrigues(self.T[:3, :3])
        return r.ravel()

    def summary(self) -> str:
        t = self.t_mm
        angle = np.degrees(np.linalg.norm(self.rvec))
        return (
            f"[{self.method_name:11s}] "
            f"tx={t[0]:+8.2f} ty={t[1]:+8.2f} tz={t[2]:+8.2f} mm  "
            f"rot={angle:6.2f}°  "
            f"Δrot={self.rotation_error_deg:.3f}° Δt={self.translation_error_mm:.2f}mm"
        )


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def _decompose(T_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split list of 4x4 transforms into separate R and t lists."""
    Rs = [T[:3, :3] for T in T_list]
    ts = [T[:3, 3].reshape(3, 1) for T in T_list]
    return Rs, ts


def solve_handeye(
    T_base_tcp_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    mode: str = "eye_on_base",
    method: int = cv2.CALIB_HAND_EYE_TSAI,
    method_name: str = "TSAI",
) -> HandEyeResult:
    """
    Solve hand-eye calibration for one method.

    Args:
        T_base_tcp_list: List of (4,4) robot base←tcp transforms.
        T_cam_board_list: List of (4,4) camera←board transforms.
        mode: "eye_on_base" or "eye_on_hand".
        method: OpenCV HandEyeCalibrationMethod enum.
        method_name: Human-readable name.

    Returns:
        HandEyeResult with the solved 4x4 transform.

    For eye_on_base (camera fixed, board on gripper):
        Input A = inv(T_base_tcp)  →  "gripper2base" in OpenCV terms
        Input B = T_cam_board      →  "target2cam"
        Output X = T_cam←base  (camera to robot base)

    For eye_on_hand (camera on flange, board fixed):
        Input A = T_base_tcp       →  "gripper2base"
        Input B = T_cam_board      →  "target2cam"
        Output X = T_gripper←cam  (end-effector to camera)
    """
    n = len(T_base_tcp_list)
    if n != len(T_cam_board_list):
        raise ValueError(
            f"Mismatched sample counts: {n} robot vs {len(T_cam_board_list)} camera"
        )
    if n < 3:
        raise ValueError(f"Need at least 3 samples, got {n}")

    if mode == "eye_on_base":
        # Camera fixed → invert robot poses
        A_list = [np.linalg.inv(T) for T in T_base_tcp_list]
    elif mode == "eye_on_hand":
        A_list = list(T_base_tcp_list)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    R_A, t_A = _decompose(A_list)
    R_B, t_B = _decompose(T_cam_board_list)

    R_X, t_X = cv2.calibrateHandEye(R_A, t_A, R_B, t_B, method=method)

    T_X = np.eye(4)
    T_X[:3, :3] = R_X
    T_X[:3, 3] = t_X.ravel()

    return HandEyeResult(
        T=T_X,
        method_name=method_name,
        mode=mode,
        n_samples=n,
        rotation_error_deg=0.0,
        translation_error_mm=0.0,
    )


def solve_all_methods(
    T_base_tcp_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    mode: str = "eye_on_base",
) -> List[HandEyeResult]:
    """
    Run all 5 OpenCV methods and return sorted results (best first).

    Consistency is measured by leave-one-out: for each method, drop each
    sample one at a time, re-solve, and measure spread of the resulting
    transforms. Lower spread → more robust.
    """
    results = []
    for name, method_enum in HANDEYE_METHODS.items():
        try:
            r = solve_handeye(
                T_base_tcp_list, T_cam_board_list,
                mode=mode, method=method_enum, method_name=name,
            )
            # Leave-one-out consistency
            rot_err, t_err = _leave_one_out(
                T_base_tcp_list, T_cam_board_list, mode, method_enum
            )
            r.rotation_error_deg = rot_err
            r.translation_error_mm = t_err
            results.append(r)
        except cv2.error as e:
            print(f"  [{name}] FAILED: {e}")

    # Sort by combined error (rotation + normalised translation)
    results.sort(key=lambda r: r.rotation_error_deg + r.translation_error_mm / 10.0)
    return results


def _leave_one_out(
    T_bt: List[np.ndarray],
    T_cb: List[np.ndarray],
    mode: str,
    method: int,
) -> Tuple[float, float]:
    """
    Leave-one-out consistency check.

    Returns (rotation_spread_deg, translation_spread_mm).
    """
    n = len(T_bt)
    if n < 5:
        # Too few samples for meaningful LOO
        return 0.0, 0.0

    Ts = []
    for i in range(n):
        bt_sub = T_bt[:i] + T_bt[i+1:]
        cb_sub = T_cb[:i] + T_cb[i+1:]
        try:
            if mode == "eye_on_base":
                A = [np.linalg.inv(T) for T in bt_sub]
            else:
                A = list(bt_sub)
            R_A, t_A = _decompose(A)
            R_B, t_B = _decompose(cb_sub)
            R_X, t_X = cv2.calibrateHandEye(R_A, t_A, R_B, t_B, method=method)
            T_X = np.eye(4)
            T_X[:3, :3] = R_X
            T_X[:3, 3] = t_X.ravel()
            Ts.append(T_X)
        except cv2.error:
            continue

    if len(Ts) < 3:
        return 99.0, 999.0

    # Translation spread
    translations = np.array([T[:3, 3] for T in Ts])
    t_spread_mm = float(np.std(translations, axis=0).mean()) * 1000.0

    # Rotation spread: angle between each LOO result and the mean rotation
    mean_R = Ts[0][:3, :3]  # use first as reference
    angles = []
    for T in Ts[1:]:
        R_diff = mean_R.T @ T[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        angles.append(np.degrees(angle))
    rot_spread_deg = float(np.mean(angles)) if angles else 0.0

    return rot_spread_deg, t_spread_mm


def pick_best(results: List[HandEyeResult]) -> HandEyeResult:
    """Return the best result (first in the sorted list from solve_all_methods)."""
    if not results:
        raise RuntimeError("No valid hand-eye results")
    return results[0]
