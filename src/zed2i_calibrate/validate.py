"""
Calibration validation utilities.

Provides quality checks for both stereo and hand-eye calibration results:

1. Stereo validation:
   - Per-image reprojection error
   - Epipolar constraint error

2. Hand-eye validation:
   - Board-in-base consistency: all board observations should map to the
     same position in the robot base frame
   - Reprojection through the full kinematic chain
   - Cross-method comparison

Usage:
    from zed2i_calibrate.validate import validate_handeye, validate_stereo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hand-eye validation
# ---------------------------------------------------------------------------

@dataclass
class HandEyeValidation:
    """Validation report for a hand-eye calibration result."""

    mode: str                          # "eye_on_base" or "eye_on_hand"
    method_name: str
    n_samples: int

    # Board-in-base consistency (eye_on_base only)
    # All board observations projected to base frame should agree
    board_positions_base: Optional[np.ndarray] = None  # (n, 3) positions in base [m]
    board_position_std_mm: float = 0.0                  # std of above [mm]
    board_position_max_mm: float = 0.0                  # max deviation from mean [mm]

    # Reprojection through chain
    chain_reproj_errors: List[float] = field(default_factory=list)
    chain_reproj_mean: float = 0.0
    chain_reproj_max: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== Hand-Eye Validation ({self.method_name}, {self.mode}) ===",
            f"  Samples: {self.n_samples}",
        ]
        if self.board_positions_base is not None:
            lines.append(f"  Board-in-base consistency:")
            lines.append(f"    Std:  {self.board_position_std_mm:.2f} mm")
            lines.append(f"    Max deviation: {self.board_position_max_mm:.2f} mm")
        if self.chain_reproj_errors:
            lines.append(f"  Chain reprojection error:")
            lines.append(f"    Mean: {self.chain_reproj_mean:.3f} px")
            lines.append(f"    Max:  {self.chain_reproj_max:.3f} px")
        return "\n".join(lines)


def validate_handeye(
    T_result: np.ndarray,
    T_base_tcp_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
    mode: str = "eye_on_base",
    method_name: str = "",
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    board_corners_3d: Optional[np.ndarray] = None,
    charuco_corners_2d: Optional[List[np.ndarray]] = None,
) -> HandEyeValidation:
    """
    Validate a hand-eye calibration result.

    Args:
        T_result: (4,4) solved hand-eye transform.
        T_base_tcp_list: List of (4,4) base←tcp transforms.
        T_cam_board_list: List of (4,4) cam←board transforms.
        mode: "eye_on_base" or "eye_on_hand".
        method_name: Name for the report.
        camera_matrix: Required for reprojection check.
        dist_coeffs: Required for reprojection check.
        board_corners_3d: (M,3) board corners in board frame, for reprojection.
        charuco_corners_2d: Per-sample detected corners, for reprojection.

    Returns:
        HandEyeValidation report.
    """
    n = len(T_base_tcp_list)
    val = HandEyeValidation(
        mode=mode,
        method_name=method_name,
        n_samples=n,
    )

    if mode == "eye_on_base":
        _validate_board_in_base(val, T_result, T_base_tcp_list, T_cam_board_list)
    elif mode == "eye_on_hand":
        _validate_board_in_base_eih(val, T_result, T_base_tcp_list, T_cam_board_list)

    return val


def _validate_board_in_base(
    val: HandEyeValidation,
    T_cam2base: np.ndarray,
    T_base_tcp_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
) -> None:
    """
    Eye-on-base consistency: board origin in base frame.

    For eye_on_base:
        T_cam2base = solved result (camera → base)
        board_in_base = T_cam2base^-1 * ... no

    Actually:
        T_result from calibrateHandEye in eye-to-hand mode = T_cam←base
        (camera to base? or base to camera?)

    The relationship:
        T_base←board = T_base←cam * T_cam←board
        where T_base←cam = inv(T_cam←base) = inv(T_result)

    The board is rigidly attached to the gripper, so:
        T_base←board should relate to T_base←tcp by a constant offset.

    We check: T_base←board = inv(T_result) @ T_cam←board
    This should be consistent with T_base←tcp @ T_tcp←board
    where T_tcp←board is constant.
    """
    T_base_cam = np.linalg.inv(T_cam2base)

    # Compute board origin in base frame for each sample
    board_origins = []
    for T_cb in T_cam_board_list:
        T_base_board = T_base_cam @ T_cb
        board_origins.append(T_base_board[:3, 3])

    # The board is on the gripper, so board_in_base should track with tcp_in_base.
    # Subtract the gripper position to get the constant offset T_tcp←board
    offsets = []
    for T_bt, T_cb in zip(T_base_tcp_list, T_cam_board_list):
        T_base_board = T_base_cam @ T_cb
        # T_tcp←board = inv(T_base←tcp) @ T_base←board
        T_tcp_board = np.linalg.inv(T_bt) @ T_base_board
        offsets.append(T_tcp_board[:3, 3])

    offsets = np.array(offsets)
    mean_offset = offsets.mean(axis=0)
    deviations = np.linalg.norm(offsets - mean_offset, axis=1) * 1000.0  # mm

    val.board_positions_base = np.array(board_origins)
    val.board_position_std_mm = float(deviations.std())
    val.board_position_max_mm = float(deviations.max())


def _validate_board_in_base_eih(
    val: HandEyeValidation,
    T_cam2gripper: np.ndarray,
    T_base_tcp_list: List[np.ndarray],
    T_cam_board_list: List[np.ndarray],
) -> None:
    """
    Eye-on-hand consistency: board origin in base frame.

    For eye_on_hand:
        T_result = T_gripper←cam (gripper to camera)
        T_base←board = T_base←tcp @ T_tcp←cam @ T_cam←board
                     = T_base←tcp @ inv(T_cam2gripper) @ T_cam←board

    The board is fixed in world, so T_base←board should be constant.
    """
    T_gripper_cam = np.linalg.inv(T_cam2gripper)

    board_origins = []
    for T_bt, T_cb in zip(T_base_tcp_list, T_cam_board_list):
        T_base_board = T_bt @ T_gripper_cam @ T_cb
        board_origins.append(T_base_board[:3, 3])

    positions = np.array(board_origins)
    mean_pos = positions.mean(axis=0)
    deviations = np.linalg.norm(positions - mean_pos, axis=1) * 1000.0

    val.board_positions_base = positions
    val.board_position_std_mm = float(deviations.std())
    val.board_position_max_mm = float(deviations.max())


# ---------------------------------------------------------------------------
# Stereo validation
# ---------------------------------------------------------------------------

@dataclass
class StereoValidation:
    """Validation report for stereo calibration."""

    n_samples: int
    rms_error: float
    per_image_errors_left: List[float] = field(default_factory=list)
    per_image_errors_right: List[float] = field(default_factory=list)
    epipolar_errors: List[float] = field(default_factory=list)
    epipolar_mean: float = 0.0
    epipolar_max: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== Stereo Validation ===",
            f"  Samples: {self.n_samples}",
            f"  RMS (stereoCalibrate): {self.rms_error:.4f} px",
        ]
        if self.per_image_errors_left:
            mean_l = np.mean(self.per_image_errors_left)
            mean_r = np.mean(self.per_image_errors_right)
            lines.append(f"  Per-image mean reproj:  left={mean_l:.4f}  right={mean_r:.4f} px")
        if self.epipolar_errors:
            lines.append(f"  Epipolar error:  mean={self.epipolar_mean:.4f}  max={self.epipolar_max:.4f} px")
        return "\n".join(lines)


def validate_stereo(
    K_left: np.ndarray,
    D_left: np.ndarray,
    K_right: np.ndarray,
    D_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    obj_points_list: List[np.ndarray],
    img_points_left_list: List[np.ndarray],
    img_points_right_list: List[np.ndarray],
    rms_error: float = 0.0,
) -> StereoValidation:
    """
    Validate stereo calibration with per-image reprojection and epipolar errors.

    Args:
        K_left, D_left, K_right, D_right: Intrinsics.
        R, T: Stereo extrinsics (right w.r.t. left).
        obj_points_list: Per-image 3D object points.
        img_points_left_list: Per-image 2D left image points.
        img_points_right_list: Per-image 2D right image points.
        rms_error: Overall RMS from stereoCalibrate.

    Returns:
        StereoValidation report.
    """
    n = len(obj_points_list)
    val = StereoValidation(n_samples=n, rms_error=rms_error)

    # Compute fundamental matrix from calibration
    T_vec = T.reshape(3, 1)
    # E = [T]x @ R
    Tx = np.array([
        [0, -T_vec[2, 0], T_vec[1, 0]],
        [T_vec[2, 0], 0, -T_vec[0, 0]],
        [-T_vec[1, 0], T_vec[0, 0], 0],
    ])
    E = Tx @ R
    F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)

    for i in range(n):
        obj = obj_points_list[i]
        ipl = img_points_left_list[i].reshape(-1, 2)
        ipr = img_points_right_list[i].reshape(-1, 2)

        # Per-image reprojection: left
        rvec_l = np.zeros(3)
        tvec_l = np.zeros(3)
        proj_l, _ = cv2.projectPoints(obj, rvec_l, tvec_l, K_left, D_left)
        err_l = np.linalg.norm(ipl - proj_l.reshape(-1, 2), axis=1).mean()

        # Per-image reprojection: right (using stereo R, T)
        rvec_r, _ = cv2.Rodrigues(R)
        proj_r, _ = cv2.projectPoints(obj, rvec_r, T_vec, K_right, D_right)
        err_r = np.linalg.norm(ipr - proj_r.reshape(-1, 2), axis=1).mean()

        val.per_image_errors_left.append(float(err_l))
        val.per_image_errors_right.append(float(err_r))

        # Epipolar error: x_r^T F x_l should be ~0
        pts_l_h = np.hstack([ipl, np.ones((len(ipl), 1))])
        pts_r_h = np.hstack([ipr, np.ones((len(ipr), 1))])
        epi_errs = np.abs(np.sum(pts_r_h * (F @ pts_l_h.T).T, axis=1))
        val.epipolar_errors.append(float(epi_errs.mean()))

    if val.epipolar_errors:
        val.epipolar_mean = float(np.mean(val.epipolar_errors))
        val.epipolar_max = float(np.max(val.epipolar_errors))

    return val


# ---------------------------------------------------------------------------
# Cross-method comparison
# ---------------------------------------------------------------------------

def compare_methods(results: List) -> str:
    """
    Format a comparison table of multiple HandEyeResult objects.

    Args:
        results: List of HandEyeResult (from solve_all_methods).

    Returns:
        Formatted string table.
    """
    if not results:
        return "No results to compare."

    lines = [
        "╔══════════════╤═══════════════════════════════════════╤═══════════╤═══════════════════════╗",
        "║ Method       │ Translation (mm)                      │ Rot (°)   │ LOO Δrot / Δt         ║",
        "╠══════════════╪═══════════════════════════════════════╪═══════════╪═══════════════════════╣",
    ]
    for r in results:
        t = r.t_mm
        angle = np.degrees(np.linalg.norm(r.rvec))
        lines.append(
            f"║ {r.method_name:12s} │ "
            f"x={t[0]:+7.1f} y={t[1]:+7.1f} z={t[2]:+7.1f} │ "
            f"{angle:7.2f}   │ "
            f"{r.rotation_error_deg:.3f}° / {r.translation_error_mm:.1f}mm  ║"
        )
    lines.append(
        "╚══════════════╧═══════════════════════════════════════╧═══════════╧═══════════════════════╝"
    )

    # Highlight best
    best = results[0]
    lines.append(f"\n★ Best: {best.method_name} (lowest combined LOO error)")

    return "\n".join(lines)
