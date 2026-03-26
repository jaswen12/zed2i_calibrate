"""
Stereo camera calibration core logic.

Pipeline:
  1. Collect stereo image pairs (left + right) with ChArUco board visible in both
  2. Detect ChArUco corners in each image independently
  3. Run cv2.calibrateCamera on left and right separately (single-camera intrinsics)
  4. Run cv2.stereoCalibrate with the single-camera results as initial guess
  5. Output: K_left, D_left, K_right, D_right, R, T, RMS error

Usage:
    from zed2i_calibrate.stereo_calibrate import StereoCalibrator
    from zed2i_calibrate.board import CharucoBoard
    from zed2i_calibrate.config import load_config

    cfg = load_config()
    board = CharucoBoard.from_config(cfg)
    calibrator = StereoCalibrator(board, cfg)

    # Add samples
    for left_gray, right_gray in image_pairs:
        calibrator.add_sample(left_gray, right_gray)

    # Calibrate
    result = calibrator.calibrate()
    print(f"RMS: {result.rms_error:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from zed2i_calibrate.board import CharucoBoard


@dataclass
class StereoSample:
    """One valid stereo sample: matched ChArUco corners in both views."""

    obj_pts: np.ndarray         # (N, 3) world points
    img_pts_left: np.ndarray    # (N, 1, 2) image points, left
    img_pts_right: np.ndarray   # (N, 1, 2) image points, right
    n_corners: int
    index: int                  # sample index for traceability


@dataclass
class StereoCalibrationResult:
    """Output of stereo calibration."""

    # Per-camera intrinsics
    K_left: np.ndarray       # (3, 3)
    D_left: np.ndarray       # (1, 5)
    K_right: np.ndarray      # (3, 3)
    D_right: np.ndarray      # (1, 5)

    # Stereo extrinsics (right camera relative to left)
    R: np.ndarray            # (3, 3) rotation
    T: np.ndarray            # (3, 1) translation

    # Quality
    rms_error: float         # RMS reprojection error from stereoCalibrate
    n_samples: int           # number of image pairs used
    image_size: Tuple[int, int]  # (width, height)

    # Per-camera RMS from individual calibration
    rms_left: float = 0.0
    rms_right: float = 0.0

    def summary(self) -> str:
        baseline_mm = np.linalg.norm(self.T) * 1000
        return (
            f"Stereo Calibration Result:\n"
            f"  Samples used:    {self.n_samples}\n"
            f"  Image size:      {self.image_size[0]}x{self.image_size[1]}\n"
            f"  Left  fx/fy:     {self.K_left[0,0]:.1f} / {self.K_left[1,1]:.1f}\n"
            f"  Right fx/fy:     {self.K_right[0,0]:.1f} / {self.K_right[1,1]:.1f}\n"
            f"  Baseline:        {baseline_mm:.1f} mm\n"
            f"  RMS (left):      {self.rms_left:.4f} px\n"
            f"  RMS (right):     {self.rms_right:.4f} px\n"
            f"  RMS (stereo):    {self.rms_error:.4f} px\n"
        )


class StereoCalibrator:
    """
    Collects stereo ChArUco samples and runs OpenCV stereo calibration.
    """

    def __init__(
        self,
        board: CharucoBoard,
        cfg: dict,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            board: CharucoBoard instance.
            cfg: Full config dict (reads stereo_calibration section).
            image_size: (width, height). If None, inferred from first sample.
        """
        self.board = board
        self.cfg = cfg
        self.image_size = image_size

        sc = cfg.get("stereo_calibration", {})
        self.min_samples = sc.get("min_samples", 30)
        self.max_samples = sc.get("max_samples", 80)
        self.use_factory_guess = sc.get("use_factory_intrinsics_as_guess", True)
        self.fix_intrinsics = sc.get("fix_intrinsics", False)

        self._samples: List[StereoSample] = []
        self._sample_counter = 0

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def is_ready(self) -> bool:
        return self.n_samples >= self.min_samples

    @property
    def is_full(self) -> bool:
        return self.n_samples >= self.max_samples

    def add_sample(
        self,
        gray_left: np.ndarray,
        gray_right: np.ndarray,
        min_corners: int = 6,
    ) -> Optional[StereoSample]:
        """
        Attempt to add a stereo sample.

        Detects ChArUco in both images. Only adds the sample if both views
        have enough corners AND the detected corner IDs overlap.

        Args:
            gray_left: Grayscale left image (uint8).
            gray_right: Grayscale right image (uint8).
            min_corners: Minimum corners in each view.

        Returns:
            StereoSample if successful, None if detection failed.
        """
        if self.is_full:
            return None

        # Infer image size from first sample
        if self.image_size is None:
            h, w = gray_left.shape[:2]
            self.image_size = (w, h)

        # Detect in both views (no pose estimation needed here)
        result_l = self.board.detect(gray_left)
        result_r = self.board.detect(gray_right)

        if not result_l.valid or not result_r.valid:
            return None
        if result_l.n_corners < min_corners or result_r.n_corners < min_corners:
            return None

        # Find common corner IDs between left and right
        ids_l = result_l.charuco_ids.ravel()
        ids_r = result_r.charuco_ids.ravel()
        common_ids = np.intersect1d(ids_l, ids_r)

        if len(common_ids) < min_corners:
            return None

        # Extract matched points in the same order
        mask_l = np.isin(ids_l, common_ids)
        mask_r = np.isin(ids_r, common_ids)

        # Sort by ID so left and right correspond
        order_l = np.argsort(ids_l[mask_l])
        order_r = np.argsort(ids_r[mask_r])

        pts_l = result_l.charuco_corners[mask_l][order_l]
        pts_r = result_r.charuco_corners[mask_r][order_r]
        matched_ids = ids_l[mask_l][order_l]

        # Get 3D object points for matched IDs
        all_board_corners = self.board.opencv_board.getChessboardCorners()
        obj_pts = all_board_corners[matched_ids].astype(np.float32)

        sample = StereoSample(
            obj_pts=obj_pts,
            img_pts_left=pts_l.reshape(-1, 1, 2).astype(np.float32),
            img_pts_right=pts_r.reshape(-1, 1, 2).astype(np.float32),
            n_corners=len(common_ids),
            index=self._sample_counter,
        )
        self._samples.append(sample)
        self._sample_counter += 1
        return sample

    def calibrate(
        self,
        K_left_init: Optional[np.ndarray] = None,
        D_left_init: Optional[np.ndarray] = None,
        K_right_init: Optional[np.ndarray] = None,
        D_right_init: Optional[np.ndarray] = None,
    ) -> StereoCalibrationResult:
        """
        Run stereo calibration on collected samples.

        Args:
            K_left_init: Initial left camera matrix (factory intrinsics).
            D_left_init: Initial left distortion.
            K_right_init: Initial right camera matrix.
            D_right_init: Initial right distortion.

        Returns:
            StereoCalibrationResult with all calibrated parameters.

        Raises:
            RuntimeError: If not enough samples collected.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Need at least {self.min_samples} samples, "
                f"have {self.n_samples}"
            )

        w, h = self.image_size

        # Prepare per-sample point arrays
        obj_points = [s.obj_pts for s in self._samples]
        img_points_l = [s.img_pts_left for s in self._samples]
        img_points_r = [s.img_pts_right for s in self._samples]

        # -----------------------------------------------------------
        # Step 1: Single-camera calibration (left and right separately)
        # -----------------------------------------------------------
        flags_single = 0
        if K_left_init is not None and self.use_factory_guess:
            flags_single |= cv2.CALIB_USE_INTRINSIC_GUESS

        rms_l, K_l, D_l, _, _ = cv2.calibrateCamera(
            obj_points, img_points_l, (w, h),
            cameraMatrix=K_left_init.copy() if K_left_init is not None else None,
            distCoeffs=D_left_init.copy() if D_left_init is not None else None,
            flags=flags_single,
        )

        flags_single_r = 0
        if K_right_init is not None and self.use_factory_guess:
            flags_single_r |= cv2.CALIB_USE_INTRINSIC_GUESS

        rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(
            obj_points, img_points_r, (w, h),
            cameraMatrix=K_right_init.copy() if K_right_init is not None else None,
            distCoeffs=D_right_init.copy() if D_right_init is not None else None,
            flags=flags_single_r,
        )

        print(f"[stereo] Single-camera RMS:  left={rms_l:.4f}  right={rms_r:.4f}")

        # -----------------------------------------------------------
        # Step 2: Stereo calibration
        # -----------------------------------------------------------
        flags_stereo = 0
        if self.fix_intrinsics:
            flags_stereo |= cv2.CALIB_FIX_INTRINSIC
        else:
            flags_stereo |= cv2.CALIB_USE_INTRINSIC_GUESS

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        )

        rms_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_l,
            img_points_r,
            K_l, D_l, K_r, D_r,
            (w, h),
            flags=flags_stereo,
            criteria=criteria,
        )

        print(f"[stereo] Stereo RMS: {rms_stereo:.4f}")

        return StereoCalibrationResult(
            K_left=K_l,
            D_left=D_l,
            K_right=K_r,
            D_right=D_r,
            R=R,
            T=T,
            rms_error=rms_stereo,
            n_samples=self.n_samples,
            image_size=(w, h),
            rms_left=rms_l,
            rms_right=rms_r,
        )

    def save_sample_images(
        self,
        output_dir: str | Path,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
    ) -> None:
        """
        Save collected image pairs to disk for later offline processing.

        Args:
            output_dir: Directory to save images.
            left_images: List of left BGR images (same length as samples).
            right_images: List of right BGR images.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for i, (left, right) in enumerate(zip(left_images, right_images)):
            cv2.imwrite(str(out / f"left_{i:04d}.png"), left)
            cv2.imwrite(str(out / f"right_{i:04d}.png"), right)

        print(f"[stereo] Saved {len(left_images)} image pairs → {out}")
