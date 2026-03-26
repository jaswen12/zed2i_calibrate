"""
ChArUco board utilities.

Centralizes all board geometry so every script uses identical parameters
loaded from config. No hardcoded values anywhere else.

Usage:
    from zed2i_calibrate.board import CharucoBoard
    from zed2i_calibrate.config import load_config

    cfg = load_config()
    board = CharucoBoard.from_config(cfg)

    # Detect in an image
    result = board.detect(gray_image, camera_matrix, dist_coeffs)
    if result.valid:
        rvec, tvec = result.rvec, result.tvec
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# Map config string names to OpenCV ArUco dict constants
_ARUCO_DICT_MAP: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
}


@dataclass
class DetectionResult:
    """Result of a single ChArUco detection attempt."""

    valid: bool                          # True if enough corners detected for pose
    charuco_corners: Optional[np.ndarray] = None  # (N, 1, 2) float32
    charuco_ids: Optional[np.ndarray] = None      # (N, 1) int32
    rvec: Optional[np.ndarray] = None   # (3,) rotation vector (camera←board)
    tvec: Optional[np.ndarray] = None   # (3,) translation vector (camera←board)
    n_corners: int = 0
    reprojection_error: Optional[float] = None

    @property
    def T_cam_board(self) -> Optional[np.ndarray]:
        """4x4 homogeneous transform: camera <- board frame."""
        if self.rvec is None or self.tvec is None:
            return None
        R, _ = cv2.Rodrigues(self.rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.tvec.ravel()
        return T


@dataclass
class CharucoBoard:
    """
    Wrapper around cv2.aruco.CharucoBoard with helpers for detection and
    image generation.

    All lengths in millimetres internally; OpenCV receives metres where needed.
    """

    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    aruco_dict_name: str
    _board: cv2.aruco.CharucoBoard = field(init=False, repr=False)
    _detector: cv2.aruco.CharucoDetector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.aruco_dict_name not in _ARUCO_DICT_MAP:
            raise ValueError(
                f"Unknown ArUco dictionary: {self.aruco_dict_name!r}. "
                f"Valid options: {list(_ARUCO_DICT_MAP)}"
            )
        aruco_dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_MAP[self.aruco_dict_name])
        self._board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            aruco_dict,
        )
        params = cv2.aruco.DetectorParameters()
        charuco_params = cv2.aruco.CharucoParameters()
        self._detector = cv2.aruco.CharucoDetector(self._board, charuco_params, params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def square_length_m(self) -> float:
        return self.square_length_mm / 1000.0

    @property
    def marker_length_m(self) -> float:
        return self.marker_length_mm / 1000.0

    @property
    def n_corners(self) -> int:
        """Total inner corner count."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def opencv_board(self) -> cv2.aruco.CharucoBoard:
        return self._board

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "CharucoBoard":
        """Construct from the 'board' section of calibration.yaml."""
        b = cfg["board"]
        return cls(
            squares_x=b["squares_x"],
            squares_y=b["squares_y"],
            square_length_mm=float(b["square_length_mm"]),
            marker_length_mm=float(b["marker_length_mm"]),
            aruco_dict_name=b["aruco_dict"],
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        gray: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        min_corners: int = 6,
    ) -> DetectionResult:
        """
        Detect ChArUco board in a grayscale image.

        Args:
            gray: Grayscale image (uint8).
            camera_matrix: 3x3 intrinsic matrix. Required for pose estimation.
            dist_coeffs: Distortion coefficients. Required for pose estimation.
            min_corners: Minimum corners needed to attempt pose estimation.

        Returns:
            DetectionResult with corners, IDs, and optional pose.
        """
        charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(gray)

        if charuco_corners is None or len(charuco_corners) < min_corners:
            return DetectionResult(valid=False)

        n = len(charuco_corners)

        # Pose estimation requires intrinsics
        if camera_matrix is None or dist_coeffs is None:
            return DetectionResult(
                valid=True,
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                n_corners=n,
            )

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self._board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )

        if not ok:
            return DetectionResult(
                valid=False,
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                n_corners=n,
            )

        # Compute reprojection error
        reproj_err = self._reprojection_error(
            charuco_corners, charuco_ids, rvec, tvec, camera_matrix, dist_coeffs
        )

        return DetectionResult(
            valid=True,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            rvec=rvec.ravel(),
            tvec=tvec.ravel(),
            n_corners=n,
            reprojection_error=reproj_err,
        )

    def _reprojection_error(
        self,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> float:
        """Mean reprojection error in pixels."""
        obj_pts = self._board.getChessboardCorners()[charuco_ids.ravel()]
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        err = np.linalg.norm(charuco_corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        return float(err.mean())

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def draw_image(self, pixels_per_square: int = 100) -> np.ndarray:
        """
        Generate a printable board image.

        Args:
            pixels_per_square: Resolution per square in pixels.

        Returns:
            BGR image of the board with a white border.
        """
        w = self.squares_x * pixels_per_square
        h = self.squares_y * pixels_per_square
        img = self._board.generateImage((w, h), marginSize=pixels_per_square // 2)
        return img

    def draw_detected(
        self,
        image: np.ndarray,
        result: DetectionResult,
        draw_axes: bool = True,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        axis_length_m: float = 0.05,
    ) -> np.ndarray:
        """Overlay detection results on a BGR image (copy)."""
        vis = image.copy()
        if result.charuco_corners is not None and result.charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                vis, result.charuco_corners, result.charuco_ids
            )
        if (
            draw_axes
            and result.rvec is not None
            and result.tvec is not None
            and camera_matrix is not None
            and dist_coeffs is not None
        ):
            cv2.drawFrameAxes(
                vis,
                camera_matrix,
                dist_coeffs,
                result.rvec,
                result.tvec,
                axis_length_m,
            )
        return vis

    def __repr__(self) -> str:
        return (
            f"CharucoBoard({self.squares_x}x{self.squares_y}, "
            f"square={self.square_length_mm}mm, "
            f"marker={self.marker_length_mm}mm, "
            f"dict={self.aruco_dict_name})"
        )
