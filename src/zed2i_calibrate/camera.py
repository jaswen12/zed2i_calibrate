"""
ZED camera abstract interface + factory.

Pattern mirrors HERMES's RobotBridge:
  - ZedCamera (ABC) defines the contract
  - ZedMockCamera  → macOS dev, no ZED SDK needed
  - ZedRealCamera  → Ubuntu + ZED SDK (imported lazily to avoid import error on macOS)

Usage:
    from zed2i_calibrate.camera import open_camera
    from zed2i_calibrate.config import load_config

    cfg = load_config()
    with open_camera(cfg) as cam:
        intrinsics = cam.get_intrinsics()
        left, right = cam.grab()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StereoIntrinsics:
    """Intrinsic parameters for both eyes of a stereo camera."""

    # Left camera
    K_left: np.ndarray    # (3, 3) camera matrix
    D_left: np.ndarray    # (1, 5) distortion coefficients [k1,k2,p1,p2,k3]

    # Right camera
    K_right: np.ndarray   # (3, 3) camera matrix
    D_right: np.ndarray   # (1, 5) distortion coefficients

    # Stereo geometry (right relative to left)
    R: np.ndarray         # (3, 3) rotation matrix
    T: np.ndarray         # (3,)   translation vector [m]

    # Image size (width, height)
    image_size: Tuple[int, int]

    def __repr__(self) -> str:
        fx_l = self.K_left[0, 0]
        fx_r = self.K_right[0, 0]
        baseline_mm = np.linalg.norm(self.T) * 1000
        w, h = self.image_size
        return (
            f"StereoIntrinsics(left_fx={fx_l:.1f}, right_fx={fx_r:.1f}, "
            f"baseline={baseline_mm:.1f}mm, size={w}x{h})"
        )


@dataclass
class StereoFrame:
    """A synchronized stereo image pair."""

    left: np.ndarray    # BGR uint8
    right: np.ndarray   # BGR uint8
    timestamp_ns: int   # nanoseconds (monotonic)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ZedCamera(ABC):
    """Abstract ZED camera interface. Implement for real or mock hardware."""

    @abstractmethod
    def open(self) -> None:
        """Open and initialize the camera."""

    @abstractmethod
    def close(self) -> None:
        """Close the camera and release resources."""

    @abstractmethod
    def grab(self) -> StereoFrame:
        """
        Grab a synchronized stereo frame.

        Returns:
            StereoFrame with left/right BGR images and timestamp.

        Raises:
            RuntimeError: If the camera is not open or grab fails.
        """

    @abstractmethod
    def get_intrinsics(self) -> StereoIntrinsics:
        """
        Return stereo intrinsic parameters.

        For real ZED: reads from SDK (factory or recalibrated).
        For mock: returns plausible synthetic values.
        """

    def __enter__(self) -> "ZedCamera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Mock implementation (macOS / no ZED SDK)
# ---------------------------------------------------------------------------

class ZedMockCamera(ZedCamera):
    """
    Synthetic stereo camera for development on macOS.

    Generates random noise images and returns plausible ZED2i-like intrinsics.
    Allows full pipeline development without hardware.
    """

    # Approximate ZED2i HD1080 factory intrinsics
    _MOCK_FX = 1059.0
    _MOCK_FY = 1059.0
    _MOCK_CX = 960.0
    _MOCK_CY = 540.0
    _MOCK_BASELINE_M = 0.12  # 120 mm

    _RESOLUTION_MAP = {
        "HD2K":  (2208, 1242),
        "HD1080": (1920, 1080),
        "HD720":  (1280, 720),
        "VGA":    (672, 376),
    }

    def __init__(self, resolution: str = "HD1080", fps: int = 15) -> None:
        self.resolution_name = resolution
        self.fps = fps
        self._open = False
        self._frame_count = 0
        self._image_size = self._RESOLUTION_MAP.get(resolution, (1920, 1080))

    def open(self) -> None:
        print(f"[ZedMockCamera] Opened (resolution={self.resolution_name}, fps={self.fps})")
        self._open = True

    def close(self) -> None:
        print("[ZedMockCamera] Closed")
        self._open = False

    def grab(self) -> StereoFrame:
        if not self._open:
            raise RuntimeError("Camera is not open")
        # Simulate frame rate
        time.sleep(1.0 / self.fps)
        w, h = self._image_size
        rng = np.random.default_rng(self._frame_count)
        left = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        right = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        self._frame_count += 1
        return StereoFrame(
            left=left,
            right=right,
            timestamp_ns=time.monotonic_ns(),
        )

    def get_intrinsics(self) -> StereoIntrinsics:
        w, h = self._image_size
        K = np.array([
            [self._MOCK_FX, 0.0, self._MOCK_CX],
            [0.0, self._MOCK_FY, self._MOCK_CY],
            [0.0, 0.0, 1.0],
        ])
        D = np.zeros((1, 5))
        R = np.eye(3)
        T = np.array([-self._MOCK_BASELINE_M, 0.0, 0.0])
        return StereoIntrinsics(
            K_left=K.copy(),
            D_left=D.copy(),
            K_right=K.copy(),
            D_right=D.copy(),
            R=R,
            T=T,
            image_size=(w, h),
        )


# ---------------------------------------------------------------------------
# Real ZED implementation (Ubuntu + ZED SDK only)
# ---------------------------------------------------------------------------

class ZedRealCamera(ZedCamera):
    """
    Real ZED2i camera via pyzed SDK.

    Import is deferred so this module can be imported on macOS without error.
    Only instantiate this on Ubuntu with ZED SDK installed.
    """

    def __init__(self, resolution: str = "HD1080", fps: int = 15,
                 serial_number: Optional[int] = None) -> None:
        try:
            import pyzed.sl as sl  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyzed is not installed or ZED SDK is not found. "
                "ZedRealCamera requires Ubuntu with ZED SDK. "
                "Use ZedMockCamera for development on macOS."
            )
        self._sl = sl
        self.resolution_name = resolution
        self.fps = fps
        self.serial_number = serial_number
        self._zed: Optional[object] = None

    def open(self) -> None:
        sl = self._sl
        init_params = sl.InitParameters()
        init_params.camera_fps = self.fps
        init_params.coordinate_units = sl.UNIT.METER

        res_map = {
            "HD2K":   sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720":  sl.RESOLUTION.HD720,
            "VGA":    sl.RESOLUTION.VGA,
        }
        init_params.camera_resolution = res_map[self.resolution_name]

        if self.serial_number:
            init_params.set_from_serial_number(self.serial_number)

        self._zed = sl.Camera()
        err = self._zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED SDK open failed: {err}")
        print(f"[ZedRealCamera] Opened (resolution={self.resolution_name}, fps={self.fps})")

    def close(self) -> None:
        if self._zed:
            self._zed.close()
            print("[ZedRealCamera] Closed")

    def grab(self) -> StereoFrame:
        sl = self._sl
        runtime = sl.RuntimeParameters()
        err = self._zed.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED grab failed: {err}")

        left_mat = sl.Mat()
        right_mat = sl.Mat()
        self._zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        self._zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

        import cv2
        left_bgr = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_RGBA2BGR)
        right_bgr = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_RGBA2BGR)

        ts = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        return StereoFrame(
            left=left_bgr,
            right=right_bgr,
            timestamp_ns=ts.get_nanoseconds(),
        )

    def get_intrinsics(self) -> StereoIntrinsics:
        sl = self._sl
        cal = self._zed.get_camera_information().camera_configuration.calibration_parameters

        def _k(p) -> np.ndarray:
            return np.array([
                [p.fx, 0.0,  p.cx],
                [0.0,  p.fy, p.cy],
                [0.0,  0.0,  1.0],
            ])

        def _d(p) -> np.ndarray:
            return np.array([[p.disto[i] for i in range(5)]])

        R_stereo = np.array(cal.stereo_transform.r)
        T_stereo = np.array(cal.stereo_transform.t) / 1000.0  # mm → m
        w = self._zed.get_camera_information().camera_configuration.resolution.width
        h = self._zed.get_camera_information().camera_configuration.resolution.height

        return StereoIntrinsics(
            K_left=_k(cal.left_cam),
            D_left=_d(cal.left_cam),
            K_right=_k(cal.right_cam),
            D_right=_d(cal.right_cam),
            R=R_stereo,
            T=T_stereo,
            image_size=(w, h),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def open_camera(cfg: dict) -> ZedCamera:
    """
    Instantiate the correct camera from config.

    Uses ZedMockCamera unless ZED SDK is available and robot_interface != mock.
    Safe to call on macOS — will always return ZedMockCamera there.

    Args:
        cfg: Loaded calibration config dict.

    Returns:
        ZedCamera instance (not yet opened).
    """
    cam_cfg = cfg["camera"]
    resolution = cam_cfg.get("resolution", "HD1080")
    fps = cam_cfg.get("fps", 15)
    serial = cam_cfg.get("serial_number")

    try:
        import pyzed.sl  # noqa: F401
        print("[camera] pyzed found → using ZedRealCamera")
        return ZedRealCamera(resolution=resolution, fps=fps, serial_number=serial)
    except ImportError:
        print("[camera] pyzed not found → using ZedMockCamera (dev mode)")
        return ZedMockCamera(resolution=resolution, fps=fps)
