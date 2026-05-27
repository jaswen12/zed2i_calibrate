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
# UVC implementation (any OS, no ZED SDK / NVIDIA required)
# ---------------------------------------------------------------------------

class ZedUVCCamera(ZedCamera):
    """
    ZED2i via UVC (USB Video Class) using cv2.VideoCapture.

    The ZED2i exposes itself as a standard UVC device that streams a
    side-by-side stereo image (left half = left eye, right half = right eye).
    No ZED SDK or NVIDIA GPU is required — works on Ubuntu without GPU,
    macOS, Windows, etc.

    Trade-offs vs. ZedRealCamera:
      LOSE: factory EEPROM intrinsics (we return approximate defaults instead),
            IMU, hardware depth, hardware rectification.
      KEEP: synchronized stereo image pairs, which is everything stereo
            calibration and ChArUco-based hand-eye need.

    The returned intrinsics are *placeholders* sized to the chosen resolution.
    They should only be used as an initial guess for stereoCalibrate
    (script 02). The real values come out of that calibration step.
    """

    # Side-by-side UVC modes (combined_width, height, max_fps)
    _RESOLUTION_MAP: dict[str, Tuple[int, int, int]] = {
        "HD2K":   (4416, 1242, 15),
        "HD1080": (3840, 1080, 30),
        "HD720":  (2560, 720, 60),
        "VGA":    (1344, 376, 100),
    }

    # Rough ZED2i factory values (per single eye) for each resolution.
    # Real calibration will refine these; we just need a reasonable initial
    # guess so cv2.calibrateCamera doesn't diverge.
    _FALLBACK_INTRINSICS: dict[str, dict] = {
        "HD2K":   {"fx": 1400.0, "fy": 1400.0, "cx": 1104.0, "cy": 621.0},
        "HD1080": {"fx": 1059.0, "fy": 1059.0, "cx": 960.0,  "cy": 540.0},
        "HD720":  {"fx": 706.0,  "fy": 706.0,  "cx": 640.0,  "cy": 360.0},
        "VGA":    {"fx": 350.0,  "fy": 350.0,  "cx": 336.0,  "cy": 188.0},
    }

    _BASELINE_M = 0.12  # ZED2i nominal stereo baseline (120 mm)

    def __init__(
        self,
        resolution: str = "HD1080",
        fps: int = 15,
        device_index: int = 0,
    ) -> None:
        self.resolution_name = resolution
        self.fps = fps
        self.device_index = device_index
        self._cap: Optional[object] = None
        self._image_size: Tuple[int, int] = (0, 0)  # single eye (w, h)
        self._combined_size: Tuple[int, int] = (0, 0)

    def open(self) -> None:
        import cv2

        if self.resolution_name not in self._RESOLUTION_MAP:
            raise ValueError(
                f"Unknown resolution: {self.resolution_name!r}. "
                f"Valid: {list(self._RESOLUTION_MAP)}"
            )
        w_total, h, max_fps = self._RESOLUTION_MAP[self.resolution_name]

        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open UVC device at index {self.device_index}. "
                f"Check `v4l2-ctl --list-devices` (Linux) or System Settings → "
                f"Privacy → Camera (macOS). Try a different index."
            )

        # MJPG is required for high-resolution stereo modes — YUYV is
        # bandwidth-limited to ~1080p side-by-side on USB 3.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_total)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, min(self.fps, max_fps))

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        if (actual_w, actual_h) != (w_total, h):
            print(
                f"[ZedUVCCamera] WARNING: requested {w_total}x{h}, "
                f"got {actual_w}x{actual_h}. Resolution name may be wrong, "
                f"or the camera is not a ZED2i."
            )

        # Single-eye image size
        self._combined_size = (actual_w, actual_h)
        self._image_size = (actual_w // 2, actual_h)
        self._cap = cap

        print(
            f"[ZedUVCCamera] Opened device {self.device_index}: "
            f"{actual_w}x{actual_h} @ {actual_fps:.0f}fps "
            f"(single eye: {self._image_size[0]}x{self._image_size[1]})"
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[ZedUVCCamera] Closed")

    def grab(self) -> StereoFrame:
        if self._cap is None:
            raise RuntimeError("Camera is not open")

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("UVC grab failed — camera disconnected?")

        # Split side-by-side. ZED stereo convention: left half = LEFT eye.
        w_total = frame.shape[1]
        mid = w_total // 2
        left = frame[:, :mid].copy()
        right = frame[:, mid:].copy()

        return StereoFrame(
            left=left,
            right=right,
            timestamp_ns=time.monotonic_ns(),
        )

    def get_intrinsics(self) -> StereoIntrinsics:
        """
        Return *placeholder* intrinsics scaled to the current resolution.

        These are approximate ZED2i factory values — not the unique calibration
        for the specific physical unit. Use them only as an initial guess for
        stereo calibration (script 02), then trust the calibrated values.
        """
        w, h = self._image_size
        params = self._FALLBACK_INTRINSICS.get(
            self.resolution_name,
            self._FALLBACK_INTRINSICS["HD1080"],
        )
        K = np.array([
            [params["fx"], 0.0,          params["cx"]],
            [0.0,          params["fy"], params["cy"]],
            [0.0,          0.0,          1.0],
        ])
        D = np.zeros((1, 5))
        R = np.eye(3)
        T = np.array([-self._BASELINE_M, 0.0, 0.0])
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

    Backend selection (config key ``camera.backend``):
        "auto"  (default): try pyzed → ZedRealCamera, else fall back to
                ZedMockCamera. Recommended for dev machines.
        "real"  / "sdk":   force ZedRealCamera (requires NVIDIA + ZED SDK).
        "uvc":             force ZedUVCCamera (raw USB capture, no GPU).
                Required for non-NVIDIA Ubuntu / macOS with a physical camera.
        "mock":            force ZedMockCamera (synthetic data for dev).

    Args:
        cfg: Loaded calibration config dict.

    Returns:
        ZedCamera instance (not yet opened).
    """
    cam_cfg = cfg["camera"]
    backend = str(cam_cfg.get("backend", "auto")).lower()
    resolution = cam_cfg.get("resolution", "HD1080")
    fps = cam_cfg.get("fps", 15)
    serial = cam_cfg.get("serial_number")
    uvc_index = int(cam_cfg.get("uvc_device_index", 0))

    if backend in ("real", "sdk"):
        print("[camera] backend=real → using ZedRealCamera (requires pyzed)")
        return ZedRealCamera(resolution=resolution, fps=fps, serial_number=serial)

    if backend == "uvc":
        print(f"[camera] backend=uvc → using ZedUVCCamera (device {uvc_index})")
        return ZedUVCCamera(resolution=resolution, fps=fps, device_index=uvc_index)

    if backend == "mock":
        print("[camera] backend=mock → using ZedMockCamera (synthetic data)")
        return ZedMockCamera(resolution=resolution, fps=fps)

    if backend != "auto":
        print(
            f"[camera] WARN: unknown backend {backend!r}, falling back to 'auto'"
        )

    try:
        import pyzed.sl  # noqa: F401
        print("[camera] backend=auto: pyzed found → using ZedRealCamera")
        return ZedRealCamera(resolution=resolution, fps=fps, serial_number=serial)
    except ImportError:
        print(
            "[camera] backend=auto: pyzed not found → using ZedMockCamera. "
            "For real capture without ZED SDK, set camera.backend: 'uvc' in config."
        )
        return ZedMockCamera(resolution=resolution, fps=fps)
