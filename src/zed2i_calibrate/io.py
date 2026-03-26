"""
OpenCV YAML / NumPy I/O utilities.

Centralizes all calibration result read/write so every script uses the same
format. Two backends:
  - OpenCV FileStorage (standard .yaml readable by cv2.FileStorage in C++)
  - NumPy .npy (for quick Python-only reload of 4x4 matrices)

Usage:
    from zed2i_calibrate.io import save_intrinsics, load_intrinsics
    from zed2i_calibrate.io import save_transform, load_transform
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from zed2i_calibrate.camera import StereoIntrinsics


# ---------------------------------------------------------------------------
# Stereo intrinsics I/O
# ---------------------------------------------------------------------------

def save_intrinsics(path: str | Path, intrinsics: StereoIntrinsics) -> Path:
    """
    Save stereo intrinsics to OpenCV YAML format.

    File structure:
        image_width, image_height
        K_left, D_left
        K_right, D_right
        R_stereo, T_stereo
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    w, h = intrinsics.image_size
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("K_left", intrinsics.K_left)
    fs.write("D_left", intrinsics.D_left)
    fs.write("K_right", intrinsics.K_right)
    fs.write("D_right", intrinsics.D_right)
    fs.write("R_stereo", intrinsics.R)
    fs.write("T_stereo", intrinsics.T.reshape(3, 1))
    fs.release()

    print(f"[io] Saved intrinsics → {path}")
    return path


def load_intrinsics(path: str | Path) -> StereoIntrinsics:
    """Load stereo intrinsics from OpenCV YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    K_left = fs.getNode("K_left").mat()
    D_left = fs.getNode("D_left").mat()
    K_right = fs.getNode("K_right").mat()
    D_right = fs.getNode("D_right").mat()
    R = fs.getNode("R_stereo").mat()
    T = fs.getNode("T_stereo").mat().ravel()
    fs.release()

    return StereoIntrinsics(
        K_left=K_left, D_left=D_left,
        K_right=K_right, D_right=D_right,
        R=R, T=T,
        image_size=(w, h),
    )


# ---------------------------------------------------------------------------
# 4x4 homogeneous transform I/O
# ---------------------------------------------------------------------------

def save_transform(
    path: str | Path,
    T: np.ndarray,
    label: str = "T",
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save a 4x4 transform to OpenCV YAML.

    Also saves an .npy sidecar for quick numpy loading.

    Args:
        path: Output .yaml path.
        T: (4, 4) homogeneous transform.
        label: Name used as the key in the YAML file.
        metadata: Optional dict of scalar metadata (reprojection_error, method, etc.)
    """
    if T.shape != (4, 4):
        raise ValueError(f"Expected (4,4) transform, got {T.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write(label, T)

    # Also decompose for readability
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    fs.write(f"{label}_rvec", rvec)
    fs.write(f"{label}_tvec", t.reshape(3, 1))

    if metadata:
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                fs.write(k, v)
            elif isinstance(v, str):
                fs.write(k, v)
    fs.release()

    # .npy sidecar
    npy_path = path.with_suffix(".npy")
    np.save(npy_path, T)

    print(f"[io] Saved transform → {path} (+ {npy_path.name})")
    return path


def load_transform(path: str | Path, label: str = "T") -> np.ndarray:
    """
    Load a 4x4 transform from OpenCV YAML.

    Args:
        path: .yaml path.
        label: Key name used when saving.

    Returns:
        (4, 4) numpy array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    T = fs.getNode(label).mat()
    fs.release()

    if T is None or T.shape != (4, 4):
        raise ValueError(f"Expected (4,4) matrix under key '{label}', got {T}")
    return T


# ---------------------------------------------------------------------------
# Stereo calibration results I/O
# ---------------------------------------------------------------------------

def save_stereo_calibration(
    path: str | Path,
    K_left: np.ndarray,
    D_left: np.ndarray,
    K_right: np.ndarray,
    D_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size: Tuple[int, int],
    rms_error: float,
    n_samples: int,
) -> Path:
    """
    Save full stereo calibration results (intrinsics + stereo extrinsics).

    This is the output of cv2.stereoCalibrate — a superset of save_intrinsics
    that also includes calibration quality metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    w, h = image_size
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("K_left", K_left)
    fs.write("D_left", D_left)
    fs.write("K_right", K_right)
    fs.write("D_right", D_right)
    fs.write("R_stereo", R)
    fs.write("T_stereo", T.reshape(3, 1))
    fs.write("rms_error", rms_error)
    fs.write("n_samples", n_samples)
    fs.release()

    print(f"[io] Saved stereo calibration → {path} (RMS={rms_error:.4f}, n={n_samples})")
    return path


def load_stereo_calibration(path: str | Path) -> dict:
    """
    Load stereo calibration results.

    Returns dict with keys:
        K_left, D_left, K_right, D_right, R, T, image_size, rms_error, n_samples
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stereo calibration file not found: {path}")

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    result = {
        "K_left": fs.getNode("K_left").mat(),
        "D_left": fs.getNode("D_left").mat(),
        "K_right": fs.getNode("K_right").mat(),
        "D_right": fs.getNode("D_right").mat(),
        "R": fs.getNode("R_stereo").mat(),
        "T": fs.getNode("T_stereo").mat().ravel(),
        "image_size": (
            int(fs.getNode("image_width").real()),
            int(fs.getNode("image_height").real()),
        ),
        "rms_error": fs.getNode("rms_error").real(),
        "n_samples": int(fs.getNode("n_samples").real()),
    }
    fs.release()
    return result
