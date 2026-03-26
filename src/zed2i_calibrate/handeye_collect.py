"""
Hand-eye calibration sample collector.

Collects synchronized pairs of:
  - Camera observation: ChArUco board pose (T_cam←board) from image
  - Robot pose: TCP pose (T_base←tcp) from HERMES bridge

Supports two configurations:
  eye_on_base:  Camera fixed externally, robot moves the board
                AX = XB where A = T_base←tcp, B = T_cam←board
  eye_on_hand:  Camera mounted on flange, board fixed
                AX = XB where A = T_tcp←base, B = T_cam←board

Usage:
    from zed2i_calibrate.handeye_collect import HandEyeCollector

    collector = HandEyeCollector(board, cfg)
    collector.add_sample(gray_image, robot_state)
    collector.save("data/eye_on_base")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from zed2i_calibrate.board import CharucoBoard, DetectionResult


@dataclass
class HandEyeSample:
    """One valid hand-eye sample: board pose + robot pose."""

    # Camera observation
    T_cam_board: np.ndarray      # (4, 4) camera ← board
    rvec_board: np.ndarray       # (3,) Rodrigues
    tvec_board: np.ndarray       # (3,)
    n_corners: int
    reprojection_error: float

    # Robot state
    tcp_pose: np.ndarray         # (7,) x, y, z, qw, qx, qy, qz
    q: np.ndarray                # (7,) joint angles [rad]
    T_base_tcp: np.ndarray       # (4, 4) base ← tcp

    # Metadata
    index: int
    timestamp: float


def tcp_pose_to_T(tcp_pose: np.ndarray) -> np.ndarray:
    """
    Convert HERMES tcp_pose (x, y, z, qw, qx, qy, qz) to 4x4 homogeneous matrix.

    HERMES uses quaternion convention: (qw, qx, qy, qz) — scalar-first.
    """
    x, y, z = tcp_pose[0], tcp_pose[1], tcp_pose[2]
    qw, qx, qy, qz = tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6]

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),      1 - 2*(qx**2 + qy**2)],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


class HandEyeCollector:
    """
    Collects and stores hand-eye calibration samples.

    Each sample pairs a ChArUco board detection (T_cam←board) with
    a robot TCP pose (T_base←tcp) captured at the same moment.
    """

    def __init__(
        self,
        board: CharucoBoard,
        cfg: dict,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """
        Args:
            board: CharucoBoard instance for detection.
            cfg: Full config dict (reads handeye section).
            camera_matrix: 3x3 intrinsic matrix (from stereo calibration or factory).
            dist_coeffs: Distortion coefficients.
        """
        self.board = board
        self.cfg = cfg
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        he = cfg.get("handeye", {})
        self.min_samples = he.get("min_samples", 15)
        self.max_samples = he.get("max_samples", 50)
        self.mode = he.get("mode", "eye_on_base")

        self._samples: List[HandEyeSample] = []
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
        gray: np.ndarray,
        tcp_pose: np.ndarray,
        q: Optional[np.ndarray] = None,
        min_corners: int = 6,
        max_reproj_error: float = 2.0,
    ) -> Optional[HandEyeSample]:
        """
        Attempt to add a hand-eye sample.

        Args:
            gray: Grayscale image (uint8).
            tcp_pose: (7,) array [x, y, z, qw, qx, qy, qz] from HERMES.
            q: (7,) joint angles [rad], optional (stored for traceability).
            min_corners: Minimum corners for valid detection.
            max_reproj_error: Max reprojection error in pixels to accept sample.

        Returns:
            HandEyeSample if successful, None otherwise.
        """
        if self.is_full:
            return None

        # Detect board
        result = self.board.detect(
            gray, self.camera_matrix, self.dist_coeffs, min_corners=min_corners
        )

        if not result.valid or result.rvec is None:
            return None

        if result.reprojection_error is not None and result.reprojection_error > max_reproj_error:
            return None

        # Convert robot pose
        tcp_arr = np.asarray(tcp_pose, dtype=np.float64)
        T_base_tcp = tcp_pose_to_T(tcp_arr)
        q_arr = np.asarray(q, dtype=np.float64) if q is not None else np.zeros(7)

        sample = HandEyeSample(
            T_cam_board=result.T_cam_board,
            rvec_board=result.rvec,
            tvec_board=result.tvec,
            n_corners=result.n_corners,
            reprojection_error=result.reprojection_error if result.reprojection_error is not None else 0.0,
            tcp_pose=tcp_arr,
            q=q_arr,
            T_base_tcp=T_base_tcp,
            index=self._sample_counter,
            timestamp=time.time(),
        )
        self._samples.append(sample)
        self._sample_counter += 1
        return sample

    def get_samples(self) -> List[HandEyeSample]:
        return list(self._samples)

    def remove_sample(self, index: int) -> bool:
        """Remove sample by its original index."""
        for i, s in enumerate(self._samples):
            if s.index == index:
                self._samples.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> Path:
        """
        Save all samples to disk.

        Creates:
            output_dir/
              samples.json         Metadata + poses (human-readable)
              T_cam_board/         Per-sample 4x4 .npy
              T_base_tcp/          Per-sample 4x4 .npy
              images/              Grayscale images (if save_images called)

        Args:
            output_dir: Directory to save into.

        Returns:
            Path to samples.json.
        """
        out = Path(output_dir)
        (out / "T_cam_board").mkdir(parents=True, exist_ok=True)
        (out / "T_base_tcp").mkdir(parents=True, exist_ok=True)

        records = []
        for s in self._samples:
            idx = s.index

            np.save(out / "T_cam_board" / f"{idx:04d}.npy", s.T_cam_board)
            np.save(out / "T_base_tcp" / f"{idx:04d}.npy", s.T_base_tcp)

            records.append({
                "index": idx,
                "timestamp": s.timestamp,
                "n_corners": s.n_corners,
                "reprojection_error": s.reprojection_error,
                "tcp_pose": s.tcp_pose.tolist(),
                "q": s.q.tolist(),
                "rvec_board": s.rvec_board.tolist(),
                "tvec_board": s.tvec_board.tolist(),
            })

        meta = {
            "mode": self.mode,
            "n_samples": len(records),
            "board": {
                "squares_x": self.board.squares_x,
                "squares_y": self.board.squares_y,
                "square_length_mm": self.board.square_length_mm,
                "marker_length_mm": self.board.marker_length_mm,
                "aruco_dict": self.board.aruco_dict_name,
            },
            "samples": records,
        }

        json_path = out / "samples.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[handeye] Saved {len(records)} samples → {out}")
        return json_path

    @classmethod
    def load_samples(cls, samples_dir: str | Path) -> Tuple[List[np.ndarray], List[np.ndarray], dict]:
        """
        Load saved samples for hand-eye solving.

        Returns:
            Tuple of (T_base_tcp_list, T_cam_board_list, metadata):
              - T_base_tcp_list: List of (4,4) base←tcp transforms
              - T_cam_board_list: List of (4,4) cam←board transforms
              - metadata: dict from samples.json
        """
        d = Path(samples_dir)
        json_path = d / "samples.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No samples.json in {d}")

        with open(json_path) as f:
            meta = json.load(f)

        T_base_tcp_list = []
        T_cam_board_list = []

        for rec in meta["samples"]:
            idx = rec["index"]
            T_bt = np.load(d / "T_base_tcp" / f"{idx:04d}.npy")
            T_cb = np.load(d / "T_cam_board" / f"{idx:04d}.npy")
            T_base_tcp_list.append(T_bt)
            T_cam_board_list.append(T_cb)

        return T_base_tcp_list, T_cam_board_list, meta
