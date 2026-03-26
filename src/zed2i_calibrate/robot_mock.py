"""
Local mock robot bridge — no HERMES dependency required.

Mimics the HERMES RobotBridge / RobotState interface just enough for
the hand-eye collection scripts to run on macOS without installing HERMES.

When HERMES IS installed, scripts use the real MockRobotBridge from HERMES
instead of this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Minimal replica of hermes.core.data.RobotState."""

    timestamp: float
    q: Tuple[float, ...]               # (7,) joint angles [rad]
    dq: Tuple[float, ...]              # (7,) joint velocities
    tau: Tuple[float, ...]             # (7,) torques
    tau_ext: Tuple[float, ...]         # (7,) external torques
    tcp_pose: Tuple[float, ...]        # (7,) [x, y, z, qw, qx, qy, qz]
    tcp_vel: Tuple[float, ...]         # (6,)
    tcp_wrench: Tuple[float, ...]      # (6,)
    fault_state: bool = False
    operational: bool = True
    gripper_width: float = 0.0
    gripper_force: float = 0.0
    gripper_moving: bool = False


class LocalMockRobotBridge:
    """
    Standalone mock robot bridge for macOS development.

    Generates random but plausible tcp_pose values each time read_state()
    is called, simulating a robot that has been moved to different poses.
    """

    def __init__(self) -> None:
        self._connected = False
        self._rng = np.random.default_rng(42)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "LocalMock-Rizon4s"

    def connect(self) -> bool:
        self._connected = True
        print(f"[{self.name}] Connected (local mock)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        print(f"[{self.name}] Disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def read_state(self) -> Optional[RobotState]:
        if not self._connected:
            return None

        self._call_count += 1
        rng = self._rng

        # Generate a plausible tcp_pose
        x = rng.uniform(-0.3, 0.3)
        y = rng.uniform(-0.3, 0.3)
        z = rng.uniform(0.35, 0.65)

        # Random quaternion (normalised)
        quat = rng.normal(size=4)
        quat[0] = abs(quat[0]) + 0.5  # bias toward qw > 0 (small rotations)
        quat /= np.linalg.norm(quat)
        qw, qx, qy, qz = quat

        tcp_pose = (x, y, z, qw, qx, qy, qz)

        # Random joint angles
        q = tuple(rng.uniform(-1.0, 1.0, 7).tolist())

        zeros7 = (0.0,) * 7
        zeros6 = (0.0,) * 6

        return RobotState(
            timestamp=time.time(),
            q=q,
            dq=zeros7,
            tau=zeros7,
            tau_ext=zeros7,
            tcp_pose=tcp_pose,
            tcp_vel=zeros6,
            tcp_wrench=zeros6,
        )
