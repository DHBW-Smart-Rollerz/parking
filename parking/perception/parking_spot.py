from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from camera_preprocessing.transformation.coordinate_transform import Unit

if TYPE_CHECKING:
    import geometry_msgs.msg
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )


@dataclass
class ParkingSpot:
    """A detected parking spot.

    The corners are given in the vehicle frame of camera_preprocessing in mm
    (x: forward, y: left, z: up).
    """

    # Shape (4, 2): x, y in mm; order: front-left, front-right, back-right, back-left
    # (seen from the vehicle driving into the parking spot)
    corners: np.ndarray
    score: float = 1.0  # Confidence of the detection [0, 1]
    label: str = ""  # Class of the parking spot, e.g. free, occupied, blocked

    @property
    def center(self) -> np.ndarray:
        """Return the center of the parking spot in mm."""
        return self.corners.mean(axis=0)

    def corners_3d(self) -> np.ndarray:
        """Return the corners as (4, 3) world points on the ground (z = 0)."""
        return np.hstack([self.corners, np.zeros((len(self.corners), 1))])

    def to_image(
        self, transform: CoordinateTransform, *, birds_eye: bool = False
    ) -> np.ndarray:
        """Project the corners into the image.

        Args:
            transform: Coordinate transformation of camera_preprocessing.
            birds_eye: Project into the bird's eye view instead of the undistorted
                image.

        Returns:
            Corners in px with shape (4, 2).
        """
        if birds_eye:
            return transform.world_to_bird(self.corners_3d(), Unit.MILLIMETERS)
        return transform.world_to_camera(self.corners_3d(), Unit.MILLIMETERS)

    def to_global(self, pose: geometry_msgs.msg.Pose) -> np.ndarray:
        """Transform the corners from the vehicle frame into the global pose frame.

        Args:
            pose: Vehicle pose of the pose_estimation (orientation.z is the yaw
                in rad).

        Returns:
            Corners in the global frame with shape (4, 2) in mm.
        """
        yaw = pose.orientation.z
        rotation = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        )
        return self.corners @ rotation.T + [pose.position.x, pose.position.y]
