from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import geometry_msgs.msg

    from parking.perception.view import ImageView


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
    # Shape (4,): the corner lies in the valid image area (not cut off), None: unknown
    visible: np.ndarray | None = None

    @property
    def center(self) -> np.ndarray:
        """Return the center of the parking spot in mm."""
        return self.corners.mean(axis=0)

    def to_image(self, view: ImageView) -> np.ndarray | None:
        """Project the corners into the image of the view.

        Args:
            view: View of the image.

        Returns:
            Corners in px with shape (4, 2), None if the camera is not calibrated.
        """
        return view.world_to_image(self.corners)

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
