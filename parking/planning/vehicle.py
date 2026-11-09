from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from parking.perception.geometry import rectangle_template

if TYPE_CHECKING:
    import numpy as np

    from parking.perception.geometry import Pose2D


@dataclass(frozen=True)
class VehicleGeometry:
    """Geometry of the vehicle for the kinematic single track model.

    The reference point of all planned poses is the center of the rear axle, because
    the single track model and the usual path tracking controllers (Pure Pursuit,
    Stanley) refer to it. All lengths in mm.
    """

    wheelbase: float = 260.0
    max_steering_angle: float = 35.0  # deg
    length: float = 370.0
    width: float = 200.0
    rear_overhang: float = 70.0  # Rear bumper behind the rear axle
    # Position of the rear axle on the x axis of the vehicle frame of
    # camera_preprocessing (origin of the pose_estimation)
    rear_axle_x: float = -85.0

    @classmethod
    def from_config(cls, config: dict) -> VehicleGeometry:
        """Create the geometry from the "vehicle" section of the planning config."""
        names = {field.name for field in fields(cls)}
        return cls(
            **{key: float(value) for key, value in config.items() if key in names}
        )

    @property
    def min_turning_radius(self) -> float:
        """Return the minimum turning radius of the rear axle center in mm."""
        return self.wheelbase / math.tan(math.radians(self.max_steering_angle))

    @property
    def front_length(self) -> float:
        """Return the distance from the rear axle to the front bumper in mm."""
        return self.length - self.rear_overhang

    @property
    def center_offset(self) -> float:
        """Return the distance from the rear axle to the center of the vehicle."""
        return self.length / 2 - self.rear_overhang

    def footprint(self, rear_axle: Pose2D, margin: float = 0.0) -> np.ndarray:
        """Return the corners of the vehicle outline.

        Args:
            rear_axle: Pose of the rear axle center.
            margin: Safety margin added on every side in mm.

        Returns:
            Corners with shape (4, 2), counterclockwise.
        """
        template = rectangle_template(self.length + 2 * margin, self.width + 2 * margin)
        template[:, 0] += self.center_offset
        return rear_axle.to_parent(template)

    def steering_angle(self, curvature: float) -> float:
        """Return the steering angle in deg for a curvature (single track model)."""
        return math.degrees(math.atan(self.wheelbase * curvature))

    def min_parallel_length(self, radius: float | None = None) -> float:
        """Return the minimum length of a parallel spot for a single maneuver.

        During the last arc into the spot, the front corner on the outer side of the
        turn moves on the largest circle around the instantaneous center of rotation
        (radius sqrt((R + b/2)^2 + l_v^2)). It must pass the corner of the obstacle
        in front of the spot, which lies at the distance R - b/2 from the center.
        This gives (Paromtchik & Laugier 1996, Vorobieva et al. 2015):

            L_min = l_h + sqrt((R + b/2)^2 + l_v^2 - (R - b/2)^2)

        Args:
            radius: Turning radius of the rear axle center (default: minimum).

        Returns:
            Minimum length in mm.
        """
        radius = self.min_turning_radius if radius is None else radius
        half_width = self.width / 2
        return self.rear_overhang + math.sqrt(
            (radius + half_width) ** 2
            + self.front_length**2
            - (radius - half_width) ** 2
        )
