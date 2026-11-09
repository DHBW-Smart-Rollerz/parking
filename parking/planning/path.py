"""Path of the vehicle as piecewise constant curvature kappa(s).

This is the interface to the controller: a list of segments with length, curvature
and driving direction. Straight lines (kappa = 0) and circular arcs (kappa = +-1/R)
are the building blocks of the classic geometric parking planners and of Dubins and
Reeds-Shepp paths. A change of the direction between two segments is a cusp.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from parking.perception.geometry import Pose2D, normalize_angle

FORWARD = 1
BACKWARD = -1
# Curvatures below this value are integrated as straight line (1/mm)
MIN_CURVATURE = 1e-9


@dataclass(frozen=True)
class PathSegment:
    """Segment with constant curvature."""

    length: float  # Driven distance in mm (>= 0)
    curvature: float  # 1/mm, positive turns left (steering to the left)
    direction: int = FORWARD  # FORWARD or BACKWARD

    def end_pose(self, start: Pose2D, distance: float | None = None) -> Pose2D:
        """Integrate the single track model along the segment (closed form).

        Args:
            start: Pose of the rear axle at the start of the segment.
            distance: Driven distance (default: the whole segment).

        Returns:
            Pose of the rear axle after the driven distance.
        """
        distance = self.length if distance is None else distance
        signed = self.direction * distance
        if abs(self.curvature) < MIN_CURVATURE:
            return Pose2D(
                start.x + signed * math.cos(start.yaw),
                start.y + signed * math.sin(start.yaw),
                start.yaw,
            )

        yaw = start.yaw + self.curvature * signed
        return Pose2D(
            start.x + (math.sin(yaw) - math.sin(start.yaw)) / self.curvature,
            start.y - (math.cos(yaw) - math.cos(start.yaw)) / self.curvature,
            normalize_angle(yaw),
        )


@dataclass
class Path:
    """Path of the rear axle center, starting at a pose."""

    start: Pose2D
    segments: list[PathSegment] = field(default_factory=list)
    name: str = ""  # Type of the maneuver, e.g. for the debug image

    @property
    def length(self) -> float:
        """Return the total driven distance in mm."""
        return sum(segment.length for segment in self.segments)

    @property
    def cusps(self) -> int:
        """Return the number of direction changes."""
        directions = [s.direction for s in self.segments if s.length > 0]
        return sum(a != b for a, b in itertools.pairwise(directions))

    def end_pose(self) -> Pose2D:
        """Return the pose at the end of the path."""
        pose = self.start
        for segment in self.segments:
            pose = segment.end_pose(pose)
        return pose

    def transformed(self, frame: Pose2D) -> Path:
        """Return the path in the parent frame of `frame`."""
        return Path(frame.compose(self.start), self.segments, self.name)

    def sample(self, step: float = 20.0) -> np.ndarray:
        """Sample poses along the path.

        Args:
            step: Distance between two samples in mm.

        Returns:
            Array with shape (N, 5): x, y, yaw, direction, curvature per pose.
        """
        samples = []
        pose = self.start
        for segment in self.segments:
            if segment.length <= 0:
                continue
            count = max(1, math.ceil(segment.length / step))
            for distance in np.linspace(0, segment.length, count + 1)[:-1]:
                sample = segment.end_pose(pose, float(distance))
                samples.append(
                    (
                        sample.x,
                        sample.y,
                        sample.yaw,
                        segment.direction,
                        segment.curvature,
                    )
                )
            pose = segment.end_pose(pose)
        last = self.segments[-1] if self.segments else PathSegment(0, 0)
        samples.append((pose.x, pose.y, pose.yaw, last.direction, last.curvature))
        return np.array(samples)
