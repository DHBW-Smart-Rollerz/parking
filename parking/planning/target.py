"""Selection of the target parking spot and planning of the path into it.

For every tracked parking spot it is checked whether the vehicle can park in it:

1. Class: The parking spot must be free (majority vote over several frames).
2. Footprint: The vehicle outline plus a safety margin, placed in the goal pose, must
   lie completely inside the measured parking spot.
3. Parallel spots: The length must be at least L_min for a single maneuver.
4. A path into the goal pose must exist.

The free spot with the shortest path is the target. Once selected, the target is kept
as long as it stays feasible (hysteresis), so the controller does not switch between
two similar spots during the maneuver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from parking.perception.geometry import (
    Pose2D,
    min_distance_to_edges,
    normalize_angle,
    rectangle_template,
)
from parking.planning.path import BACKWARD, FORWARD, Path
from parking.planning.planner import (
    PlanningError,
    plan_parallel,
    plan_straight_arc_straight,
)

if TYPE_CHECKING:
    from parking.perception.tracker import SpotTracker, Track
    from parking.planning.vehicle import VehicleGeometry

PERPENDICULAR = "perpendicular"
PARALLEL = "parallel"
DIRECTIONS = {"forward": FORWARD, "backward": BACKWARD}


@dataclass
class ParkingCandidate:
    """A tracked parking spot with the result of the checks (vehicle frame)."""

    track: Track
    spot: Pose2D  # Center and direction of the long side
    corners: np.ndarray  # Measured outline with shape (4, 2), counterclockwise
    goal: Pose2D | None = None  # Goal pose of the rear axle center
    footprint: np.ndarray | None = None  # Vehicle outline in the goal pose
    clearance: float = 0.0  # Minimum distance of the footprint to the spot edges
    path: Path | None = None
    reason: str = ""  # Why the spot is not feasible, empty if it is

    @property
    def feasible(self) -> bool:
        """Return whether the vehicle can park in the spot."""
        return self.path is not None and not self.reason


class ParkingPlanner:
    """Checks the tracked parking spots and plans the path into the best one."""

    def __init__(
        self, vehicle: VehicleGeometry, tracker: SpotTracker, config: dict
    ) -> None:
        """Initialize the ParkingPlanner.

        Args:
            vehicle: Geometry of the vehicle.
            tracker: Tracker of the parking spots (to check the confirmation).
            config: "planner" section of the planning config.
        """
        self.vehicle = vehicle
        self.tracker = tracker
        self.safety_margin = float(config["safety_margin"])
        self.radius = vehicle.min_turning_radius * float(config["radius_factor"])
        self.perpendicular_entry = [
            DIRECTIONS[name] for name in config["perpendicular_entry"]
        ]
        self.max_heading_error = math.radians(config["max_heading_error"])
        self.preferred_kind = config.get("preferred_kind", "any")
        self.target_id: int | None = None  # Track id of the last target

    def evaluate(self, track: Track, track_to_vehicle: Pose2D) -> ParkingCandidate:
        """Check a tracked parking spot and plan the path into it.

        Args:
            track: The tracked parking spot.
            track_to_vehicle: Pose of the tracker frame in the vehicle frame.

        Returns:
            The candidate in the vehicle frame.
        """
        spot = track_to_vehicle.compose(track.pose)
        candidate = ParkingCandidate(
            track=track,
            spot=spot,
            corners=spot.to_parent(rectangle_template(track.length, track.width)),
        )

        if not self.tracker.confirmed(track):
            candidate.reason = (
                f"unconfirmed ({track.hits}/{self.tracker.min_hits} hits, "
                f"{track.valid_hits}/{self.tracker.min_valid_hits} valid)"
            )
            return candidate
        if track.status != "free":
            candidate.reason = track.status
            return candidate

        start = Pose2D(self.vehicle.rear_axle_x, 0.0, 0.0)
        if track.kind == PERPENDICULAR:
            self._plan_perpendicular(candidate, start)
        else:
            self._plan_parallel(candidate, start)
        return candidate

    def select(self, candidates: list[ParkingCandidate]) -> ParkingCandidate | None:
        """Return the last target if still feasible, else the shortest path."""
        feasible = [
            c
            for c in candidates
            if c.feasible and self.preferred_kind in {"any", c.track.kind}
        ]
        target = next((c for c in feasible if c.track.id == self.target_id), None)
        if target is None:
            target = min(feasible, key=lambda c: c.path.length, default=None)  # type: ignore[union-attr]
        self.target_id = None if target is None else target.track.id
        return target

    def goal_pose(self, spot: Pose2D, heading: float) -> Pose2D:
        """Return the rear axle pose that centers the vehicle in the spot.

        p_rear_axle = c_spot - d * (cos(psi), sin(psi)), with d the distance from the
        rear axle to the center of the vehicle.
        """
        direction = np.array([math.cos(heading), math.sin(heading)])
        x, y = spot.position - self.vehicle.center_offset * direction
        return Pose2D(float(x), float(y), normalize_angle(heading))

    def _check_footprint(self, candidate: ParkingCandidate, goal: Pose2D) -> bool:
        """Place the vehicle in the goal pose and check that it fits into the spot."""
        candidate.goal = goal
        candidate.footprint = self.vehicle.footprint(goal, self.safety_margin)
        candidate.clearance = min_distance_to_edges(
            candidate.footprint, candidate.corners
        )
        if candidate.clearance < 0:
            candidate.reason = (
                f"vehicle does not fit ({candidate.track.length:.0f}x"
                f"{candidate.track.width:.0f} mm, clearance "
                f"{candidate.clearance:.0f} mm)"
            )
            return False
        return True

    def _plan_perpendicular(self, candidate: ParkingCandidate, start: Pose2D) -> None:
        """Plan into a perpendicular spot, the vehicle is parallel to the long side."""
        spot = candidate.spot
        # Long side direction that points away from the vehicle (into the spot)
        inward = spot.yaw
        if np.dot(spot.heading, spot.position - start.position) < 0:
            inward += math.pi

        reasons = []
        for direction in self.perpendicular_entry:
            # Forward: drive into the spot, backward: the vehicle faces the road
            heading = inward if direction == FORWARD else inward + math.pi
            if not self._check_footprint(candidate, self.goal_pose(spot, heading)):
                return
            try:
                candidate.path = plan_straight_arc_straight(
                    start, candidate.goal, self.radius, direction
                )
            except PlanningError as error:
                reasons.append(f"{'fw' if direction == FORWARD else 'bw'}: {error}")
            else:
                return
        candidate.reason = "; ".join(reasons)

    def _plan_parallel(self, candidate: ParkingCandidate, start: Pose2D) -> None:
        """Plan into a parallel spot, the vehicle is parallel to the long side."""
        spot = candidate.spot
        # Long side direction that is closer to the vehicle heading
        heading = spot.yaw if math.cos(spot.yaw) >= 0 else spot.yaw + math.pi
        if not self._check_footprint(candidate, self.goal_pose(spot, heading)):
            return

        min_length = self.vehicle.min_parallel_length(self.radius)
        if candidate.track.length < min_length:
            candidate.reason = (
                f"too short for one maneuver ({candidate.track.length:.0f} < "
                f"L_min {min_length:.0f} mm)"
            )
            return
        try:
            candidate.path = plan_parallel(
                start, candidate.goal, self.radius, self.max_heading_error
            )
        except PlanningError as error:
            candidate.reason = str(error)
