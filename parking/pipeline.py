"""Parking pipeline after the detection, independent of ROS.

ParkingSpot (vehicle frame)
  -> rectangle fit with the known geometry        (perception.geometry)
  -> tracking in the global or vehicle frame      (perception.tracker)
  -> checks: class, footprint, L_min              (planning.target)
  -> goal pose and path                           (planning.planner)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from parking.perception.geometry import Pose2D, fit_rectangle
from parking.perception.tracker import Measurement, SpotTracker
from parking.planning.target import ParkingCandidate, ParkingPlanner
from parking.planning.vehicle import VehicleGeometry

if TYPE_CHECKING:
    from parking.perception.parking_spot import ParkingSpot

GLOBAL_FRAME = "global"
VEHICLE_FRAME = "vehicle"


@dataclass
class ParkingResult:
    """Result of one frame, everything in the vehicle frame."""

    measurements: list[Measurement] = field(default_factory=list)
    candidates: list[ParkingCandidate] = field(default_factory=list)
    target: ParkingCandidate | None = None
    frame: str = VEHICLE_FRAME  # Frame of the tracker


class ParkingPipeline:
    """Turns the detected parking spots into a target spot and a path."""

    def __init__(self, config: dict):
        """Initialize the ParkingPipeline.

        Args:
            config: Planning config (config/planning.yaml).
        """
        self.config = config
        self.classes: dict[str, dict] = config["classes"]
        self.vehicle = VehicleGeometry.from_config(config["vehicle"])
        self.tracker = SpotTracker(**config["tracker"])
        self.planner = ParkingPlanner(self.vehicle, self.tracker, config["planner"])
        self.frame: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> ParkingPipeline:
        """Create the pipeline from a planning config file."""
        with Path(path).open() as file:
            return cls(yaml.safe_load(file))

    def reset(self) -> None:
        """Delete all tracked parking spots."""
        self.tracker.reset()
        self.planner.target_id = None
        self.frame = None

    def measure(self, spot: ParkingSpot) -> Measurement | None:
        """Fit the rectangle to a detected parking spot and check its geometry.

        Args:
            spot: Detected parking spot in the vehicle frame.

        Returns:
            The measurement in the vehicle frame, None for an unknown class.
        """
        if spot.label not in self.classes:
            return None
        kind = self.classes[spot.label]["kind"]
        geometry = self.config["spot_geometry"][kind]
        fit_config = self.config["rectangle_fit"]
        visible = np.ones(4, bool) if spot.visible is None else spot.visible
        fit = fit_rectangle(
            spot.corners,
            geometry["length"],
            geometry["width"],
            visible=visible,
            refine_iterations=fit_config["refine_iterations"],
        )

        reason = ""
        tolerance = fit_config["size_tolerance"]
        if visible.sum() < fit_config["min_visible_corners"]:
            reason = f"cut off ({visible.sum()} corners visible)"
        elif fit.residual > fit_config["max_residual"]:
            reason = f"not rectangular ({fit.residual:.0f} mm)"
        elif (
            not self._length_ok(fit.measured_length, geometry, tolerance)
            or abs(fit.measured_width / fit.width - 1) > tolerance
        ):
            reason = (
                f"wrong size ({fit.measured_length:.0f}x{fit.measured_width:.0f} mm)"
            )

        return Measurement(
            kind=kind,
            status=self.classes[spot.label]["status"],
            score=spot.score,
            fit=fit,
            valid=not reason,
            visible_corners=int(visible.sum()),
            reason=reason,
        )

    @staticmethod
    def _length_ok(length: float, geometry: dict, tolerance: float) -> bool:
        """Check the measured length against the spot geometry of the config.

        Spots with a variable length (parallel spots) only need the minimum length,
        the others must have the nominal length within the tolerance.
        """
        if geometry.get("variable_length", False):
            return length >= geometry["min_length"]
        return abs(length / geometry["length"] - 1) <= tolerance

    def update(
        self, spots: list[ParkingSpot], vehicle_pose: Pose2D | None = None
    ) -> ParkingResult:
        """Process the detected parking spots of a new frame.

        Args:
            spots: Detected parking spots in the vehicle frame.
            vehicle_pose: Pose of the vehicle in the global frame of the
                pose_estimation. Without a pose the spots are tracked in the vehicle
                frame, which only works while the vehicle moves slowly.

        Returns:
            Measurements, checked candidates and target in the vehicle frame.
        """
        frame = VEHICLE_FRAME if vehicle_pose is None else GLOBAL_FRAME
        if frame != self.frame:
            self.reset()
            self.frame = frame
        vehicle_to_tracker = vehicle_pose or Pose2D()

        measurements = [m for m in map(self.measure, spots) if m is not None]
        tracks = self.tracker.update(
            [
                replace(
                    m, fit=replace(m.fit, pose=vehicle_to_tracker.compose(m.fit.pose))
                )
                for m in measurements
            ]
        )

        tracker_to_vehicle = vehicle_to_tracker.inverse()
        candidates = [self.planner.evaluate(t, tracker_to_vehicle) for t in tracks]
        return ParkingResult(
            measurements=measurements,
            candidates=candidates,
            target=self.planner.select(candidates),
            frame=frame,
        )
