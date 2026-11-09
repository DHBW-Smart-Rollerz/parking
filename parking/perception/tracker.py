"""Tracking of the detected parking spots over several frames.

Single detections are noisy and the parking spot leaves the image during the
maneuver. Every parking spot is therefore stored as a track in a fixed frame (the
global frame of the pose_estimation, or the vehicle frame if no pose is available):

- Prediction (optional): All parking spots are static, so in the frame of the
  tracker they all move with the same velocity (zero in the global frame, the
  inverse ego motion in the vehicle frame). It is estimated from the assigned
  detections and used to predict the tracks before the association, like the
  prediction step of a Kalman filter with a constant velocity model.
- Association: nearest neighbor of the centers within a gate (greedy).
- Pose: exponential moving average of the center, the yaw is averaged as doubled
  angle, because the rectangle is symmetric by 180 deg.
- Class: majority vote over the last detections.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from parking.perception.geometry import Pose2D, RectangleFit


@dataclass
class Measurement:
    """A detected parking spot after the rectangle fit."""

    kind: str  # "perpendicular" or "parallel"
    status: str  # "free", "occupied" or "blocked"
    score: float
    fit: RectangleFit  # In the frame of the tracker
    valid: bool  # The geometry is reliable (enough corners visible, rectangular)
    visible_corners: int = 4  # Corners not cut off, only with 4 the size is measured
    reason: str = ""  # Why the geometry is not reliable


@dataclass
class Track:
    """A parking spot tracked over several frames."""

    id: int
    kind: str
    pose: Pose2D  # Center and direction of the long side in the tracker frame
    length: float  # Measured length (long side) in mm
    width: float  # Measured width (short side) in mm
    votes: deque[str]
    hits: int = 1
    valid_hits: int = 0
    misses: int = 0
    scores: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    @property
    def status(self) -> str:
        """Return the class with the most votes."""
        return Counter(self.votes).most_common(1)[0][0]

    @property
    def score(self) -> float:
        """Return the mean score of the last detections."""
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def vote_ratio(self) -> float:
        """Return the share of the votes for the current class."""
        return Counter(self.votes)[self.status] / len(self.votes)


class SpotTracker:
    """Tracks parking spots with nearest neighbor association."""

    def __init__(
        self,
        *,
        gate: float = 250.0,
        smoothing: float = 0.4,
        min_hits: int = 3,
        min_valid_hits: int = 2,
        max_misses: int = 10,
        vote_window: int = 15,
        predict_motion: bool = True,
    ):
        """Initialize the SpotTracker.

        Args:
            gate: Maximum distance of the centers for the association in mm.
            smoothing: Weight of a new measurement in the moving average (0, 1].
            min_hits: Detections until a track is confirmed.
            min_valid_hits: Detections with reliable geometry until confirmed.
            max_misses: Frames without detection until a track is deleted.
            vote_window: Number of the last detections for the class vote.
            predict_motion: Predict the tracks with the common velocity of all
                spots (needed for tracking in the moving vehicle frame).
        """
        self.gate = gate
        self.smoothing = smoothing
        self.min_hits = min_hits
        self.min_valid_hits = min_valid_hits
        self.max_misses = max_misses
        self.vote_window = vote_window
        self.predict_motion = predict_motion
        self.tracks: list[Track] = []
        self.motion = Pose2D()  # Common motion of all spots per frame
        self._ids = count()

    def reset(self) -> None:
        """Delete all tracks."""
        self.tracks = []
        self.motion = Pose2D()

    def confirmed(self, track: Track) -> bool:
        """Return whether the track was detected often enough."""
        return track.hits >= self.min_hits and track.valid_hits >= self.min_valid_hits

    def update(self, measurements: list[Measurement]) -> list[Track]:
        """Update the tracks with the measurements of a new frame.

        Args:
            measurements: Detected parking spots in the frame of the tracker.

        Returns:
            All current tracks.
        """
        previous = {track.id: track.pose.position for track in self.tracks}
        if self.predict_motion:
            for track in self.tracks:
                track.pose = self.motion.compose(track.pose)

        pairs = sorted(
            (float(np.linalg.norm(m.fit.pose.position - t.pose.position)), i, j)
            for i, m in enumerate(measurements)
            for j, t in enumerate(self.tracks)
            if m.kind == t.kind
        )
        used_measurements: set[int] = set()
        used_tracks: set[int] = set()
        moved: list[tuple[np.ndarray, np.ndarray]] = []
        for distance, i, j in pairs:
            if distance > self.gate:
                break
            if i in used_measurements or j in used_tracks:
                continue
            used_measurements.add(i)
            used_tracks.add(j)
            track, measurement = self.tracks[j], measurements[i]
            if measurement.valid and track.valid_hits > 0:
                moved.append((previous[track.id], measurement.fit.pose.position))
            self._update_track(track, measurement)

        if self.predict_motion:
            self._update_motion(moved)

        for j, track in enumerate(self.tracks):
            if j not in used_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for i, measurement in enumerate(measurements):
            if i not in used_measurements:
                self.tracks.append(self._new_track(measurement))
        return self.tracks

    def _update_motion(self, moved: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Estimate the common motion of the spots since the last frame.

        Only the translation is estimated (median of the offsets, robust against
        wrong assignments). The rotation estimated from a few noisy spot centers was
        too unstable.

        Args:
            moved: Position of the assigned tracks in the last frame and of their
                detections in this frame.
        """
        alpha = self.smoothing
        if moved:
            x, y = np.median([target - source for source, target in moved], axis=0)
        else:
            x, y = 0.0, 0.0  # Without detections the motion decays
        self.motion = Pose2D(
            (1 - alpha) * self.motion.x + alpha * float(x),
            (1 - alpha) * self.motion.y + alpha * float(y),
        )

    def _new_track(self, measurement: Measurement) -> Track:
        """Create a track from an unassigned measurement."""
        fit = measurement.fit
        track = Track(
            id=next(self._ids),
            kind=measurement.kind,
            pose=fit.pose,
            length=fit.measured_length,
            width=fit.measured_width,
            votes=deque([measurement.status], maxlen=self.vote_window),
            valid_hits=int(measurement.valid),
        )
        track.scores.append(measurement.score)
        return track

    def _update_track(self, track: Track, measurement: Measurement) -> None:
        """Merge a measurement into a track."""
        track.hits += 1
        track.misses = 0
        track.votes.append(measurement.status)
        track.scores.append(measurement.score)

        # Unreliable geometry (e.g. cut off at the image border) only counts for the
        # class, as long as the track already has a reliable pose
        if not measurement.valid and track.valid_hits > 0:
            return
        # The first reliable measurement replaces the unreliable pose completely
        alpha = 1.0 if measurement.valid and track.valid_hits == 0 else self.smoothing
        track.valid_hits += int(measurement.valid)

        fit = measurement.fit
        position = (1 - alpha) * track.pose.position + alpha * fit.pose.position
        # Average of the doubled angle, the yaw of a rectangle is ambiguous by 180 deg
        doubled = (1 - alpha) * np.array(
            [math.cos(2 * track.pose.yaw), math.sin(2 * track.pose.yaw)]
        ) + alpha * np.array([math.cos(2 * fit.pose.yaw), math.sin(2 * fit.pose.yaw)])
        track.pose = Pose2D(
            float(position[0]),
            float(position[1]),
            math.atan2(doubled[1], doubled[0]) / 2,
        )
        track.length = (1 - alpha) * track.length + alpha * fit.measured_length
        track.width = (1 - alpha) * track.width + alpha * fit.measured_width
