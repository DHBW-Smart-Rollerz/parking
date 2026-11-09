"""2D poses and the fit of a rectangle with known size to detected corners.

The rectangle fit is a least squares registration of the nominal parking spot
(template) onto the four detected corners with 3 degrees of freedom (x, y, yaw), i.e.
the 2D case of the Kabsch/Umeyama algorithm. The network then only has to provide the
position of the parking spot, the shape is known from the track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi).

    Args:
        angle: Angle in rad.

    Returns:
        Normalized angle in rad.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    """Position (mm) and orientation (rad) of a frame in its parent frame."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """Return the position as array with shape (2,)."""
        return np.array([self.x, self.y])

    @property
    def heading(self) -> np.ndarray:
        """Return the unit vector of the orientation with shape (2,)."""
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])

    def rotation(self) -> np.ndarray:
        """Return the rotation matrix with shape (2, 2)."""
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[cos, -sin], [sin, cos]])

    def to_parent(self, points: np.ndarray) -> np.ndarray:
        """Transform points from this frame into the parent frame.

        Args:
            points: Points with shape (N, 2).

        Returns:
            Points in the parent frame with shape (N, 2).
        """
        return np.asarray(points, dtype=float) @ self.rotation().T + self.position

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Transform points from the parent frame into this frame.

        Args:
            points: Points with shape (N, 2).

        Returns:
            Points in this frame with shape (N, 2).
        """
        return (np.asarray(points, dtype=float) - self.position) @ self.rotation()

    def compose(self, other: Pose2D) -> Pose2D:
        """Return the pose `other` (given in this frame) in the parent frame."""
        x, y = self.to_parent(other.position[None])[0]
        return Pose2D(x, y, normalize_angle(self.yaw + other.yaw))

    def inverse(self) -> Pose2D:
        """Return the pose of the parent frame in this frame."""
        x, y = self.to_local(np.zeros((1, 2)))[0]
        return Pose2D(x, y, normalize_angle(-self.yaw))


@dataclass(frozen=True)
class RectangleFit:
    """Result of the rectangle fit.

    The yaw is the direction of the long side (local x axis) and is ambiguous by
    180 deg, it is normalized to [-90, 90) deg.
    """

    pose: Pose2D  # Center and orientation of the rectangle
    length: float  # Nominal length (long side) in mm
    width: float  # Nominal width (short side) in mm
    residual: float  # RMS distance between the fitted and the detected corners in mm
    measured_length: float  # Length of the detected corners along the long side
    measured_width: float  # Width of the detected corners along the short side

    def corners(self, *, measured: bool = False) -> np.ndarray:
        """Return the corners of the rectangle in the parent frame.

        Args:
            measured: Use the measured instead of the nominal size.

        Returns:
            Corners with shape (4, 2), counterclockwise.
        """
        if measured:
            return self.pose.to_parent(
                rectangle_template(self.measured_length, self.measured_width)
            )
        return self.pose.to_parent(rectangle_template(self.length, self.width))


def rectangle_template(length: float, width: float) -> np.ndarray:
    """Return the corners of a rectangle centered at the origin.

    Args:
        length: Size along the x axis.
        width: Size along the y axis.

    Returns:
        Corners with shape (4, 2), counterclockwise starting at front-left.
    """
    half_length, half_width = length / 2, width / 2
    return np.array(
        [
            [half_length, half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
            [half_length, -half_width],
        ]
    )


def fit_rectangle(
    corners: np.ndarray,
    length: float,
    width: float,
    visible: np.ndarray | None = None,
    refine_iterations: int = 0,
) -> RectangleFit:
    """Fit a rectangle with known size to four detected corners (least squares).

    The order of the detected corners is unknown in relation to the rectangle (the
    detector sorts them in the image and the transformation into the vehicle frame
    mirrors the order). Therefore all 8 assignments (4 start corners, 2 directions)
    are fitted and the one with the smallest residual is used.

    For a fixed assignment, the optimal translation is the difference of the
    weighted centroids and the optimal rotation follows in closed form from the
    weighted cross covariance (Kabsch 1976, Umeyama 1991). Three visible corners
    already determine the rectangle. Corners that are not visible (cut off at the
    image border, the detector guesses them) therefore get a low weight, and with
    three visible corners the size is measured from the two visible edges.

    Args:
        corners: Detected corners with shape (4, 2) in mm, order along the outline.
        length: Nominal length (long side) in mm.
        width: Nominal width (short side) in mm.
        visible: Shape (4,), corners that are not visible get a low weight
            (default: all visible).
        refine_iterations: Number of refits with the measured instead of the
            nominal size, if all corners are visible. With refinement the residual
            only measures how rectangular the corners are, the size is checked
            separately.

    Returns:
        The best fit (length and width stay the nominal size). The residual only
        considers the visible corners.
    """
    corners = np.asarray(corners, dtype=float)
    visible = np.ones(len(corners), bool) if visible is None else np.asarray(visible)

    fit = _fit_rectangle(corners, length, width, visible)
    if visible.sum() == len(corners) - 1:
        # Size from the two edges at the corner opposite to the hidden one
        hidden = int(np.flatnonzero(~visible)[0])
        middle = corners[(hidden + 2) % 4]
        edges = [
            float(np.linalg.norm(corners[(hidden + offset) % 4] - middle))
            for offset in (1, 3)
        ]
        refined = _fit_rectangle(corners, max(edges), min(edges), visible)
        return replace(
            refined,
            length=length,
            width=width,
            measured_length=max(edges),
            measured_width=min(edges),
        )
    if not visible.all():
        # The guessed corners are no measurement of the size
        return replace(fit, measured_length=length, measured_width=width)
    for _ in range(refine_iterations):
        # The long side always stays the local x axis
        refined = _fit_rectangle(
            corners,
            max(fit.measured_length, fit.measured_width),
            min(fit.measured_length, fit.measured_width),
            visible,
        )
        fit = replace(refined, length=length, width=width)
    return fit


def fit_rigid_transform(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> Pose2D:
    """Find the rotation and translation that maps source onto target (least squares).

    2D case of the Kabsch/Umeyama algorithm: The optimal translation maps the
    weighted centroids onto each other, the optimal rotation is the angle of the
    weighted sums of the cross and dot products of the centered points.

    Args:
        source: Points with shape (N, 2).
        target: Corresponding points with shape (N, 2).
        weights: Weights of the points with shape (N,) (default: equal).

    Returns:
        The transformation as pose, target = pose.to_parent(source).
    """
    weights = np.ones(len(source)) if weights is None else np.asarray(weights)
    weights = weights / weights.sum()
    source_center = weights @ source
    target_center = weights @ target
    source_centered = source - source_center
    target_centered = target - target_center
    cross = weights @ (
        source_centered[:, 0] * target_centered[:, 1]
        - source_centered[:, 1] * target_centered[:, 0]
    )
    dot = weights @ np.sum(source_centered * target_centered, axis=1)
    rotation = Pose2D(yaw=math.atan2(float(cross), float(dot)))
    x, y = target_center - rotation.rotation() @ source_center
    return Pose2D(float(x), float(y), rotation.yaw)


# Weight of a corner that is not visible in the rectangle fit
HIDDEN_CORNER_WEIGHT = 0.01


def _fit_rectangle(
    corners: np.ndarray, length: float, width: float, visible: np.ndarray
) -> RectangleFit:
    """Fit a rectangle with a fixed size, see fit_rectangle."""
    template = rectangle_template(length, width)
    weights = np.where(visible, 1.0, HIDDEN_CORNER_WEIGHT)
    # The residual only considers the visible corners
    residual_weights = visible / visible.sum()

    best: tuple[float, Pose2D, np.ndarray] | None = None
    for direction in (1, -1):
        for shift in range(4):
            # Detected corner of every template corner
            order = np.roll(np.arange(4)[::direction], -shift)
            target = corners[order]
            residual_weight = residual_weights[order]
            pose = fit_rigid_transform(template, target, weights[order])

            errors = np.sum((pose.to_parent(template) - target) ** 2, axis=1)
            residual = float(np.sqrt(np.sum(residual_weight * errors)))
            if best is None or residual < best[0]:
                best = (residual, pose, target)

    residual, pose, target = best  # type: ignore[misc]
    # The rectangle is symmetric, use the yaw in [-90, 90) deg
    pose = Pose2D(pose.x, pose.y, normalize_angle(2 * pose.yaw) / 2)
    local = pose.to_local(target)
    return RectangleFit(
        pose=pose,
        length=length,
        width=width,
        residual=residual,
        measured_length=2 * float(np.mean(np.abs(local[:, 0]))),
        measured_width=2 * float(np.mean(np.abs(local[:, 1]))),
    )


def points_in_convex_polygon(
    points: np.ndarray, polygon: np.ndarray, margin: float = 0.0
) -> bool:
    """Check whether all points lie inside a convex polygon.

    Args:
        points: Points with shape (N, 2).
        polygon: Corners of the convex polygon with shape (M, 2), counterclockwise.
        margin: Minimum distance of the points to the edges.

    Returns:
        True if all points are inside with at least the given margin.
    """
    return bool(min_distance_to_edges(points, polygon) >= margin)


def min_distance_to_edges(points: np.ndarray, polygon: np.ndarray) -> float:
    """Return the minimum signed distance of the points to the edges of a polygon.

    Args:
        points: Points with shape (N, 2).
        polygon: Corners of the convex polygon with shape (M, 2), counterclockwise.

    Returns:
        Minimum distance in mm, positive inside and negative outside of the polygon.
    """
    edges = np.roll(polygon, -1, axis=0) - polygon
    # Inward normals of a counterclockwise polygon
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    distances = np.einsum("nmk,mk->nm", points[:, None, :] - polygon[None], normals)
    return float(distances.min())
