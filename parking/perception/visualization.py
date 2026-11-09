from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from parking.perception.geometry import Pose2D
from parking.planning.path import FORWARD

if TYPE_CHECKING:
    from collections.abc import Callable

    from parking.perception.parking_spot import ParkingSpot
    from parking.perception.view import ImageView
    from parking.pipeline import ParkingResult
    from parking.planning.vehicle import VehicleGeometry

# Colors (BGR) of the parking spot status
STATUS_COLORS = {
    "free": (0, 200, 0),
    "occupied": (0, 0, 255),
    "blocked": (0, 140, 255),
}
FORWARD_COLOR = (0, 255, 255)
BACKWARD_COLOR = (255, 0, 255)
VEHICLE_COLOR = (255, 200, 0)
INVALID_COLOR = (160, 160, 160)
TEXT_COLOR = (255, 255, 255)
DETECTION_COLOR = (255, 120, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_spots(
    image: np.ndarray,
    spots: list[ParkingSpot],
    view: ImageView,
) -> np.ndarray:
    """Draws the parking spots into the image.

    Args:
        image (np.ndarray): grayscale image of the view
        spots (list[ParkingSpot]): detected parking spots in the vehicle frame
        view (ImageView): view of the image

    Returns:
        np.ndarray: BGR image with the parking spots
    """
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for spot in spots:
        corners_px = spot.to_image(view)
        if corners_px is None:
            continue
        cv2.polylines(
            debug_image,
            [corners_px.round().astype(np.int32)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )
        x, y = corners_px.mean(axis=0)
        cv2.circle(debug_image, (round(x), round(y)), 4, (0, 255, 0), -1)

        # Front-left corner, to check the corner order
        fx, fy = corners_px[0]
        cv2.circle(debug_image, (round(fx), round(fy)), 4, (255, 0, 0), -1)

        cv2.putText(
            debug_image,
            f"{spot.label} {spot.score:.2f}",
            (round(x) + 6, round(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return debug_image


def draw_parking(
    image: np.ndarray,
    result: ParkingResult,
    view: ImageView,
    vehicle: VehicleGeometry,
) -> np.ndarray:
    """Draws the tracked parking spots, the target and the planned path.

    - Thin gray: detection with unreliable geometry (e.g. cut off), with the reason
    - Green/red/orange: tracked spot (free/occupied/blocked), thick if confirmed
    - Light blue: vehicle outline now and in the goal pose of the target
    - Yellow/magenta: planned path forward/backward, white dots are cusps

    Args:
        image (np.ndarray): grayscale image of the view
        result (ParkingResult): result of the parking pipeline (vehicle frame)
        view (ImageView): view of the image
        vehicle (VehicleGeometry): geometry of the vehicle

    Returns:
        np.ndarray: BGR image
    """
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def to_image(points: np.ndarray) -> np.ndarray | None:
        pixels = view.world_to_image(points)
        if pixels is None:
            return None  # Camera not calibrated
        return pixels.round().astype(np.int32)

    def polygon(points: np.ndarray, color: tuple, thickness: int) -> None:
        pixels = to_image(points)
        if pixels is not None:
            cv2.polylines(
                debug_image, [pixels], isClosed=True, color=color, thickness=thickness
            )

    def text(points: np.ndarray, label: str, color: tuple) -> None:
        pixels = to_image(points.mean(axis=0, keepdims=True))
        if pixels is not None:
            x, y = pixels[0]
            cv2.putText(debug_image, label, (x - 40, y), FONT, 0.4, color, 1)

    for measurement in result.measurements:
        if not measurement.valid:
            corners = measurement.fit.corners(measured=True)
            polygon(corners, INVALID_COLOR, 1)
            text(corners, measurement.reason, INVALID_COLOR)

    for candidate in result.candidates:
        track = candidate.track
        color = STATUS_COLORS.get(track.status, INVALID_COLOR)
        confirmed = not candidate.reason.startswith("unconfirmed")
        polygon(candidate.corners, color, 2 if confirmed else 1)
        label = f"#{track.id} {track.status} {track.score:.2f}"
        text(candidate.corners, label, color)
        if candidate.reason and confirmed and track.status == "free":
            text(candidate.corners - [80, 0], candidate.reason[:40], color)

    # Vehicle outline at the current pose
    polygon(vehicle.footprint(Pose2D(vehicle.rear_axle_x, 0, 0)), VEHICLE_COLOR, 1)

    target = result.target
    if target is not None and target.path is not None:
        polygon(target.corners, STATUS_COLORS["free"], 4)
        polygon(target.footprint, VEHICLE_COLOR, 2)  # type: ignore[arg-type]
        _draw_path(debug_image, target.path.sample(20.0), to_image)
        status = (
            f"target #{target.track.id} {target.track.kind}: {target.path.name}, "
            f"{target.path.length:.0f} mm, {target.path.cusps} cusp(s), "
            f"clearance {target.clearance:.0f} mm"
        )
    else:
        status = "no target"
    for row, line in enumerate([status, f"tracking: {result.frame} frame"]):
        cv2.putText(debug_image, line, (5, 15 + 17 * row), FONT, 0.45, TEXT_COLOR, 1)
    return debug_image


def draw_detections(
    debug_image: np.ndarray,
    spots: list[ParkingSpot],
    view: ImageView,
) -> np.ndarray:
    """Draws the raw detections of the network (before fit and tracking).

    Visible corners are filled, guessed corners (cut off) are hollow.

    Args:
        debug_image (np.ndarray): BGR image to draw into (modified in place)
        spots (list[ParkingSpot]): detected parking spots in the vehicle frame
        view (ImageView): view of the image

    Returns:
        np.ndarray: the BGR image
    """
    for spot in spots:
        corners_px = spot.to_image(view)
        if corners_px is None:
            continue
        corners_px = corners_px.round().astype(np.int32)
        cv2.polylines(
            debug_image, [corners_px], isClosed=True, color=DETECTION_COLOR, thickness=1
        )
        visible = (
            np.ones(len(corners_px), bool) if spot.visible is None else spot.visible
        )
        for corner, corner_visible in zip(corners_px, visible, strict=True):
            cv2.circle(
                debug_image,
                tuple(corner),
                4,
                DETECTION_COLOR,
                -1 if corner_visible else 1,
            )
        label = spot.label.replace("Parking_", "")
        x, y = corners_px[0]
        cv2.putText(
            debug_image,
            f"{label} {spot.score:.2f}",
            (x + 4, y - 4),
            FONT,
            0.35,
            DETECTION_COLOR,
            1,
        )
    return debug_image


def _draw_path(
    debug_image: np.ndarray, samples: np.ndarray, to_image: Callable
) -> None:
    """Draws the sampled path, colored by the driving direction.

    Args:
        debug_image (np.ndarray): BGR image to draw into
        samples (np.ndarray): samples of Path.sample with shape (N, 5)
        to_image (Callable): projection of (N, 2) points in mm into the image
    """
    pixels = to_image(samples[:, :2])
    if pixels is None:
        return
    directions = samples[:, 3]
    for i in range(len(samples) - 1):
        color = FORWARD_COLOR if directions[i] == FORWARD else BACKWARD_COLOR
        cv2.line(debug_image, tuple(pixels[i]), tuple(pixels[i + 1]), color, 2)
    for i in np.flatnonzero(np.diff(directions) != 0):
        cv2.circle(debug_image, tuple(pixels[i + 1]), 5, TEXT_COLOR, -1)
    # Goal pose: rear axle with the heading
    cv2.circle(debug_image, tuple(pixels[-1]), 5, VEHICLE_COLOR, -1)
