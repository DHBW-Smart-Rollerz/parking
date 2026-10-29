from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )

    from parking.perception.parking_spot import ParkingSpot


def draw_spots(
    image: np.ndarray,
    spots: list[ParkingSpot],
    transform: CoordinateTransform,
    *,
    birds_eye: bool = False,
) -> np.ndarray:
    """Draws the parking spots into the image.

    Args:
        image (np.ndarray): undistorted or bird's eye view grayscale image
        spots (list[ParkingSpot]): detected parking spots in the vehicle frame
        transform (CoordinateTransform): transformation of camera_preprocessing
        birds_eye (bool): the image is in the bird's eye view

    Returns:
        np.ndarray: BGR image with the parking spots
    """
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for spot in spots:
        corners_px = spot.to_image(transform, birds_eye=birds_eye)
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
