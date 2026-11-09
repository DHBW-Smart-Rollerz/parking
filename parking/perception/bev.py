"""Bird's eye view of the parking package with a configurable ground region.

The bird's eye view of camera_preprocessing only covers about x = -120..1280 mm and
y = -920..1000 mm, so many parking spots are cut off at its border although they are
visible in the camera image. This view is computed from the undistorted image with
the calibration of camera_preprocessing for a freely chosen rectangle on the ground.
It does not change the bird's eye view of camera_preprocessing, which other nodes use.

Image axes: up = +x (forward), right = -y (right), like the view of
camera_preprocessing. The scale is the same everywhere (resolution in mm/px).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from camera_preprocessing.transformation.coordinate_transform import Unit

if TYPE_CHECKING:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )


class ParkingBirdseye:
    """Bird's eye view of a rectangle on the ground (vehicle frame, mm)."""

    def __init__(
        self,
        transform: CoordinateTransform,
        x_range: tuple[float, float] = (-150.0, 2000.0),
        y_range: tuple[float, float] = (-1200.0, 1200.0),
        resolution: float = 3.0,
    ):
        """Initialize the ParkingBirdseye.

        Args:
            transform: Transformation of camera_preprocessing (calibration).
            x_range: Region on the ground in driving direction (min, max) in mm.
            y_range: Region on the ground to the left (min, max) in mm.
            resolution: Size of a pixel in mm.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.size = (
            round((y_range[1] - y_range[0]) / resolution),  # width
            round((x_range[1] - x_range[0]) / resolution),  # height
        )

        # Homography from the pixels of this view to the undistorted image, fitted
        # to a grid of ground points (exact for a plane, the grid averages rounding)
        u, v = np.meshgrid(
            np.linspace(0, self.size[0], 9), np.linspace(0, self.size[1], 9)
        )
        pixels = np.stack([u.ravel(), v.ravel()], axis=1)
        world = self.pixel_to_world(pixels)
        camera = transform.world_to_camera(
            np.hstack([world, np.zeros((len(world), 1))]), Unit.MILLIMETERS
        )
        if camera is None:
            raise RuntimeError("camera_preprocessing is not calibrated")
        self.homography, _ = cv2.findHomography(
            pixels.astype(np.float32),
            np.asarray(camera, dtype=np.float32).reshape(-1, 2),
        )

    @classmethod
    def from_config(
        cls, transform: CoordinateTransform, config: dict
    ) -> ParkingBirdseye:
        """Create the view from the "parking_bev" section of config/model.yaml."""
        return cls(
            transform,
            x_range=tuple(config["x_range"]),
            y_range=tuple(config["y_range"]),
            resolution=float(config["resolution"]),
        )

    def warp(self, undistorted: np.ndarray) -> np.ndarray:
        """Transform the undistorted image into this bird's eye view.

        Args:
            undistorted: Undistorted image of camera_preprocessing.

        Returns:
            The bird's eye view, black outside of the camera's field of view.
        """
        return cv2.warpPerspective(
            undistorted,
            self.homography,
            self.size,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def pixel_to_world(self, pixels: np.ndarray) -> np.ndarray:
        """Convert pixels of this view to the vehicle frame.

        Args:
            pixels: Points in px with shape (N, 2).

        Returns:
            Points in mm with shape (N, 2).
        """
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        x = self.x_range[1] - pixels[:, 1] * self.resolution
        y = self.y_range[1] - pixels[:, 0] * self.resolution
        return np.stack([x, y], axis=1)

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Convert points of the vehicle frame to pixels of this view.

        Args:
            points: Points in mm with shape (N, 2) or (N, 3).

        Returns:
            Points in px with shape (N, 2).
        """
        points = np.asarray(points, dtype=float).reshape(len(points), -1)
        u = (self.y_range[1] - points[:, 1]) / self.resolution
        v = (self.x_range[1] - points[:, 0]) / self.resolution
        return np.stack([u, v], axis=1)
