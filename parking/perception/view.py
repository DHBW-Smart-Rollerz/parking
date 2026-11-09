"""Image views of the detection and their conversion between px and the vehicle frame.

The detector runs on one of three views, selected with the ROS parameter "view":

- bev:         bird's eye view of camera_preprocessing (/camera/image/bev)
- undistorted: undistorted camera image (/camera/image/undistorted)
- parking_bev: larger bird's eye view of the parking package (perception/bev.py),
               computed in the node from the undistorted image
               (/camera/image/undistorted), section parking_bev of config/model.yaml

The model has to be trained on the same view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from camera_preprocessing.transformation.coordinate_transform import Unit

from parking.perception.bev import ParkingBirdseye

if TYPE_CHECKING:
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )

VIEWS = ("bev", "undistorted", "parking_bev")


class ImageView:
    """View of camera_preprocessing (bird's eye view or undistorted image)."""

    def __init__(self, transform: CoordinateTransform, *, birds_eye: bool):
        """Initialize the ImageView.

        Args:
            transform: Transformation of camera_preprocessing (calibration).
            birds_eye: Bird's eye view of camera_preprocessing instead of the
                undistorted image.
        """
        self.transform = transform
        self.birds_eye = birds_eye
        self.name = "bev" if birds_eye else "undistorted"

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Turn the subscribed image into the input image of the detector."""
        return image

    def image_to_world(self, pixels: np.ndarray) -> np.ndarray | None:
        """Convert pixels of the view to the vehicle frame.

        Args:
            pixels: Points in px with shape (N, 2).

        Returns:
            Points in mm with shape (N, 2), None if the camera is not calibrated.
        """
        if self.birds_eye:
            points = self.transform.bird_to_world(pixels)
        else:
            points = self.transform.camera_to_world(pixels)
        if points is None:
            return None
        return np.asarray(points, dtype=float)[:, :2]

    def world_to_image(self, points: np.ndarray) -> np.ndarray | None:
        """Convert points of the vehicle frame to pixels of the view.

        Args:
            points: Points in mm with shape (N, 2).

        Returns:
            Points in px with shape (N, 2), None if the camera is not calibrated.
        """
        points_3d = np.hstack([points, np.zeros((len(points), 1))])
        if self.birds_eye:
            pixels = self.transform.world_to_bird(points_3d, Unit.MILLIMETERS)
        else:
            pixels = self.transform.world_to_camera(points_3d, Unit.MILLIMETERS)
        if pixels is None:
            return None
        return np.asarray(pixels, dtype=float).reshape(-1, 2)


class ParkingBevView(ImageView):
    """Bird's eye view of the parking package, computed from the undistorted image."""

    def __init__(self, transform: CoordinateTransform, config: dict):
        """Initialize the ParkingBevView.

        Args:
            transform: Transformation of camera_preprocessing (calibration).
            config: Section parking_bev of config/model.yaml.
        """
        super().__init__(transform, birds_eye=True)
        self.name = "parking_bev"
        self.birdseye = ParkingBirdseye.from_config(transform, config)

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Compute the bird's eye view from the undistorted image."""
        return self.birdseye.warp(image)

    def image_to_world(self, pixels: np.ndarray) -> np.ndarray | None:
        """Convert pixels of the view to the vehicle frame (mm)."""
        return self.birdseye.pixel_to_world(pixels)

    def world_to_image(self, points: np.ndarray) -> np.ndarray | None:
        """Convert points of the vehicle frame (mm) to pixels of the view."""
        return self.birdseye.world_to_pixel(points)


def create_view(
    name: str, transform: CoordinateTransform, model_config: dict
) -> ImageView:
    """Create the view with the given name.

    Args:
        name: "bev", "undistorted" or "parking_bev".
        transform: Transformation of camera_preprocessing (calibration).
        model_config: Content of config/model.yaml.

    Returns:
        The view.
    """
    if name == "parking_bev":
        return ParkingBevView(transform, model_config["parking_bev"])
    if name in VIEWS:
        return ImageView(transform, birds_eye=name == "bev")
    raise ValueError(f"Unknown view '{name}', expected one of {VIEWS}")
