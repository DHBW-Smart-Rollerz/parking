from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch
import yaml
from rclpy.logging import get_logger

from parking.perception.network import ParkingNet, decode
from parking.perception.parking_spot import ParkingSpot

if TYPE_CHECKING:
    import numpy as np
    from camera_preprocessing.transformation.coordinate_transform import (
        CoordinateTransform,
    )


class ParkingSpotModel:
    """Model for detecting parking spots."""

    def __init__(
        self,
        root: str,
        model_config_path: str,
        transform: CoordinateTransform,
        *,
        birds_eye: bool = False,
    ):
        """Initializes the ParkingSpotModel.

        Args:
            root (str): Root path of the package
            model_config_path (str): Path to the model configuration
            transform (CoordinateTransform): Transformation of camera_preprocessing
                (image px <-> vehicle frame mm)
            birds_eye (bool): The input images are in the bird's eye view instead
                of the undistorted camera view
        """
        self.logger = get_logger("parking_spot_model")
        self.cfg = self._load_config(Path(root) / model_config_path)
        self.transform = transform
        self.birds_eye = birds_eye
        self.network: ParkingNet | None = None
        self._load_model(Path(root) / self.cfg["model_path"])

    def predict(self, image: np.ndarray) -> list[ParkingSpot]:
        """Detects the parking spots in the image.

        Args:
            image (np.ndarray): undistorted or bird's eye view grayscale image of
                camera_preprocessing

        Returns:
            list[ParkingSpot]: detected parking spots in the vehicle frame
        """
        if self.network is None:
            return []

        # Same input as in the training: resized grayscale image in [0, 1]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float()[None, None] / 255.0

        with torch.inference_mode():
            heatmap, corners = self.network(tensor)
        detections = decode(heatmap[0], corners[0], self.cfg["score_threshold"])

        # Scale the corners back to the image and convert them to mm
        scale = [
            image.shape[1] / self.input_size[0],
            image.shape[0] / self.input_size[1],
        ]
        spots = []
        for class_id, score, corners_px in detections:
            corners_mm = self.image_to_world(corners_px * scale)
            if corners_mm is None:
                continue  # Camera not calibrated
            spots.append(
                ParkingSpot(
                    corners=corners_mm[:, :2].astype(float),
                    score=score,
                    label=self.classes[class_id],
                )
            )
        return spots

    def image_to_world(self, points: np.ndarray) -> np.ndarray:
        """Converts pixel coordinates of the input image to the vehicle frame.

        Args:
            points (np.ndarray): points in px with shape (N, 2)

        Returns:
            np.ndarray: points in mm with shape (N, 3) (z = 0)
        """
        if self.birds_eye:
            return self.transform.bird_to_world(points)
        return self.transform.camera_to_world(points)

    def _load_config(self, path: Path) -> dict:
        """Loads the model configuration.

        Args:
            path (Path): path to the configuration

        Returns:
            dict: configuration
        """
        with path.open() as file:
            return yaml.safe_load(file)

    def _load_model(self, path: Path) -> None:
        """Loads the trained network (checkpoint of the training notebook).

        Args:
            path (Path): path to the model
        """
        if not path.exists():
            self.logger.warning(
                f"No trained model found at {path}. No parking spots are detected."
            )
            return

        checkpoint = torch.load(path, map_location="cpu")
        self.classes: list[str] = checkpoint["classes"]
        self.input_size: tuple[int, int] = tuple(checkpoint["input_size"])
        self.network = ParkingNet(len(self.classes), width=checkpoint["width"])
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        self.logger.info(f"Loaded model {path} with classes {self.classes}")
