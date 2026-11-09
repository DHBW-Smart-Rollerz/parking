from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import yaml
from rclpy.logging import get_logger

from parking.perception.network import ParkingNet, decode
from parking.perception.parking_spot import ParkingSpot

if TYPE_CHECKING:
    from parking.perception.view import ImageView


# Class id, score, corners in px with shape (4, 2), confidence per corner (or None)
Detection = tuple[int, float, np.ndarray, "np.ndarray | None"]


class ParkingSpotModel:
    """Model for detecting parking spots with the ParkingNet."""

    # Section of the detector in the model configuration
    DETECTOR = "parking_net"

    def __init__(
        self,
        root: str,
        model_config_path: str,
        view: ImageView,
    ):
        """Initializes the ParkingSpotModel.

        Args:
            root (str): Root path of the package
            model_config_path (str): Path to the model configuration
            view (ImageView): View of the input images (image px <-> vehicle
                frame mm)
        """
        self.logger = get_logger("parking_spot_model")
        self.cfg = self._load_config(Path(root) / model_config_path)
        self.view = view
        self.network: torch.nn.Module | None = None
        self.classes: list[str] = []
        self._load_model(Path(root) / self.detector_config["model_path"])

    @property
    def detector_config(self) -> dict:
        """Returns the section of the detector in the model configuration."""
        return self.cfg[self.DETECTOR]

    def predict(self, image: np.ndarray) -> list[ParkingSpot]:
        """Detects the parking spots in the image.

        Args:
            image (np.ndarray): grayscale image of the view (ImageView.prepare)

        Returns:
            list[ParkingSpot]: detected parking spots in the vehicle frame
        """
        if self.network is None:
            return []

        detections = self._detect(image)
        # Which corners are reliable: "image_area" (not in the black area of the bird's
        # eye view) or "keypoint_confidence" (confidence of the network per corner,
        # for networks trained with the corner visibility of yolo_export)
        mode = self.detector_config.get("corner_visibility", "image_area")
        threshold = self.detector_config.get("keypoint_threshold", 0.5)
        valid_area = self._valid_area(image) if detections else None

        spots = []
        for class_id, score, corners_px, corner_conf in detections:
            corners_mm = self.view.image_to_world(corners_px)
            if corners_mm is None:
                continue  # Camera not calibrated
            if mode == "keypoint_confidence" and corner_conf is not None:
                visible = np.asarray(corner_conf) >= threshold
            else:
                visible = self._corners_visible(corners_px, valid_area)
            spots.append(
                ParkingSpot(
                    corners=corners_mm,
                    score=score,
                    label=self.classes[class_id],
                    visible=visible,
                )
            )
        return spots

    def _valid_area(self, image: np.ndarray) -> np.ndarray:
        """Returns the mask of the image area with camera content.

        The bird's eye view is black outside of the field of view of the camera. A
        parking spot with a corner there is cut off, its corners are guessed.

        Args:
            image (np.ndarray): grayscale image of camera_preprocessing

        Returns:
            np.ndarray: mask with the same shape as the image, eroded by the border
        """
        border = int(self.cfg.get("visibility_border", 8))
        mask = (image > 0).astype(np.uint8)
        # Close small black spots inside the image (e.g. dark tar)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        return cv2.erode(mask, np.ones((2 * border + 1,) * 2, np.uint8))[1:-1, 1:-1]

    @staticmethod
    def _corners_visible(corners_px: np.ndarray, valid_area: np.ndarray) -> np.ndarray:
        """Checks for every corner whether it lies in the valid image area.

        Args:
            corners_px (np.ndarray): corners in px with shape (4, 2)
            valid_area (np.ndarray): mask of _valid_area

        Returns:
            np.ndarray: bool array with shape (4,)
        """
        height, width = valid_area.shape
        x, y = np.round(corners_px).astype(int).T
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        visible = np.zeros(len(corners_px), dtype=bool)
        visible[inside] = valid_area[y[inside], x[inside]] > 0
        return visible

    def _detect(self, image: np.ndarray) -> list[Detection]:
        """Runs the network on the image.

        Args:
            image (np.ndarray): grayscale image of the view

        Returns:
            list[Detection]: class id, score, corners in px of the image with shape
                (4, 2) and confidence per corner (None: not available) per parking
                spot
        """
        # Same input as in the training: resized grayscale image in [0, 1]
        resized = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float()[None, None] / 255.0

        with torch.inference_mode():
            heatmap, corners = self.network(tensor)
        detections = decode(heatmap[0], corners[0], self.cfg["score_threshold"])

        # Scale the corners back to the image
        scale = [
            image.shape[1] / self.input_size[0],
            image.shape[0] / self.input_size[1],
        ]
        return [
            (class_id, score, corners_px * scale, None)
            for class_id, score, corners_px in detections
        ]

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
        self.classes = checkpoint["classes"]
        self.input_size: tuple[int, int] = tuple(checkpoint["input_size"])
        self.network = ParkingNet(len(self.classes), width=checkpoint["width"])
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        self.logger.info(f"Loaded model {path} with classes {self.classes}")


def create_model(
    root: str,
    model_config_path: str,
    view: ImageView,
) -> ParkingSpotModel:
    """Creates the detector selected in the model configuration (key "detector").

    Args:
        root (str): Root path of the package
        model_config_path (str): Path to the model configuration
        view (ImageView): View of the input images

    Returns:
        ParkingSpotModel: ParkingNet or YOLO pose model
    """
    with (Path(root) / model_config_path).open() as file:
        detector = yaml.safe_load(file)["detector"]

    if detector == ParkingSpotModel.DETECTOR:
        model_class = ParkingSpotModel
    elif detector == "yolo_pose":
        # Import only if used, Ultralytics is not needed for the ParkingNet
        from parking.perception.yolo_model import YoloParkingSpotModel  # noqa: PLC0415

        model_class = YoloParkingSpotModel
    else:
        raise ValueError(f"Unknown detector '{detector}' in {model_config_path}")

    return model_class(root, model_config_path, view)
