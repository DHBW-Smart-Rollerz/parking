from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
from ultralytics import YOLO

from parking.perception.model import Detection, ParkingSpotModel

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class YoloParkingSpotModel(ParkingSpotModel):
    """Model for detecting parking spots with a YOLO pose network (Ultralytics).

    Every parking spot is one object with its four corners as keypoints, trained with
    notebooks/train_yolo_pose.ipynb.
    """

    DETECTOR = "yolo_pose"

    @property
    def detector_config(self) -> dict:
        """Returns the network and input size for the view of the images."""
        networks = self.cfg[self.DETECTOR]
        if self.view.name not in networks:
            raise ValueError(
                f"No YOLO network for the view '{self.view.name}' in the section "
                f"{self.DETECTOR} of the model configuration"
            )
        return networks[self.view.name]

    def _detect(self, image: np.ndarray) -> list[Detection]:
        """Runs the network on the image.

        Args:
            image (np.ndarray): grayscale image of the view

        Returns:
            list[Detection]: class id, score, corners in px of the image with shape
                (4, 2) and confidence per corner per parking spot
        """
        # Ultralytics expects a BGR image, the training images were loaded the same way
        result = self.network.predict(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            imgsz=self.detector_config["image_size"],
            conf=self.cfg["score_threshold"],
            verbose=False,
        )[0]
        if result.keypoints is None:
            return []

        keypoints = result.keypoints
        corner_conf = (
            keypoints.conf.cpu().numpy()
            if keypoints.conf is not None
            else [None] * len(keypoints)
        )
        return list(
            zip(
                result.boxes.cls.int().tolist(),
                result.boxes.conf.tolist(),
                keypoints.xy.cpu().numpy(),
                corner_conf,
                strict=True,
            )
        )

    def _load_model(self, path: Path) -> None:
        """Loads the trained network (best.pt of the Ultralytics training).

        Args:
            path (Path): path to the model
        """
        if not path.exists():
            self.logger.warning(
                f"No trained model found at {path}. No parking spots are detected."
            )
            return

        self.network = YOLO(path, task="pose")
        names = self.network.names
        self.classes = [names[class_id] for class_id in range(len(names))]
        self.logger.info(f"Loaded YOLO model {path} with classes {self.classes}")
