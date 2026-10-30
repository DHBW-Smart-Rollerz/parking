from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
from ultralytics import YOLO

from parking.perception.model import ParkingSpotModel

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class YoloParkingSpotModel(ParkingSpotModel):
    """Model for detecting parking spots with a YOLO pose network (Ultralytics).

    Every parking spot is one object with its four corners as keypoints, trained with
    notebooks/train_yolo_pose.ipynb.
    """

    DETECTOR = "yolo_pose"

    def _detect(self, image: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
        """Runs the network on the image.

        Args:
            image (np.ndarray): grayscale image of camera_preprocessing

        Returns:
            list[tuple[int, float, np.ndarray]]: class id, score and corners in px of
                the image with shape (4, 2) per parking spot
        """
        # Ultralytics expects a BGR image, the training images were loaded the same way
        result = self.network.predict(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            imgsz=self.cfg[self.DETECTOR]["image_size"],
            conf=self.cfg["score_threshold"],
            verbose=False,
        )[0]
        if result.keypoints is None:
            return []

        return list(
            zip(
                result.boxes.cls.int().tolist(),
                result.boxes.conf.tolist(),
                result.keypoints.xy.cpu().numpy(),
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
