"""CVAT auto-annotation function with the trained YOLO pose model.

Same as cvat_function.py, but with the model of notebooks/train_yolo_pose.ipynb.

Usage:
    cvat-cli --server-host http://localhost --server-port 8080 --auth <user> \
        task auto-annotate <task id> \
        --function-file src/parking/tools/cvat_function_yolo.py \
        -p model_path=str:src/parking/models/yolo_pose.pt \
        --conf-threshold 0.3 --allow-unmatched-labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cvat_sdk.auto_annotation as cvataa
import numpy as np
from ultralytics import YOLO

if TYPE_CHECKING:
    import PIL.Image

DEFAULT_CONF_THRESHOLD = 0.3
IMAGE_SIZE = 640


class ParkingSpotFunction:
    """Detection function for cvat-cli."""

    def __init__(self, model_path: str):
        """Loads the trained network.

        Args:
            model_path: Path to yolo_pose.pt of the training notebook
        """
        self.network = YOLO(model_path, task="pose")
        self.spec = cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(name, class_id, type="polygon")
                for class_id, name in self.network.names.items()
            ]
        )

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list:
        """Detects the parking spots in one image of the task.

        Args:
            context: Context of CVAT with the confidence threshold
            image: Image of the task

        Returns:
            One polygon per parking spot
        """
        # Same input as in the training: grayscale image as BGR
        bgr = np.asarray(image.convert("L").convert("RGB"))[:, :, ::-1]
        result = self.network.predict(
            np.ascontiguousarray(bgr),
            imgsz=IMAGE_SIZE,
            conf=context.conf_threshold or DEFAULT_CONF_THRESHOLD,
            verbose=False,
        )[0]
        if result.keypoints is None:
            return []

        return [
            cvataa.polygon(class_id, corners.flatten().tolist())
            for class_id, corners in zip(
                result.boxes.cls.int().tolist(),
                result.keypoints.xy.cpu().numpy(),
                strict=True,
            )
        ]


def create(model_path: str) -> ParkingSpotFunction:
    """Factory called by cvat-cli with the -p parameters.

    Args:
        model_path: Path to yolo_pose.pt of the training notebook

    Returns:
        The detection function
    """
    return ParkingSpotFunction(model_path)
