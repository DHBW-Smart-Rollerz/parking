"""CVAT auto-annotation function with the trained ParkingNet.

Pre-annotates a CVAT task with polygons (four corners per parking spot), which then
only have to be corrected. Runs on the local machine with cvat-cli and therefore
also works with the self-hosted CVAT Community.

Usage:
    cvat-cli --server-host http://localhost --server-port 8080 --auth <user> \
        task auto-annotate <task id> \
        --function-file src/parking/tools/cvat_function.py \
        -p model_path=str:src/parking/models/parking_net.pt \
        --conf-threshold 0.3 --allow-unmatched-labels
"""

from __future__ import annotations

import sys
from pathlib import Path

import cvat_sdk.auto_annotation as cvataa
import numpy as np
import PIL.Image
import torch

# Import the parking package without sourcing the ROS workspace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parking.perception.network import ParkingNet, decode

DEFAULT_CONF_THRESHOLD = 0.3


class ParkingSpotFunction:
    """Detection function for cvat-cli."""

    def __init__(self, model_path: str):
        """Loads the trained network.

        Args:
            model_path: Path to parking_net.pt of the training notebook
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        self.classes: list[str] = checkpoint["classes"]
        self.input_size: tuple[int, int] = tuple(checkpoint["input_size"])
        self.network = ParkingNet(len(self.classes), width=checkpoint["width"])
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()

        self.spec = cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(name, class_id, type="polygon")
                for class_id, name in enumerate(self.classes)
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
        # Same input as in the training: resized grayscale image in [0, 1]
        gray = image.convert("L")
        resized = np.asarray(gray.resize(self.input_size, PIL.Image.Resampling.BOX))
        tensor = torch.from_numpy(resized.copy()).float()[None, None] / 255.0

        with torch.inference_mode():
            heatmap, corners = self.network(tensor)

        threshold = context.conf_threshold or DEFAULT_CONF_THRESHOLD
        scale = np.array(gray.size) / self.input_size
        return [
            cvataa.polygon(class_id, (spot_corners * scale).flatten().tolist())
            for class_id, _, spot_corners in decode(heatmap[0], corners[0], threshold)
        ]


def create(model_path: str) -> ParkingSpotFunction:
    """Factory called by cvat-cli with the -p parameters.

    Args:
        model_path: Path to parking_net.pt of the training notebook

    Returns:
        The detection function
    """
    return ParkingSpotFunction(model_path)
