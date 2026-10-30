"""Dataset of parking spots annotated in CVAT.

Expected annotation: One polygon with exactly four points per parking spot, exported
as "CVAT for images 1.1". The label of the polygon is the class (e.g. free,
occupied, blocked). The order of the clicked points does not matter, the corners are
sorted into a fixed order when loading (see canonical_corner_order).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from parking.perception.network import NUM_CORNERS, STRIDE

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Annotated parking spot."""

    class_id: int
    corners: np.ndarray  # Shape (4, 2) in px of the original image


@dataclass
class Sample:
    """Annotated image."""

    image_path: Path
    spots: list[Annotation]


def cvat_labels(xml_path: str | Path) -> list[str]:
    """Reads the label names of a CVAT export in the order defined in CVAT.

    Args:
        xml_path: Path to annotations.xml

    Returns:
        Label names, the index is the class id
    """
    root = ET.parse(xml_path).getroot()  # noqa: S314 (own annotation files)
    return [label.findtext("name") for label in root.iter("label")]


def canonical_corner_order(corners: np.ndarray) -> np.ndarray:
    """Sorts the four corners of a parking spot into a fixed order.

    The network learns the corners in a fixed order, so it must not depend on the
    order in which they were clicked in CVAT. Order in the image: bottom-left,
    bottom-right, top-right, top-left.

    Args:
        corners: Corners in px with shape (4, 2) in any order along the outline

    Returns:
        The sorted corners with shape (4, 2)
    """
    x, y = corners[:, 0], corners[:, 1]
    # Signed area (shoelace) is negative for bottom-left -> bottom-right -> ...
    # in image coordinates (y down), otherwise reverse the direction
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) > 0:
        corners = corners[::-1]
    # Start at the bottom-left corner (largest y - x)
    start = int(np.argmax(corners[:, 1] - corners[:, 0]))
    return np.roll(corners, -start, axis=0)


def load_cvat_xml(
    xml_path: str | Path, image_dir: str | Path, classes: list[str]
) -> list[Sample]:
    """Loads the annotations of a CVAT export ("CVAT for images 1.1").

    Args:
        xml_path: Path to annotations.xml
        image_dir: Directory of the annotated images
        classes: Class names, the index is the class id

    Returns:
        All annotated images. Images without parking spots are kept as negatives.
        Images with an invalid polygon (unknown label or not four points) are
        skipped completely, otherwise the parking spot would be trained as
        background.
    """
    image_dir = Path(image_dir)
    root = ET.parse(xml_path).getroot()  # noqa: S314 (own annotation files)

    samples = []
    skipped = []
    for image in root.iter("image"):
        spots = []
        for polygon in image.iter("polygon"):
            label = polygon.get("label")
            points = np.array(
                [p.split(",") for p in polygon.get("points").split(";")], dtype=float
            )
            if label not in classes or len(points) != NUM_CORNERS:
                skipped.append(f"{image.get('name')} ({label}, {len(points)} points)")
                break
            spots.append(
                Annotation(classes.index(label), canonical_corner_order(points))
            )
        else:
            # CVAT keeps the folder structure of the upload or share in the name
            image_path = image_dir / image.get("name")
            if not image_path.exists():
                image_path = image_dir / Path(image.get("name")).name
            samples.append(Sample(image_path, spots))

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} images with an unknown label or a polygon "
            f"without {NUM_CORNERS} points, e.g. {skipped[:3]}"
        )
    return samples


def bag_name(sample: Sample) -> str:
    """Returns the rosbag of an image (file name without the frame index).

    Args:
        sample: Annotated image of extract_frames.py

    Returns:
        Name of the rosbag
    """
    return sample.image_path.stem.rsplit("_", 1)[0]


def split_by_blocks(
    samples: list[Sample], val_fraction: float, block_size: int, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Splits the images into training and validation in blocks of consecutive frames.

    Consecutive frames of a rosbag are almost identical, a random split per image
    would put nearly the same image into training and validation. Whole rosbags as
    validation only work with many rosbags. Therefore every rosbag is cut into blocks
    of consecutive frames and a random part of the blocks of every rosbag is used for
    the validation, so all rosbags (and maneuvers) are in both sets.

    Args:
        samples: Annotated images of extract_frames.py
        val_fraction: Part of the blocks of every rosbag used for the validation
        block_size: Number of consecutive images per block
        seed: Seed of the random selection

    Returns:
        Training and validation images
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for bag in sorted({bag_name(s) for s in samples}):
        frames = sorted(
            (s for s in samples if bag_name(s) == bag), key=lambda s: s.image_path.stem
        )
        blocks = [frames[i : i + block_size] for i in range(0, len(frames), block_size)]
        num_val = max(1, round(val_fraction * len(blocks)))
        val_blocks = set(rng.choice(len(blocks), num_val, replace=False).tolist())
        for index, block in enumerate(blocks):
            (val if index in val_blocks else train).extend(block)
    return train, val


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], sigma: float) -> None:
    """Draws a gaussian peak into the heatmap (keeps the maximum of overlaps).

    Args:
        heatmap: Heatmap of one class with shape (h, w)
        center: Center cell (x, y)
        sigma: Standard deviation in cells
    """
    h, w = heatmap.shape
    radius = int(3 * sigma)
    x, y = center
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    gaussian = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)


class ParkingSpotDataset(Dataset):
    """Torch dataset that creates the network targets from the annotations."""

    def __init__(
        self,
        samples: list[Sample],
        num_classes: int,
        input_size: tuple[int, int],
        *,
        augment: bool = False,
        flip: bool = False,
        sigma: float = 2.0,
    ):
        """Initializes the dataset.

        Args:
            samples: Annotated images
            num_classes: Number of classes
            input_size: Network input size (width, height), divisible by 16
            augment: Apply random augmentation (only for training)
            flip: Also mirror images horizontally during augmentation. Only useful
                if the type of a parking spot does not depend on the side of the
                road (on the track: 90 degree left, parallel right).
            sigma: Size of the heatmap peaks in output cells
        """
        self.samples = samples
        self.num_classes = num_classes
        self.input_size = input_size
        self.augment = augment
        self.flip = flip
        self.sigma = sigma
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(sample.image_path)

        # Resize to the network input and scale the corners accordingly
        scale = np.array(self.input_size) / [image.shape[1], image.shape[0]]
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_AREA)
        spots = [(s.class_id, s.corners * scale) for s in sample.spots]

        if self.augment:
            image, spots = self._augment(image, spots)

        return {
            "image": torch.from_numpy(image).float().unsqueeze(0) / 255.0,
            **self._targets(spots),
        }

    def _augment(
        self, image: np.ndarray, spots: list[tuple[int, np.ndarray]]
    ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
        # Brightness and contrast
        alpha = self.rng.uniform(0.6, 1.4)
        beta = self.rng.uniform(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Blur
        if self.rng.random() < 0.3:  # noqa: PLR2004
            image = cv2.GaussianBlur(image, (5, 5), 0)

        # Horizontal flip: front-left <-> front-right, back-right <-> back-left
        if self.flip and self.rng.random() < 0.5:  # noqa: PLR2004
            image = image[:, ::-1].copy()
            width = image.shape[1]
            flipped = []
            for class_id, corners in spots:
                mirrored = corners.copy()
                mirrored[:, 0] = width - 1 - mirrored[:, 0]
                flipped.append((class_id, mirrored[[1, 0, 3, 2]]))
            spots = flipped

        return image, spots

    def _targets(self, spots: list[tuple[int, np.ndarray]]) -> dict[str, torch.Tensor]:
        w, h = self.input_size[0] // STRIDE, self.input_size[1] // STRIDE
        heatmap = np.zeros((self.num_classes, h, w), dtype=np.float32)
        corners = np.zeros((2 * NUM_CORNERS, h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)

        for class_id, spot_corners in spots:
            cells = spot_corners / STRIDE
            cx, cy = np.floor(cells.mean(axis=0)).astype(int)
            if not (0 <= cx < w and 0 <= cy < h):
                continue  # Center outside of the image
            draw_gaussian(heatmap[class_id], (cx, cy), self.sigma)
            corners[:, cy, cx] = (cells - [cx, cy]).reshape(-1)
            mask[cy, cx] = 1.0

        return {
            "heatmap": torch.from_numpy(heatmap),
            "corners": torch.from_numpy(corners),
            "mask": torch.from_numpy(mask),
        }
