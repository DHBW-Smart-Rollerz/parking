"""Export of the CVAT annotations to the YOLO pose format of Ultralytics.

Every parking spot becomes one object: the class, the bounding box around the four
corners and the four corners as keypoints (in the order of canonical_corner_order).

Structure of the export:

    <out_dir>/
    ├── data.yaml
    ├── images/train/*.png, images/val/*.png
    └── labels/train/*.txt, labels/val/*.txt

One line per parking spot in the label files (all values normalized to [0, 1]):

    class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import yaml

from parking.perception.network import NUM_CORNERS

if TYPE_CHECKING:
    from parking.training.dataset import Sample

# Keypoint visibility of YOLO: 2 = labeled and visible
VISIBLE = 2

# Corner order after a horizontal flip (bottom-left <-> bottom-right, ...). Only
# used by Ultralytics if the flip augmentation (fliplr) is enabled.
FLIP_IDX = [1, 0, 3, 2]


def label_line(class_id: int, corners: np.ndarray, image_size: tuple[int, int]) -> str:
    """Creates the YOLO pose label of one parking spot.

    Args:
        class_id: Class of the parking spot
        corners: Corners in px of the image with shape (4, 2)
        image_size: Size of the image (width, height)

    Returns:
        Line of the label file
    """
    points = np.clip(corners / image_size, 0.0, 1.0)
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    values = [str(class_id)]
    values += [f"{v:.6f}" for v in ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)]
    for x, y in points:
        values += [f"{x:.6f}", f"{y:.6f}", str(VISIBLE)]
    return " ".join(values)


def export_yolo_pose(
    splits: dict[str, list[Sample]], classes: list[str], out_dir: str | Path
) -> Path:
    """Exports the annotated images as YOLO pose dataset.

    Args:
        splits: Annotated images per split, e.g. {"train": [...], "val": [...]}
        classes: Class names, the index is the class id
        out_dir: Directory of the export, it is deleted and created again

    Returns:
        Path to data.yaml for the training with Ultralytics
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, samples in splits.items():
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        for sample in samples:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(sample.image_path)
            image_size = (image.shape[1], image.shape[0])

            shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
            lines = [
                label_line(spot.class_id, spot.corners, image_size)
                for spot in sample.spots
            ]
            # Images without parking spots get an empty label file (negatives)
            (label_dir / f"{sample.image_path.stem}.txt").write_text(
                "".join(f"{line}\n" for line in lines)
            )

    data_yaml = out_dir / "data.yaml"
    data = {
        "path": str(out_dir.resolve()),
        **{split: f"images/{split}" for split in splits},
        "kpt_shape": [NUM_CORNERS, 3],
        "flip_idx": FLIP_IDX,
        "names": dict(enumerate(classes)),
    }
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return data_yaml
