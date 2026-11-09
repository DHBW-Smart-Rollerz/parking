"""Converts CVAT annotations of the old bird's eye view to the parking bird's eye view.

The old view is the one of camera_preprocessing, the new one is defined in the
section parking_bev of config/model.yaml.

Every polygon point is transformed from the old view to the vehicle frame (mm) and
into the new view. The new view covers a larger region, so parking spots that were
cut off in the old view are now (partly) visible. Their polygons only contain the old
visible part and must be corrected in CVAT: a polygon with a corner at the border or
in the black area of the old image is marked as "occluded" (dashed in CVAT).

The images of the new view are created with
    python3 src/parking/tools/extract_frames.py <bag> <output> --view parking_bev
using the same --every as for the old images, so the image names match.

Usage:
    python3 src/parking/tools/convert_annotations_to_parking_bev.py \
        <old annotations.xml> <old image dir> <new annotations.xml>
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml
from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)

from parking.perception.bev import ParkingBirdseye

PACKAGE_DIR = Path(__file__).resolve().parents[1]
# Distance in px to the border or the black area of the old image, below which a
# corner counts as cut off or guessed
BORDER = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("annotations", type=Path, help="CVAT XML of the old view")
    parser.add_argument("image_dir", type=Path, help="Images of the old view")
    parser.add_argument("output", type=Path, help="CVAT XML of the new view")
    args = parser.parse_args()

    calibration = Calibration()
    calibration.setup()
    transform = CoordinateTransform(calib=calibration)
    with (PACKAGE_DIR / "config" / "model.yaml").open() as file:
        birdseye = ParkingBirdseye.from_config(
            transform, yaml.safe_load(file)["parking_bev"]
        )

    tree = ET.parse(args.annotations)  # noqa: S314 (own annotation files)
    images = spots = cut_off = 0
    for image in tree.getroot().iter("image"):
        old_image = load_image(args.image_dir, image.get("name", ""))
        valid_area = valid_old_area(old_image)
        image.set("width", str(birdseye.size[0]))
        image.set("height", str(birdseye.size[1]))
        images += 1

        for polygon in image.iter("polygon"):
            points = np.array(
                [p.split(",") for p in polygon.get("points", "").split(";")],
                dtype=float,
            )
            world = transform.bird_to_world(points)[:, :2].astype(float)
            new_points = birdseye.world_to_pixel(world)
            polygon.set("points", ";".join(f"{x:.2f},{y:.2f}" for x, y in new_points))
            spots += 1
            if not all_inside(points, valid_area):
                polygon.set("occluded", "1")
                cut_off += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(
        f"Converted {spots} parking spots in {images} images to {args.output}, "
        f"{cut_off} were cut off in the old view and are marked as occluded"
    )


def load_image(image_dir: Path, name: str) -> np.ndarray:
    """Loads an image of the old view (CVAT keeps the folder in the name).

    Args:
        image_dir: Directory of the old images
        name: Name of the image in the CVAT XML

    Returns:
        Grayscale image
    """
    path = image_dir / name
    if not path.exists():
        path = image_dir / Path(name).name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def valid_old_area(image: np.ndarray) -> np.ndarray:
    """Returns the mask of the old image without the black area and the border."""
    mask = (image > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((2 * BORDER + 1, 2 * BORDER + 1), np.uint8)
    return cv2.erode(mask, kernel)[1:-1, 1:-1]


def all_inside(points: np.ndarray, valid_area: np.ndarray) -> bool:
    """Checks whether all points lie in the valid area of the old image."""
    height, width = valid_area.shape
    x, y = np.round(points).astype(int).T
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return bool(inside.all() and (valid_area[y, x] > 0).all())


if __name__ == "__main__":
    main()
