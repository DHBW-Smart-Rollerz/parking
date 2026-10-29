"""Extracts preprocessed frames from a rosbag for the annotation in CVAT.

The raw images are processed in the same way as in the camera_preprocessing node
(grayscale, downscale, brightness, undistortion, bird's eye view) with the
calibration installed in the camera_preprocessing package.

Usage:
    python3 src/parking/tools/extract_frames.py <bag> <output dir> \
        [--every 5] [--view bev|undistorted] [--topic /camera/image/raw]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from camera_preprocessing.transformation.birds_eyed_view import Birdseye
from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.distortion import Distortion
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

TARGET_BRIGHTNESS = 50  # Same value as in the camera_preprocessing node


def adjust_brightness(image: np.ndarray, target_brightness: float) -> np.ndarray:
    """Scales the image to the target mean brightness (as camera_preprocessing).

    Args:
        image: Grayscale image
        target_brightness: Target mean value [0, 255]

    Returns:
        Adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=target_brightness / np.mean(image))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", type=Path, help="Path to the rosbag directory")
    parser.add_argument("output", type=Path, help="Output directory of the images")
    parser.add_argument("--every", type=int, default=5, help="Keep every n-th frame")
    parser.add_argument("--view", choices=["bev", "undistorted"], default="bev")
    parser.add_argument("--topic", default="/camera/image/raw")
    args = parser.parse_args()

    calibration = Calibration()
    calibration.setup()
    distortion = Distortion(calibration)
    birdseye = Birdseye(calibration, distortion)
    bridge = CvBridge()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag)),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))

    args.output.mkdir(parents=True, exist_ok=True)
    index = 0
    saved = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        if index % args.every == 0:
            msg = deserialize_message(data, Image)
            image = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            image = calibration.downscale_image(image)
            image = adjust_brightness(image, TARGET_BRIGHTNESS)
            image = distortion.undistort_image(image)
            if args.view == "bev":
                image = birdseye.transform_img(image)

            cv2.imwrite(str(args.output / f"{args.bag.name}_{index:06d}.png"), image)
            saved += 1
        index += 1

    print(f"Saved {saved} of {index} frames to {args.output}")


if __name__ == "__main__":
    main()
