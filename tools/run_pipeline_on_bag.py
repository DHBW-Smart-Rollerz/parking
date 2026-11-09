"""Runs the detection and the parking pipeline offline on a rosbag.

The raw images are preprocessed like in camera_preprocessing (see extract_frames.py),
then the configured detector, the rectangle fit, the tracking and the path planning
run on every n-th frame. The debug images are written as PNG and/or video, at the end
a summary is printed (reliable detections, measured spot sizes, targets, run times).

The rosbags only contain the camera image, so the spots are tracked in the vehicle
frame. No running ROS system is needed.

Usage:
    python3 src/parking/tools/run_pipeline_on_bag.py <bag> <output dir> \
        [--every 3] [--video] [--no-images] [--max-frames N] [--show-detections] \
        [--view bev|undistorted|parking_bev]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import rosbag2_py
import yaml
from camera_preprocessing.transformation.birds_eyed_view import Birdseye
from camera_preprocessing.transformation.calibration import Calibration
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)
from camera_preprocessing.transformation.distortion import Distortion
from cv_bridge import CvBridge
from extract_frames import TARGET_BRIGHTNESS, adjust_brightness
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

from parking.perception import model, visualization
from parking.perception.view import VIEWS, create_view
from parking.pipeline import ParkingPipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

    from parking.pipeline import ParkingResult

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CAMERA_FPS = 30  # Frame rate of the camera in the rosbags
NUM_CORNERS = 4  # Only with all corners visible the size is measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", type=Path, help="Path to the rosbag directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--every", type=int, default=3, help="Use every n-th frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0: all frames")
    parser.add_argument("--video", action="store_true", help="Write debug.mp4")
    parser.add_argument("--no-images", action="store_true", help="No PNG per frame")
    parser.add_argument(
        "--show-detections",
        action="store_true",
        help="Also draw the raw detections of the network (pink)",
    )
    parser.add_argument(
        "--view",
        choices=VIEWS,
        default="bev",
        help="View of the detector, the model has to be trained on it",
    )
    parser.add_argument("--topic", default="/camera/image/raw")
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--planning-config", default="config/planning.yaml")
    args = parser.parse_args()

    calibration = Calibration()
    calibration.setup()
    transform = CoordinateTransform(calib=calibration)
    with (PACKAGE_DIR / args.model_config).open() as file:
        view = create_view(args.view, transform, yaml.safe_load(file))
    detector = model.create_model(
        root=str(PACKAGE_DIR),
        model_config_path=args.model_config,
        view=view,
    )
    pipeline = ParkingPipeline.from_file(PACKAGE_DIR / args.planning_config)
    args.output.mkdir(parents=True, exist_ok=True)

    video = None
    statistics = Statistics()
    images = read_images(
        args.bag, args.topic, calibration, args.every, birds_eye=args.view == "bev"
    )
    for index, topic_image in images:
        if args.max_frames and statistics.frames >= args.max_frames:
            break
        image = view.prepare(topic_image)

        start = time.perf_counter()
        spots = detector.predict(image)
        statistics.times["detection"].append(time.perf_counter() - start)
        start = time.perf_counter()
        result = pipeline.update(spots)
        statistics.times["pipeline"].append(time.perf_counter() - start)
        statistics.add(result)

        debug_image = visualization.draw_parking(image, result, view, pipeline.vehicle)
        if args.show_detections:
            visualization.draw_detections(debug_image, spots, view)
        if not args.no_images:
            name = f"{args.bag.name}_{index:06d}.png"
            cv2.imwrite(str(args.output / name), debug_image)
        if args.video:
            if video is None:
                height, width = debug_image.shape[:2]
                video = cv2.VideoWriter(
                    str(args.output / "debug.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    CAMERA_FPS / args.every,
                    (width, height),
                )
            video.write(debug_image)

    if video is not None:
        video.release()
    statistics.print()


def read_images(
    bag: Path, topic: str, calibration: Calibration, every: int, *, birds_eye: bool
) -> Iterator[tuple[int, np.ndarray]]:
    """Reads every n-th image of the rosbag, preprocessed like camera_preprocessing.

    Args:
        bag: Path to the rosbag directory
        topic: Topic of the raw camera image
        calibration: Calibration of camera_preprocessing
        every: Use every n-th image
        birds_eye: Bird's eye view of camera_preprocessing (/camera/image/bev)
            instead of the undistorted image (/camera/image/undistorted)

    Yields:
        Index of the image in the rosbag and the preprocessed image
    """
    distortion = Distortion(calibration)
    birdseye = Birdseye(calibration, distortion)
    bridge = CvBridge()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag)),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    index = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        if index % every == 0:
            image = bridge.imgmsg_to_cv2(
                deserialize_message(data, Image), desired_encoding="mono8"
            )
            image = calibration.downscale_image(image)
            image = adjust_brightness(image, TARGET_BRIGHTNESS)
            image = distortion.undistort_image(image)
            yield index, birdseye.transform_img(image) if birds_eye else image
        index += 1


@dataclass
class Statistics:
    """Statistics of the run over all frames."""

    frames: int = 0
    # Counts of the measurement results per kind, of the free spots and the targets
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # Measured length and width of the reliable, completely visible detections
    sizes: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    # Run times per step in s
    times: dict[str, list] = field(default_factory=lambda: defaultdict(list))

    def add(self, result: ParkingResult) -> None:
        """Adds the result of a frame."""
        self.frames += 1
        for measurement in result.measurements:
            reason = measurement.reason.split(" (")[0] or "ok"
            self.counts[f"{measurement.kind} detections"][reason] += 1
            if measurement.valid and measurement.visible_corners == NUM_CORNERS:
                fit = measurement.fit
                self.sizes[measurement.kind].append(
                    (fit.measured_length, fit.measured_width)
                )
        for candidate in result.candidates:
            if candidate.track.status == "free":
                reason = candidate.reason.split(" (")[0] or "feasible"
                self.counts[f"free {candidate.track.kind} spots"][reason] += 1
        target = result.target
        self.counts["target"][target.track.kind if target else "none"] += 1

    def print(self) -> None:
        """Prints the statistics."""
        print(f"\nProcessed frames: {self.frames}")
        for name, counter in self.counts.items():
            print(f"{name}: {dict(counter)}")
        for kind, values in self.sizes.items():
            length, width = np.median(values, axis=0)
            print(
                f"Measured size {kind} (median of {len(values)} reliable detections): "
                f"{length:.0f} x {width:.0f} mm"
            )
        for name, values in self.times.items():
            print(f"Run time {name}: {1000 * np.mean(values):.1f} ms per frame (mean)")


if __name__ == "__main__":
    main()
