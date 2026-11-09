import math
from pathlib import Path

import cv_bridge
import geometry_msgs.msg
import nav_msgs.msg
import rclpy
import sensor_msgs.msg
import std_msgs.msg
import yaml
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)
from rclpy.executors import ExternalShutdownException
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer

from parking.perception import model, visualization
from parking.perception.geometry import Pose2D
from parking.perception.view import create_view
from parking.pipeline import ParkingPipeline, ParkingResult


class ParkingPerceptionNode(SmartyNode):
    """Detects parking spots in the camera image and plans the path into one."""

    def __init__(self):
        """Initialize the ParkingPerceptionNode."""
        super().__init__(
            "parking_perception_node",
            "parking",
            node_parameters={
                # Subscriber topics
                "image_subscriber": "/camera/image/bev",
                "pose_subscriber": "/pose_estimation/pose",
                # Publisher topics
                "debug_image_publisher": "/parking/debug_image",
                "path_publisher": "/parking/path",
                "target_publisher": "/parking/target_pose",
                "segments_publisher": "/parking/path_segments",
                # Parameters
                "model_config_path": "config/model.yaml",
                "planning_config_path": "config/planning.yaml",
                "view": "bev",
                "use_pose": True,
                "frame_id": "vehicle",
                "state": NodeState.ACTIVE.value,
                "debug": False,
            },
            subscribed_topics={
                "image_subscriber": (
                    sensor_msgs.msg.Image,
                    self.image_callback,
                    1,
                ),
                "pose_subscriber": (
                    geometry_msgs.msg.Pose,
                    self.pose_callback,
                    1,
                ),
            },
            published_topics={
                "debug_image_publisher": (sensor_msgs.msg.Image, 1),
                "path_publisher": (nav_msgs.msg.Path, 1),
                "target_publisher": (geometry_msgs.msg.PoseStamped, 1),
                "segments_publisher": (std_msgs.msg.Float32MultiArray, 1),
            },
        )

        # Get the parameters from the yaml file
        self.get_logger().info(f"Package Path: {self.package_path}")

        # Create required objects
        self.cv_bridge = cv_bridge.CvBridge()
        self.latest_pose: geometry_msgs.msg.Pose | None = None

        # Transformation between the image (px) and the vehicle frame (mm). Loads the
        # calibration (calib.bin) of the camera_preprocessing package.
        self.transform = CoordinateTransform(debug=self._debug)

        # View of the detector (bev, undistorted or parking_bev), converts between
        # the image and the vehicle frame
        with (Path(self.package_path) / self.model_config_path).open() as file:
            model_config = yaml.safe_load(file)
        self.view = create_view(self.view_name, self.transform, model_config)

        self.model = model.create_model(
            root=self.package_path,
            model_config_path=self.model_config_path,
            view=self.view,
        )
        self.pipeline = ParkingPipeline.from_file(
            f"{self.package_path}/{self.planning_config_path}"
        )

        self.get_logger().info("ParkingPerceptionNode initialized")

    @property
    def model_config_path(self) -> str:
        """Return the model config path."""
        return self.get_parameter("model_config_path").value  # type: ignore

    @property
    def planning_config_path(self) -> str:
        """Return the planning config path."""
        return self.get_parameter("planning_config_path").value  # type: ignore

    @property
    def use_pose(self) -> bool:
        """Return whether the spots are tracked in the global frame of the pose."""
        return self.get_parameter("use_pose").value  # type: ignore

    @property
    def frame_id(self) -> str:
        """Return the frame id of the published path (vehicle frame)."""
        return self.get_parameter("frame_id").value  # type: ignore

    @property
    def view_name(self) -> str:
        """Return the view of the detector (bev, undistorted or parking_bev)."""
        return self.get_parameter("view").value  # type: ignore

    def _reset(self) -> None:
        """Reset the node to its initial state."""
        del self.model
        self.latest_pose = None
        self.model = model.create_model(
            root=self.package_path,
            model_config_path=self.model_config_path,
            view=self.view,
        )
        self.pipeline.reset()

    def pose_callback(self, msg: geometry_msgs.msg.Pose) -> None:
        """Store the latest vehicle pose from the pose estimation."""
        self.latest_pose = msg

    def image_callback(self, msg: sensor_msgs.msg.Image) -> None:
        """Executed by the ROS2 system whenever a new image is received."""
        # Execute the prediction
        self.execute_prediction(msg)

    @timer.Timer(name="total", filter_strength=40)
    def execute_prediction(self, msg: sensor_msgs.msg.Image) -> None:
        """Execute the prediction.

        Args:
            msg: The image message.
        """
        with timer.Timer(name="msg_transport", filter_strength=40):
            # The image has to be retrieved from the message
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

        with timer.Timer(name="view", filter_strength=40):
            # E.g. the parking bird's eye view from the undistorted image
            image = self.view.prepare(image)

        with timer.Timer(name="prediction", filter_strength=40):
            # Get the detected parking spots in the vehicle frame
            spots = self.model.predict(image)

        with timer.Timer(name="planning", filter_strength=40):
            # Rectangle fit, tracking, checks of the spots and path into the target
            vehicle_pose = None
            if self.use_pose and self.latest_pose is not None:
                vehicle_pose = Pose2D(
                    self.latest_pose.position.x,
                    self.latest_pose.position.y,
                    self.latest_pose.orientation.z,  # yaw in rad (pose_estimation)
                )
            result = self.pipeline.update(spots, vehicle_pose)

        with timer.Timer(name="publish", filter_strength=40):
            self.publish_result(result, msg.header)

        if self._debug:
            with timer.Timer(name="debug", filter_strength=40):
                debug_image = visualization.draw_parking(
                    image, result, self.view, self.pipeline.vehicle
                )

                self.debug_image_publisher.publish(  # type: ignore
                    self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
                )

        timer.Timer(logger=self.get_logger().info).print()  # type: ignore

    def publish_result(
        self, result: ParkingResult, image_header: std_msgs.msg.Header
    ) -> None:
        """Publish the path into the target spot in the vehicle frame.

        Without a target, an empty path and no segments are published, so the
        controller knows that there is no target (anymore). All positions in mm (as
        the pose_estimation), the orientations are quaternions.

        Args:
            result: Result of the parking pipeline.
            image_header: Header of the image (time stamp of the detection).
        """
        header = std_msgs.msg.Header(stamp=image_header.stamp, frame_id=self.frame_id)
        path_msg = nav_msgs.msg.Path(header=header)
        segments_msg = std_msgs.msg.Float32MultiArray()

        target = result.target
        if target is not None and target.path is not None and target.goal is not None:
            step = self.pipeline.config["planner"]["sample_step"]
            path_msg.poses = [
                geometry_msgs.msg.PoseStamped(header=header, pose=to_pose(x, y, yaw))
                for x, y, yaw, _, _ in target.path.sample(step)
            ]
            self.target_publisher.publish(  # type: ignore
                geometry_msgs.msg.PoseStamped(
                    header=header,
                    pose=to_pose(target.goal.x, target.goal.y, target.goal.yaw),
                )
            )
            # kappa(s) for the controller: [length (mm), curvature (1/mm),
            # direction (1: forward, -1: backward)] per segment, starting at the
            # current rear axle position
            segments_msg.layout.dim = [
                std_msgs.msg.MultiArrayDimension(
                    label="segments", size=len(target.path.segments), stride=3
                ),
                std_msgs.msg.MultiArrayDimension(
                    label="length,curvature,direction", size=3, stride=1
                ),
            ]
            segments_msg.data = [
                float(value)
                for segment in target.path.segments
                for value in (segment.length, segment.curvature, segment.direction)
            ]
            self.get_logger().debug(
                f"Target spot #{target.track.id} ({target.track.kind}), "
                f"{target.path.name}, {target.path.length:.0f} mm"
            )

        self.path_publisher.publish(path_msg)  # type: ignore
        self.segments_publisher.publish(segments_msg)  # type: ignore


def to_pose(x: float, y: float, yaw: float) -> geometry_msgs.msg.Pose:
    """Create a pose message in the plane.

    Args:
        x: Position in mm.
        y: Position in mm.
        yaw: Orientation in rad.

    Returns:
        The pose with the orientation as quaternion.
    """
    pose = geometry_msgs.msg.Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.orientation.z = math.sin(yaw / 2)
    pose.orientation.w = math.cos(yaw / 2)
    return pose


def main(args: list[str] | None = None) -> None:
    """Main function to start the ParkingPerceptionNode.

    Args:
        args: Launch arguments (default: None)
    """
    rclpy.init(args=args)
    node = ParkingPerceptionNode()

    # We have 2 options on how to run the node:
    # 1. Let the node idle in the background with 'rclpy.spin(node)' if we want to let
    #   subscriber callback function handle the execution of our code.
    #   TODO: is it possible in this way, that our callback gets executed multiple times
    #       in parallel?
    # 2. Run the node in a while loop that waits for incoming messages and then executes
    #   our code. This makes sure that always the latest message is processed and never
    #   multiple messages in parallel. It should be used if the processing of the
    #   message/execution of our code takes longer than the time between incoming #
    #   messages.

    try:
        use_wait_for_message = True
        if use_wait_for_message:
            while rclpy.ok():
                rclpy.spin_once(node)
        else:
            rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        node.destroy_node()

        # Shutdown if not already done by the ROS2 launch system
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
