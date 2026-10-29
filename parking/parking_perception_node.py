import cv_bridge
import geometry_msgs.msg
import rclpy
import sensor_msgs.msg
from camera_preprocessing.transformation.coordinate_transform import (
    CoordinateTransform,
)
from rclpy.executors import ExternalShutdownException
from smarty_utils.enums import NodeState
from smarty_utils.smarty_node import SmartyNode
from timing import timer

from parking.perception import model, visualization


class ParkingPerceptionNode(SmartyNode):
    """Detects parking spots in the camera image."""

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
                # Parameters
                "model_config_path": "config/model.yaml",
                "birds_eye": True,
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

        self.model = model.ParkingSpotModel(
            root=self.package_path,
            model_config_path=self.model_config_path,
            transform=self.transform,
            birds_eye=self.birds_eye,
        )

        self.get_logger().info("ParkingPerceptionNode initialized")

    @property
    def model_config_path(self) -> str:
        """Return the model config path."""
        return self.get_parameter("model_config_path").value  # type: ignore

    @property
    def birds_eye(self) -> bool:
        """Return whether the subscribed image is in the bird's eye view."""
        return self.get_parameter("birds_eye").value  # type: ignore

    def _reset(self) -> None:
        """Reset the node to its initial state."""
        del self.model
        self.latest_pose = None
        self.model = model.ParkingSpotModel(
            root=self.package_path,
            model_config_path=self.model_config_path,
            transform=self.transform,
            birds_eye=self.birds_eye,
        )

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

        with timer.Timer(name="prediction", filter_strength=40):
            # Get the detected parking spots in the vehicle frame
            spots = self.model.predict(image)

        with timer.Timer(name="publish", filter_strength=40):
            # TODO: Publish the detected parking spots (parking_msgs/ParkingSpotArray)
            self.get_logger().debug(f"Detected {len(spots)} parking spot(s)")
            if self.latest_pose is not None:
                for spot in spots:
                    self.get_logger().debug(
                        f"Spot center (global): {spot.to_global(self.latest_pose)}"
                    )

        if self._debug:
            with timer.Timer(name="debug", filter_strength=40):
                debug_image = visualization.draw_spots(
                    image, spots, self.transform, birds_eye=self.birds_eye
                )

                self.debug_image_publisher.publish(  # type: ignore
                    self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
                )

        timer.Timer(logger=self.get_logger().info).print()  # type: ignore


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
