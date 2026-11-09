# Parking

[![Build](https://github.com/DHBW-Smart-Rollerz/parking/actions/workflows/build.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/parking/actions/workflows/build.yaml) [![Test](https://github.com/DHBW-Smart-Rollerz/parking/actions/workflows/test.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/parking/actions/workflows/test.yaml) [![Lint](https://github.com/DHBW-Smart-Rollerz/parking/actions/workflows/lint.yaml/badge.svg)](https://github.com/DHBW-Smart-Rollerz/parking/actions/workflows/lint.yaml)

This package contains the AI-based parking spot detection of the Smarty vehicle. It replaces the classic detection in `parking_spot`.

## Usage

Build the package:

```bash
colcon build --symlink-install --packages-select parking
```

Start the parking perception:

```bash
ros2 launch parking parking_perception.launch.py
```

The following launch arguments exist:

- `params_file`: Path to the ROS parameters file (default: `config/ros_params.yaml`).
- `debug`: Enable debug mode to publish the debug image (default: `False`).

## Pipeline

```text
camera_preprocessing ──/camera/image/bev─────────▶ parking_perception_node
pose_estimation      ──/pose_estimation/pose─────▶        │
                                                          ├─ ParkingSpotModel.predict()  → list[ParkingSpot]
                                                          ├─ ParkingPipeline.update()    → target spot + path
                                                          ├─ /parking/path, /parking/target_pose, /parking/path_segments
                                                          └─ /parking/debug_image (debug only)
```

The prototype of the parking pipeline after the detection (rectangle fit, tracking, checks of the spots and geometric path planning) is described in detail in [INFO_Prototyp.md](INFO_Prototyp.md) (German).

- The node uses the `CoordinateTransform` of `camera_preprocessing` (calibration `calib.bin` of its share directory) to convert between pixels of the image and the vehicle frame in mm. `parking/perception/view.py` selects the conversion of the configured view (parameter `view`). The calibration has to exist, i.e. `camera_preprocessing` must be calibrated and built.
- `parking/perception/model.py`: Runs the network and converts the detected corners to the vehicle frame.
- `parking/perception/parking_spot.py`: `ParkingSpot` with the corners in the vehicle frame (mm). `to_image()` projects them into the image, `to_global()` transforms them with the vehicle pose into the global frame.
- `parking/perception/visualization.py`: Draws the spots into the debug image.

- `parking/pipeline.py`: `ParkingPipeline` turns the detected spots into a target spot and a path (independent of ROS): `perception/geometry.py` (rectangle fit), `perception/tracker.py` (tracking), `planning/` (vehicle geometry, checks, path planning).

The model path and the score threshold are defined in `config/model.yaml`, the parking spot and vehicle geometry, the tracking and the path planning in `config/planning.yaml`.

## Parking spot detection network

The detection uses a small PyTorch network (`parking/perception/network.py`) in the style of CenterNet: For every parking spot it predicts the center as heatmap peak (one heatmap per class: free, occupied, blocked) and the offsets of the four corners. Without a trained model (`models/parking_net.pt`) the node runs but detects no parking spots.

Workflow:

1. **Extract frames** from a rosbag. They are preprocessed like in `camera_preprocessing` (bird's eye view by default, `--view undistorted` for the undistorted image):

   ```bash
   python3 src/parking/tools/extract_frames.py <bag> src/parking/data/frames --every 5
   ```

2. **Annotate** the frames in CVAT: one polygon with four points per parking spot, the label is the class (e.g. `Parking_90_free`). The order of the points does not matter, they are sorted into bottom-left, bottom-right, top-right, top-left when loading. Export as "CVAT for images 1.1" with images to `data/dataset_v1/`.

3. **Train** with `notebooks/train_parking_net.ipynb` (dependencies: `pip install -r requirements-training.txt`). The best model is saved to `models/parking_net.pt`.

4. **Build** the package again (`colcon build --packages-select parking`), so the model is installed, and start the node.

### Alternative: YOLO pose

`notebooks/train_yolo_pose.ipynb` trains a YOLO pose network of [Ultralytics](https://docs.ultralytics.com/tasks/pose/) on the same CVAT annotations: every parking spot is one object with its four corners as keypoints. The notebook exports the annotations to `data/yolo/<dataset>/`, trains (runs in `runs/pose/`) and copies the best model to `models/yolo_pose.pt`.

The detector of the node is selected in `config/model.yaml` (`detector: 'parking_net'` or `'yolo_pose'`). Dependencies: `pip install -r requirements-yolo.txt` (without `-U`, numpy 2.x breaks `cv_bridge`). For pre-annotation in CVAT use `tools/cvat_function_yolo.py` instead of `tools/cvat_function.py`.

`pip install torch` installs the CUDA version (several GB). Without an NVIDIA GPU use `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Settings

The parameters of the `parking_perception_node` are defined in `config/ros_params.yaml`:

- **model_config_path**: Path to the model configuration incl. the selected detector (default: `"config/model.yaml"`).
- **planning_config_path**: Path to the planning configuration (default: `"config/planning.yaml"`).
- **use_pose**: Track the spots in the global frame of `pose_subscriber`, if a pose is received (default: `true`).
- **frame_id**: Frame id of the published path and target pose (default: `"vehicle"`).
- **view**: View of the detector, the model has to be trained on it (default: `"bev"`): `"bev"` (bird's eye view of camera_preprocessing, `image_subscriber: /camera/image/bev`), `"undistorted"` (`image_subscriber: /camera/image/undistorted`) or `"parking_bev"` (larger bird's eye view of this package, section `parking_bev` of `config/model.yaml`, computed in the node from `image_subscriber: /camera/image/undistorted`).
- **image_subscriber**: Topic of the preprocessed camera image (default: `"/camera/image/bev"`, undistorted: `"/camera/image/undistorted"`).
- **pose_subscriber**: Topic of the vehicle pose (default: `"/pose_estimation/pose"`).
- **debug_image_publisher**: Topic of the debug image (default: `"/parking/debug_image"`).
- **path_publisher**: Path of the rear axle into the target spot as `nav_msgs/Path` in mm, empty without target (default: `"/parking/path"`).
- **target_publisher**: Goal pose of the rear axle as `geometry_msgs/PoseStamped` (default: `"/parking/target_pose"`).
- **segments_publisher**: Path as `std_msgs/Float32MultiArray` with `[length (mm), curvature (1/mm), direction (±1)]` per segment (default: `"/parking/path_segments"`).

## Structure

- `.github/workflows/`: CI pipelines, which call the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows):
  - `build.yaml`: Builds the package
  - `test.yaml`: Runs the tests of the package
  - `lint.yaml`: Checks the code with ruff (Python) and ament_uncrustify (C++)
- `config/`: All configurations (most of the time yaml files)
- `launch/`: Contains all launch files. Launch files can start multiple nodes with yaml-configurations
- `models/`: Contains the trained network `parking_net.pt`
- `notebooks/`: Training notebook
- `tools/`: Helper scripts, e.g. extracting frames from rosbags or running the whole pipeline offline on a rosbag (`run_pipeline_on_bag.py`)
- `resource/`: Contains the package name (required to build with colcon)
- `parking`: Contains all nodes and sources for the ros package
- `test/`: Contains all tests
- `package.xml`: Contains metadata about the package
- `setup.py`: Used for Python package configuration
- `setup.cfg`: Additional configuration for the package
- `requirements.txt`: Python dependencies

## Contributing

Thank you for considering contributing to this repository! Here are a few guidelines to get you started:

1. Fork the repository and clone it locally.
2. Create a new branch for your contribution.
3. Make your changes and ensure they are properly tested.
4. Commit your changes and push them to your forked repository.
5. Submit a pull request with a clear description of your changes.

We appreciate your contributions and look forward to reviewing them!

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.
