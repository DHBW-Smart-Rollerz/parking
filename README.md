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

## Settings

The parameters of the `parking_perception_node` are defined in `config/ros_params.yaml`:

- **model_config_path**: Path to the model configuration (default: `"config/model.yaml"`).
- **image_subscriber**: Topic of the preprocessed camera image (default: `"/camera/image/undistorted"`).
- **pose_subscriber**: Topic of the vehicle pose (default: `"/pose_estimation/pose"`).
- **debug_image_publisher**: Topic of the debug image (default: `"/parking/debug_image"`).

## Structure

- `.github/workflows/`: CI pipelines, which call the shared workflows of the [smarty_workspace](https://github.com/DHBW-Smart-Rollerz/smarty_workspace/tree/jazzy/.github/workflows):
  - `build.yaml`: Builds the package
  - `test.yaml`: Runs the tests of the package
  - `lint.yaml`: Checks the code with ruff (Python) and ament_uncrustify (C++)
- `config/`: All configurations (most of the time yaml files)
- `launch/`: Contains all launch files. Launch files can start multiple nodes with yaml-configurations
- `models/`: Contains all models (optional) and only necessary for machine learning nodes
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
