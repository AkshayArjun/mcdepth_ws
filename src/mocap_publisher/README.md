# OAK-D ROS 2 Mocap Publisher

This module provides a standalone ROS 2 node that uses an OAK-D RGB camera and Google MediaPipe to track a human's upper body skeleton and publish the geometry payload to ROS 2 topics. 

It tracks the user's right arm (shoulder, elbow, wrist) and hand orientation (quaternion), and calculates continuous scaling vectors to map the human's physical arm length correctly to a robotic arm model.

## Requirements & Environment

This node requires a Python environment with ROS 2 (Humble) and several computer vision dependencies installed.

### Dependencies
- **ROS 2 Humble** (via `rclpy`)
- **Python 3.10+**
- `depthai` (for OAK-D camera interfacing)
- `mediapipe` (for skeletal tracking)
- `opencv-python` (`cv2` for image processing & visualization)
- `numpy` (for coordinate matrix math)

If you are using conda, activate your ROS 2 environment:
```bash
conda activate ros2_humble
```

## How to Run the Camera Tracker

1. Ensure your OAK-D camera is plugged in via USB/Type-C adapter to your computer.
2. Navigate to the `mocap_publisher` directory.
3. If you intend to transmit the skeletal data to another computer over the same Wi-Fi network (rather than just the local machine), you must configure your DDS network environment variables before running:

```bash
# Set a shared domain ID for your specific robot/project group to prevent network collisions
export ROS_DOMAIN_ID=144 

# Allow data to leave the computer over the local network
export ROS_LOCALHOST_ONLY=0

# Run the tracker node
python3 camera_tracker.py
```

## Calibration Process

When the script launches, a camera preview window titled "Mocap Camera Tracker" will open.

The system begins in **CALIBRATION** mode. For the tracker to begin publishing data, you must perform a T-Pose calibration:
1. Step back so your entire upper body is visible inside the "TRACKING ZONE".
2. Stretch both of your arms straight out horizontally to the sides (T-Pose).
3. Hold perfectly still. A green progress bar will appear and fill up (takes ~3 seconds).
4. Once the bar fills, the system calculates your arm's scale factors (`scale_upper` and `scale_fore`).

The screen will then display **"Tracking Active"**. From this point forward, the node is actively publishing your skeletal data.

## Subscribing to the Data

You can verify that data is streaming successfully by echoing the ROS 2 topic in another terminal window. *Be sure to match the same network variables.*

```bash
conda activate ros2_humble
export ROS_DOMAIN_ID=144
export ROS_LOCALHOST_ONLY=0

ros2 topic echo /human/skeletal_data
```

### Published Topics

1. `/human/skeletal_data` (`geometry_msgs/PoseArray`)
   - Emits an array of exactly 3 Poses at 15Hz.
   - **Pose 0**: Shoulder anchor (origin, `[0, 0, 0]`)
   - **Pose 1**: Elbow position relative to shoulder
   - **Pose 2**: Wrist position relative to shoulder, plus Hand Orientation Quaternion (`[x, y, z, w]`).
   - Coordinate conversions have been applied (MuJoCo/ROS standard: `X`=forward, `Y`=left, `Z`=up). Z values are passed through a modulus explicitly to ensure they are strictly positive.

2. `/human/gripper_cmd` (`std_msgs/Float64`)
   - Calculates the distance between the tracked thumb tip and index finger tip.
   - Publishes a normalized float from `0.0` (closed pinch) to `1.0` (fully open).

3. `/mocap/state` (`std_msgs/String`)
   - Repeatedly publishes `"CALIBRATION"` or `"TRACKING"` to indicate the current phase of the script.

## Controls

While the camera preview window is in focus:
- Press **`r`** to reset the calibration and force the system back into T-Pose detection mode.
- Press **`q`** to cleanly shutdown the node and close the camera stream.
