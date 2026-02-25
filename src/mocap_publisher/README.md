# Mocap Publisher — OAK-D + MediaPipe → ROS2

> **Author:** Shreya Shah
> **Part of:** `mcdepth_ws` — collaborative depth-based motion capture workspace

---

## Overview

This package captures human arm motion using an **OAK-D camera + MediaPipe** and publishes the joint positions as **ROS2 topics** so the robot retargeting node can subscribe.

```
OAK-D Camera
    │
    ▼
camera_tracker.py          ← captures shoulder/elbow/wrist from MediaPipe
    │  UDP (port 5005)
    ▼
mocap_pub.py               ← THIS PACKAGE — bridges UDP → ROS2
    │  ROS2 topics
    ▼
Your partner's subscriber  ← consumes /mocap/joints etc.
    │
    ▼
mujoco_retargeting.py      ← runs Analytical IK → drives RX200 in MuJoCo
```

---

## ROS2 Topics Published

| Topic | Type | Description |
|-------|------|-------------|
| `/mocap/joints` | `geometry_msgs/PoseArray` | `[0]`=shoulder, `[1]`=elbow, `[2]`=wrist (3D XYZ) |
| `/mocap/hand_orientation` | `geometry_msgs/QuaternionStamped` | Hand roll/pitch/yaw quaternion |
| `/mocap/state` | `std_msgs/String` | `"TRACKING"` or `"CALIBRATION"` |

All positions are in **MuJoCo world frame** (X=right, Y=forward, Z=up).

---

## 5-DOF Biometric Calibration

### ReactorX 200 Joint Mapping

| # | Robot Joint | Human Motion |
|---|-------------|--------------|
| 0 | **waist** | Shoulder Yaw (left/right) |
| 1 | **shoulder** | Shoulder Pitch (up/down) |
| 2 | **elbow** | Elbow Flexion |
| 3 | **wrist_angle** | Wrist Pitch |
| 4 | **wrist_rotate** | Wrist Roll (from MediaPipe Hands quaternion) |

### T-Pose Calibration Protocol

On launch, the user holds a **T-Pose** (both arms stretched straight out to the sides, parallel to the floor) for ~2 seconds.

MediaPipe measures the user's actual arm segment lengths:
- `upper_arm_len` = distance(shoulder → elbow) in metres
- `forearm_len` = distance(elbow → wrist) in metres

Scale factors are then computed to map the user's proportions to the robot's:
```python
# Robot link lengths (from rx200_mujoco.xml)
ROBOT_UPPER_ARM = 0.200  # shoulder → elbow
ROBOT_FOREARM   = 0.265  # elbow → wrist + gripper

scale_upper = ROBOT_UPPER_ARM / upper_arm_len
scale_fore  = ROBOT_FOREARM   / forearm_len
```

### Per-Frame Retargeting Math

```python
# Scale human direction vectors to robot workspace
elbow_vec = (h_elbow - h_shoulder) * scale_upper
wrist_vec = (h_wrist  - h_elbow)   * scale_fore

# Unit vectors drive analytical IK
upper_dir = elbow_vec / |elbow_vec|
fore_dir  = wrist_vec / |wrist_vec|

# Two-link analytical IK → waist, shoulder, elbow, wrist_angle
# Wrist rotate ← hand quaternion roll from MediaPipe Hands
```

---

## Files

| File | Description |
|------|-------------|
| `mocap_publisher/mocap_pub.py` | **ROS2 publisher node** — UDP → ROS2 bridge |
| `mocap_publisher/camera_tracker.py` | OAK-D capture + MediaPipe pose + T-pose calibration |
| `mocap_publisher/mujoco_retargeting.py` | Analytical IK + MuJoCo simulation |
| `mocap_publisher/extract_3d_joints.py` | 3D joint extraction utility |
| `mocap_publisher/visualize_3d_joints.py` | 3D skeleton visualizer |
| `mocap_publisher/simulate_robot_arm.py` | Standalone robot arm simulator |
| `rx200_mujoco.xml` | RX200 robot MJCF model |
| `scene.xml` | MuJoCo scene configuration |
| `rx200.urdf` | RX200 URDF (for ROS TF / RVIZ) |

---

## How to Run

### Prerequisites
```bash
pip install mediapipe depthai opencv-python numpy mujoco
```

### Step 1 — Build and source the package
```bash
cd ~/mcdepth_ws
colcon build --packages-select mocap_publisher
source install/setup.bash
export ROS_DOMAIN_ID=144
```

### Step 2 — Start the camera tracker (sends ROS2 topics)
```bash
conda activate ros2_humble
export ROS_DOMAIN_ID=144
python3 src/mocap_publisher/mocap_publisher/camera_tracker.py
```
> Hold T-Pose until the progress bar reaches 100% to calibrate.

### Step 3 — Start the ROS2 publisher node (if needing UDP bridge)
```bash
export ROS_DOMAIN_ID=144
ros2 run mocap_publisher mocap_pub_node
```

### Step 4 — Subscribe and verify
```bash
export ROS_DOMAIN_ID=144
ros2 topic echo /human/skeletal_data
```

### Step 5 — Run MuJoCo retargeting (optional local sim)
```bash
python3 src/mocap_publisher/mocap_publisher/mujoco_retargeting.py
```

---

## Coordinate Frame

All joint positions are converted from MediaPipe world frame to **MuJoCo frame** before publishing:

```python
# MediaPipe world → MuJoCo
mujoco_vec = [mp_x, -mp_z, -mp_y]
# X = right, Y = forward, Z = up
```
