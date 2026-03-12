# MonkeySee_MonkeyDo 
## Human-to-Robot Arm Motion Retargeting

Real-time teleoperation of **any URDF-described robot arm** by retargeting human arm motion captured via an **OAK-D depth camera** and **MediaPipe** pose estimation. Implements a modified version of the OCRA algorithm (Optimization-based Customizable Retargeting Algorithm) from Mohan & Kuchenbecker, ICRA Workshop 2023 [[1]](#references), with forward kinematics and gradient computation powered by **JAX** [[2]](#references) and robot-agnostic kinematic modelling via **PyRoki** [[3]](#references).

---

## Overview

The system maps the human operator's arm configuration to the robot's joint space in real time. Three components run concurrently over ROS2:

1. **`camera_tracker.py`** — captures RGB-D frames from an OAK-D camera, runs MediaPipe pose estimation to extract 3D skeleton landmarks, and publishes shoulder / elbow / wrist positions as a `PoseArray`.
2. **`ocra_node.py`** (or `ocra_sim_node.py`) — receives the skeleton, solves the OCRA optimisation at each control step, and sends joint commands to the robot.
3. **`ocra_visualizer.html`** — browser-based live visualiser (via rosbridge) rendering both chains in 3D with FK computed in JavaScript.

---

## Full Pipeline

### Conceptual flow

```
 HUMAN OPERATOR
 ─────────────────────────────────────────────────────────────────
                                                
  o   <── shoulder keypoint (px,py,pz)          
  │                                              
  │   <── elbow keypoint   (px,py,pz)            
  │                                              
  └─  <── wrist keypoint   (px,py,pz)            
                                                
       captured by OAK-D stereo depth camera
       estimated by MediaPipe BlazePose [4]
                                                
 ─────────────────────────────────────────────────────────────────
                         │  PoseArray
                         │  /human/skeletal_data
                         ▼
 ╔═══════════════════════════════════════════════╗
 ║           camera_tracker.py                  ║
 ║                                              ║
 ║  · OAK-D pipeline (DepthAI)                  ║
 ║  · MediaPipe Pose (BlazePose backbone)        ║
 ║  · Arm length normalisation                  ║
 ║    → scale human arm to robot proportions    ║
 ║  · Gripper pinch detection                   ║
 ║    → hysteresis + majority-vote debounce     ║
 ╚═══════════════════════════════════════════════╝
                         │
           ┌─────────────┴──────────────┐
           │                            │
           ▼                            ▼
   /human/skeletal_data        /human/gripper_cmd
   (geometry_msgs/PoseArray)   (std_msgs/Bool)
           │                            │
           ▼                            ▼
 ╔══════════════════════════════════════════════════════╗
 ║                  ocra_node.py                       ║
 ║                                                     ║
 ║  ┌──────────────────────────────────────────────┐   ║
 ║  │  OCRA Loss  L(q) = α·εs² + β·εo² + γ·εee²  │   ║
 ║  └──────────────────────────────────────────────┘   ║
 ║                                                     ║
 ║  Forward Kinematics ──► PyRoki [3]                  ║
 ║    · loads robot from URDF                          ║
 ║    · computes FK for any kinematic chain            ║
 ║    · robot-agnostic: swap URDF → new robot          ║
 ║                                                     ║
 ║  Gradients ──► JAX value_and_grad [2]               ║
 ║    · automatic differentiation of OCRA loss        ║
 ║    · JIT-compiled for real-time performance         ║
 ║                                                     ║
 ║  Solver ──► SLSQP (scipy.optimize)                  ║
 ║    · warm-started from previous solution            ║
 ║    · joint limits enforced as box constraints       ║
 ╚══════════════════════════════════════════════════════╝
           │                            │
           ▼                            ▼
   /ocra/commanded_joints      gripper action client
   (sensor_msgs/JointState)    (/gripper_controller/…)
           │
           ▼
 ╔══════════════════════════════╗
 ║    Robot Controller          ║
 ║                              ║
 ║  · FollowJointTrajectory     ║
 ║    action server             ║
 ║  · single-point PTP goal     ║
 ║  · joint impedance control   ║
 ╚══════════════════════════════╝
           │
           ▼
    ROBOT ARM MOVES
```

### Chain retargeting — what the optimiser sees

At each 10–20 Hz control step the OCRA loss treats both arms as abstract chains and finds the robot joint configuration `q` that best mirrors the human pose:

```
  Human arm (source)          Robot arm (target)

    ●  shoulder                 ●  base / shoulder
    │                           │
    │  seg_1                    │  seg_1
    │                           │
    ●  elbow          ───►      ●  elbow proxy
    │                           │
    │  seg_2                    │  seg_2
    │                           │
    ●  wrist/hand               ●  end-effector (EE)

   chain H = {sh, el, ha}      chain R = {base, el_r, EE}

   ε_s : bidirectional chain-to-chain distance (normalised)
   ε_o : EE orientation error  (quaternion, arctan2 form)
   ε_ee: direct hand ↔ EE Euclidean distance  (extension)
```

### Why PyRoki makes this robot-agnostic

Previously, the forward kinematics were hand-coded as a Product-of-Exponentials (PoE) function specific to the rx200's screw axes and home configuration. This means porting to a new robot required re-deriving the entire FK model.

With PyRoki [[3]](#references), the robot is described purely by its **URDF** file. PyRoki parses the URDF into a JAX-differentiable kinematic tree at startup. The OCRA optimiser calls `robot.forward_kinematics(q)` and PyRoki handles the rest — screw axes, joint ordering, link transforms — automatically. Swapping to a different robot arm (e.g. from the rx200 to the Addverb Heal 6R) requires only pointing at a different URDF.

```
  Old approach (robot-specific):
    hand-coded screw axes S₁…S₅  ──►  FK function for rx200 only

  New approach (robot-agnostic via PyRoki):
    URDF file  ──►  pyroki.Robot.from_urdf()
                         │
                         ▼
                    JAX FK tree  ──►  any robot arm
```

---

## Algorithm — OCRA

The loss function minimised over joint angles `q` at each control step:

```
L(q) = α · ε_s² + β · ε_o² + γ · ε_ee²
```

`ε_s` and `ε_o` follow the original paper [[1]](#references). `ε_ee` is our extension described below.

---

### Skeleton Error `ε_s`

Bidirectional chain-to-chain distance, normalised by total arm length:

```
ε_s = ( Σ s_i + Σ t_j ) / ℓ
```

- `s_i` = distance from human joint `i` to the nearest point on the robot chain
- `t_j` = distance from robot joint `j` to the nearest point on the human chain
- `ℓ = (human_seg1 + human_seg2) + (robot_seg1 + robot_seg2)` — total arc length of both chains, normalises `ε_s` to `[0, 1]`

Point-to-chain distance uses a **smooth minimum** (log-sum-exp) instead of `min()` to preserve differentiability:

```python
# smooth min over two segment distances d1, d2
-(1/α) · log( exp(-α·d1) + exp(-α·d2) )    # α = 10
```

---

### Orientation Error `ε_o`

Axis-angle magnitude of the relative end-effector rotation, normalised to `[0, 1]`:

```
Q_d  = q_robot · q_target⁻¹          # relative rotation quaternion
ε_o  = 2 · arctan2(|xyz|, |w|) / π   # arctan2 stable at identity
```

`arctan2` is used instead of `arccos` because `arccos(1.0)` has an undefined gradient — this was the source of NaN gradients at the identity rotation.

---

### End-Effector Matching Term `ε_ee` *(extension, not in paper)*

The skeleton error `ε_s` alone creates a degenerate local minimum where the robot EE parks near the human **elbow** (satisfying the chain distance term) while the human **hand** is ignored entirely. To fix this, an explicit EE correspondence term is added:

```python
hand_err = ‖ r_EE - t_hand ‖ / (ℓ + 1e-8)
```

This directly penalises the Euclidean distance between the robot end-effector and the human hand position, forcing the optimiser to treat hand-to-EE correspondence as a hard goal rather than an incidental benefit of chain alignment. Normalised by `ℓ` to remain on the same scale as `ε_s`.

```python
# Full loss
return alpha * (skel_err ** 2) + beta * (orient_err ** 2) + gamma * (hand_err ** 2)
```

---

### Optimisation

SLSQP (Sequential Least Squares Programming) via `scipy.optimize.minimize`, with analytical gradients computed by JAX `value_and_grad` [[2]](#references). Warm-started from the previous solution after the first solve.

```python
loss_and_grad_fn = jax.value_and_grad(ocra_loss)
```

---

## Repository Structure

```
robot_retarget/
├── rx200_kinematics.py     # JAX OCRA loss + value_and_grad (PyRoki FK)
├── ocra_sim_node.py        # ROS2 node → Gazebo (JointTrajectoryController)
├── ocra_node.py            # ROS2 node → hardware (action client)
├── camera_tracker.py       # OAK-D + MediaPipe → PoseArray publisher
├── fake_skele_pub.py       # Synthetic skeleton for testing without camera
└── ocra_visualizer.html    # Live browser visualiser (via rosbridge)
```

---

## Dependencies

### ROS2 & Robot

- ROS2 Humble
- `rosbridge_suite` — for live visualiser
- Robot-specific driver (e.g. `interbotix_xsarm_control` for rx200, or manufacturer ROS2 package for other arms)

```bash
sudo apt install ros-humble-rosbridge-suite
```

### Python

```bash
# Core
pip install jax jaxlib scipy numpy --break-system-packages

# PyRoki — robot-agnostic FK and kinematic optimisation
pip install pyroki --break-system-packages

# Camera tracker only
pip install depthai mediapipe --break-system-packages
```

---

## Setup & Launch

### Simulation

```bash
# Terminal 1 — Gazebo (example: rx200)
ros2 launch interbotix_xsarm_sim xsarm_gz_classic.launch.py robot_model:=rx200

# Terminal 2 — OCRA controller
ros2 run robot_retarget ocra_sim_node

# Terminal 3 — Skeleton source (pick one)
python3 fake_skele_pub.py        # synthetic test data
python3 camera_tracker.py        # real OAK-D camera
```

### Hardware

```bash
# Terminal 1 — Robot driver
ros2 launch <your_robot_bringup> ...

# Terminal 2
ros2 run robot_retarget ocra_node

# Terminal 3
python3 camera_tracker.py
```

### Live Visualiser

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Open `ocra_visualizer.html` in a browser. Toggle **SIM / HARDWARE** in the top bar.

| Mode | Topics subscribed |
|------|-------------------|
| SIM | `/human/skeletal_data`, `/rx200/commands/joint_group` |
| HW | `/human/skeletal_data`, `/<robot>/joint_states`, `/ocra/commanded_joints` |

In HW mode two FK chains are rendered: **cyan** = where the robot actually is (`/joint_states`), **orange dashed** = where OCRA commanded it (`/ocra/commanded_joints`). Drag to rotate, scroll to zoom. EE glow: 🟢 good / 🟡 ok / 🔴 bad loss.

---

## Porting to a New Robot

Because FK is now delegated to PyRoki, changing the target robot requires only:

1. Place the robot URDF somewhere accessible (e.g. installed via its ROS2 description package).
2. Update `URDF_PATH` in `rx200_kinematics.py` to point at the new URDF.
3. Update `JOINT_NAMES` and `BOUNDS` in `ocra_node.py` to match the new robot's actuated joints and limits.
4. Tune `ALPHA`, `BETA`, `GAMMA` for the new arm proportions.

No FK code changes are required.

---

## Key Parameters

| Parameter | Location | Value | Description |
|---|---|---|---|
| `LOOP_RATE` | `ocra_node.py` | 10–20 Hz | Control loop frequency |
| `ALPHA` | `ocra_node.py` | 0.67 | Skeleton error weight `α` |
| `BETA` | `ocra_node.py` | 0.33 | Orientation error weight `β` |
| `GAMMA` | `ocra_node.py` | 2×α | EE matching weight `γ` |
| `maxiter` (first solve) | `ocra_node.py` | 50 | SLSQP iterations, cold start |
| `maxiter` (warm start) | `ocra_node.py` | 10 | SLSQP iterations, subsequent |
| `ftol` | `ocra_node.py` | 1e-4 | Optimizer convergence tolerance |

---

## Demo 

![Demonstration Video](media/demonstration.mp4)

[Direct link to video](https://github.com/AkshayArjun/MonkeySee_MonkeyDo/raw/master/media/demonstration.mp4)



## Known Limitations

- **IK redundancy** — the optimiser can find kinematically valid but visually unnatural solutions when the target is reachable by a contorted configuration. The `ε_ee` term significantly mitigates the worst case (robot EE parking at human elbow).
- **Workspace mismatch** — targets outside the robot's reachable workspace cause the optimiser to push joints to their limits. The node filters skeleton frames where `|hand| > reach_limit`.
- **Orientation term at identity** — `arccos(1.0)` has an undefined gradient. Replaced with `arctan2(|xyz_norm|, |w|)` which is numerically stable everywhere.
- **Single-arm only** — currently tracks the right arm kinematic chain. Left arm and full-body retargeting are not implemented.
- **PyRoki kinematic trees only** — parallel manipulators and closed-loop mechanisms are not supported by PyRoki [[3]](#references).

---

## References

**[1]** Mohan, M. and Kuchenbecker, K. J. (2023). *OCRA: An Optimization-Based Customizable Retargeting Algorithm for Teleoperation.* Workshop paper presented at the ICRA Workshop "Toward Robot Avatars", London, UK, May 2023. Max Planck Institute for Intelligent Systems, Haptic Intelligence Dept. URL: https://www.ais.uni-bonn.de/ICRA2023AvatarWS/contributions/ICRA_2023_Avatar_WS_Mohan.pdf

```bibtex
@misc{Mohan23-ICRAWS-OCRA,
  title   = {{OCRA}: An Optimization-Based Customizable Retargeting
             Algorithm for Teleoperation},
  author  = {Mohan, Mayumi and Kuchenbecker, Katherine J.},
  howpublished = {Workshop paper presented at the ICRA Workshop
                  Toward Robot Avatars},
  address = {London, UK},
  month   = may,
  year    = {2023},
  url     = {https://www.ais.uni-bonn.de/ICRA2023AvatarWS/contributions/
             ICRA_2023_Avatar_WS_Mohan.pdf}
}
```

---

**[2]** Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., and Zhang, Q. (2018). *JAX: composable transformations of Python+NumPy programs.* Google. URL: https://github.com/jax-ml/jax

```bibtex
@software{jax2018github,
  author  = {James Bradbury and Roy Frostig and Peter Hawkins and
             Matthew James Johnson and Chris Leary and Dougal Maclaurin
             and George Necula and Adam Paszke and Jake Vander{P}las and
             Skye Wanderman-{M}ilne and Qiao Zhang},
  title   = {{JAX}: composable transformations of
             {P}ython+{N}um{P}y programs},
  url     = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year    = {2018}
}
```

---

**[3]** Kim\*, C. M., Yi\*, B., Choi, H., Ma, Y., Goldberg, K., and Kanazawa, A. (2025). *PyRoki: A Modular Toolkit for Robot Kinematic Optimization.* In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). arXiv:2505.03728. URL: https://pyroki-toolkit.github.io

```bibtex
@inproceedings{kim2025pyroki,
  title     = {{PyRoki}: A Modular Toolkit for Robot Kinematic
               Optimization},
  author    = {Kim*, Chung Min and Yi*, Brent and Choi, Hongsuk and
               Ma, Yi and Goldberg, Ken and Kanazawa, Angjoo},
  booktitle = {2025 IEEE/RSJ International Conference on Intelligent
               Robots and Systems (IROS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.03728}
}
```

---

**[4]** Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., and Grundmann, M. (2019). *MediaPipe: A Framework for Building Perception Pipelines.* Google. arXiv:1906.08172. URL: https://arxiv.org/abs/1906.08172

```bibtex
@misc{Lugaresi2019MediaPipe,
  title     = {{MediaPipe}: A Framework for Building Perception
               Pipelines},
  author    = {Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon
               and McClanahan, Chris and Uboweja, Esha and Hays,
               Michael and Zhang, Fan and Chang, Chuo-Ling and Yong,
               Ming Guang and Lee, Juhyun and Chang, Wan-Teh and Hua,
               Wei and Georg, Manfred and Grundmann, Matthias},
  year      = {2019},
  publisher = {arXiv},
  doi       = {10.48550/arXiv.1906.08172}
}
```
