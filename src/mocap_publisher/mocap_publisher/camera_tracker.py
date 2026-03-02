"""
camera_tracker.py — OAK-D + MediaPipe → ROS2 Publisher

Topics:
  /human/skeletal_data   (PoseArray)  — shoulder, elbow, wrist + hand quat
  /mocap/state           (String)     — CALIBRATION / TRACKING

Calibration:
  1. T-POSE — auto (hold still, fills green bar)

Keys: Q=quit  R=restart
"""

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String, Float64

ROBOT_UPPER_ARM = 0.20
ROBOT_FOREARM   = 0.265


# ── Coordinate helpers ────────────────────────────────────────────────────────

def vec3(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def to_mujoco(mp_vec):
    """
    MP:     X=camera-right (person-left), Y=down, Z=toward camera (person-fwd)
    MuJoCo: X=forward, Y=left, Z=up
      MuJoCo_X = -MP_Z
      MuJoCo_Y =  MP_X
      MuJoCo_Z = -MP_Y
    """
    return np.array([-mp_vec[2], mp_vec[0], -mp_vec[1]], dtype=np.float32)


def rotation_matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return [float((R[2,1]-R[1,2])*s), float((R[0,2]-R[2,0])*s),
                float((R[1,0]-R[0,1])*s), float(0.25/s)]
    return [0., 0., 0., 1.]


# ── OAK-D pipeline ────────────────────────────────────────────────────────────

def init_oakd():
    pipeline = dai.Pipeline()
    cam  = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setFps(15)
    cam.setPreviewSize(640, 360)
    cam.preview.link(xout.input)
    return pipeline


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class CameraTrackerNode(Node):
    def __init__(self):
        super().__init__('camera_tracker')
        self.pub_joints  = self.create_publisher(PoseArray, '/human/skeletal_data', 10)
        self.pub_state   = self.create_publisher(String,    '/mocap/state', 10)
        self.pub_gripper = self.create_publisher(Float64,   '/human/gripper_cmd', 10)

    def _pose(self, xyz, quat=None):
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(xyz[0]), float(xyz[1]), abs(float(xyz[2]))
        if quat:
            p.orientation.x, p.orientation.y = float(quat[0]), float(quat[1])
            p.orientation.z, p.orientation.w = float(quat[2]), float(quat[3])
        else:
            p.orientation.w = 1.0
        return p

    def publish_joints(self, sh, el, wr, quat):
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'mocap_world'
        msg.poses = [self._pose(sh), self._pose(el), self._pose(wr, quat)]
        self.pub_joints.publish(msg)


    def publish_state(self, s):
        m = String(); m.data = s
        self.pub_state.publish(m)

    def publish_gripper(self, val):
        msg = Float64()
        msg.data = float(val)
        self.pub_gripper.publish(msg)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_guide_frame(frame, h, w):
    """Draw a white rectangle showing the 'stand here' tracking zone."""
    pad_x, pad_y = w // 8, h // 10
    cv2.rectangle(frame,
                  (pad_x, pad_y), (w - pad_x, h - pad_y),
                  (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, "TRACKING ZONE", (pad_x + 8, pad_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_arm_target(frame, h, w, step):
    """Draw a stick-arm guide showing target pose for each calibration step."""
    cx, cy = w // 2, h // 2          # image centre
    slen   = min(w, h) // 5          # segment length in pixels

    shoulder_pt = (cx, cy - slen // 2)

    def arrow(img, p1, p2, col):
        cv2.arrowedLine(img, p1, p2, col, 3, cv2.LINE_AA, tipLength=0.25)

    if step == "TPOSE":
        # Both arms horizontal
        arrow(frame, shoulder_pt, (cx + slen, cy - slen // 2), (0,255,255))
        arrow(frame, shoulder_pt, (cx - slen, cy - slen // 2), (0,255,255))
        cv2.putText(frame, "Hold BOTH arms OUT to sides",
                    (cx - 160, cy + slen), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)


def draw_joint_hud(frame, h, w, waist, shoulder, elbow, wrist_a, wrist_r):
    """Live joint angle overlay in bottom-left corner of camera frame."""
    y0 = h - 130
    cv2.rectangle(frame, (0, y0 - 10), (280, h), (0, 0, 0), -1)
    cv2.putText(frame, "JOINT ANGLES (rad)", (8, y0 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,160,160), 1)
    labels = [
        f"  Waist   : {waist:+.2f}",
        f"  Shoulder: {shoulder:+.2f}",
        f"  Elbow   : {elbow:+.2f}   <- bend arm!",
        f"  Wrist-A : {wrist_a:+.2f}",
        f"  Wrist-R : {wrist_r:+.2f}   <- rotate palm",
    ]
    colors = [
        (255,255,255), (255,255,255),
        (0,200,255),   # highlight elbow in cyan
        (255,255,255),
        (200,255,200),
    ]
    for i, (txt, col) in enumerate(zip(labels, colors)):
        cv2.putText(frame, txt, (8, y0 + 28 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1)


def draw_progress_bar(frame, label, pct, y0, w, color):
    bw = w - 28
    cv2.rectangle(frame, (0, y0), (w, y0 + 72), (0, 0, 0), -1)
    cv2.putText(frame, label, (14, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    cv2.rectangle(frame, (14, y0 + 36), (14 + bw, y0 + 60), (50, 50, 50), -1)
    cv2.rectangle(frame, (14, y0 + 36), (14 + bw * pct // 100, y0 + 60), color, -1)
    cv2.putText(frame, f"{pct}%", (14 + bw + 4, y0 + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CameraTrackerNode()

    print("Initializing OAK-D (RGB-only, 640×360)...")
    pipeline = init_oakd()

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode=False, model_complexity=0, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    L    = mp_pose.PoseLandmark

    # ── State ──────────────────────────────────────────────────────────────
    CALIB_N = 45    # T-pose hold frames (~3 sec @ 15fps)

    state   = "TPOSE"
    c_upper, c_fore = [], []
    scale_upper = scale_fore = 1.0

    cur_upper_dir = None

    last_hand_quat   = [0., 0., 0., 1.]
    last_gripper_val = 1.0
    frame_idx        = 0
    banner_timer    = 0
    BANNER_FRAMES   = 90

    # Live IK angles for HUD
    hud_angles = [0., 0., 0., 0., 0.]   # [waist, shoulder, elbow, wrist_a, wrist_r]

    print("\n[Cal] T-POSE: stretch both arms straight out to the sides.")
    print("      (Hold still for ~3 seconds — auto-detects, then tracking starts)")

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)

            inRgb = q_rgb.tryGet()
            if inRgb is None:
                cv2.waitKey(1)
                continue

            frame = inRgb.getCvFrame()
            h, w  = frame.shape[:2]
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = pose.process(rgb)
            hand_res = hands.process(rgb) if (frame_idx % 2 == 0) else None
            rgb.flags.writeable = True

            # ── Hand orientation ──────────────────────────────────────────
            if hand_res and hand_res.multi_hand_world_landmarks:
                hlm  = hand_res.multi_hand_world_landmarks[0].landmark
                H    = mp_hands.HandLandmark
                wp   = to_mujoco(vec3(hlm[H.WRIST]))
                im   = to_mujoco(vec3(hlm[H.INDEX_FINGER_MCP]))
                pm   = to_mujoco(vec3(hlm[H.PINKY_MCP]))
                fwd  = im - wp
                side = pm - wp
                xax  = fwd  / (np.linalg.norm(fwd)  + 1e-8)
                ztmp = np.cross(side, fwd)
                zax  = ztmp / (np.linalg.norm(ztmp) + 1e-8)
                yax  = np.cross(zax, xax)
                last_hand_quat = rotation_matrix_to_quat(np.column_stack([xax, yax, zax]))
                
                # Gripper pinch (Thumb tip to Index tip distance)
                thumb = vec3(hlm[H.THUMB_TIP])
                index = vec3(hlm[H.INDEX_FINGER_TIP])
                pinch_dist = np.linalg.norm(thumb - index)
                # Map 0.02m (closed) -> 0.08m (open) to a normalized 0.0 to 1.0 range
                last_gripper_val = float(np.clip((pinch_dist - 0.02) / 0.06, 0.0, 1.0))
                draw.draw_landmarks(
                    frame, hand_res.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                    draw.DrawingSpec(color=(0,220,220), thickness=1, circle_radius=2),
                    draw.DrawingSpec(color=(0,180,180), thickness=1))

            # ── Pose joints ───────────────────────────────────────────────
            pose_ok = (pose_res.pose_landmarks and pose_res.pose_world_landmarks)

            if pose_ok:
                draw.draw_landmarks(
                    frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    draw.DrawingSpec(color=(255,120,0), thickness=2, circle_radius=3),
                    draw.DrawingSpec(color=(200, 90,0), thickness=2))

                WL = pose_res.pose_world_landmarks.landmark
                PL = pose_res.pose_landmarks.landmark

                rs = vec3(WL[L.RIGHT_SHOULDER.value])
                re = vec3(WL[L.RIGHT_ELBOW.value])
                rw = vec3(WL[L.RIGHT_WRIST.value])

                rv_wr = PL[L.RIGHT_WRIST.value].visibility
                rv_el = PL[L.RIGHT_ELBOW.value].visibility
                lv_wr = PL[L.LEFT_WRIST.value].visibility

                raw_u = to_mujoco(re) - to_mujoco(rs)
                raw_f = to_mujoco(rw) - to_mujoco(re)
                ul = np.linalg.norm(raw_u)
                fl = np.linalg.norm(raw_f)
                if ul > 1e-4 and fl > 1e-4:
                    cur_upper_dir = raw_u / ul
                    cur_fore_dir  = raw_f / fl

                    # --- Live IK angles for HUD ---
                    ux, uy, uz = cur_upper_dir
                    cos_bend = float(np.clip(np.dot(cur_upper_dir, cur_fore_dir), -1, 1))
                    waist_v    = float(np.arctan2(ux, uy))
                    shoulder_v = float(np.arcsin(np.clip(uz, -1, 1))) - np.pi/2
                    elbow_v    = -np.arccos(cos_bend)
                    wrist_a_v  = -(shoulder_v + elbow_v)
                    qx,qy,qz,qw = last_hand_quat
                    wrist_r_v  = float(np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)))
                    hud_angles = [waist_v, shoulder_v, elbow_v, wrist_a_v, wrist_r_v]

            # ── Guide frame (always shown) ────────────────────────────────
            draw_guide_frame(frame, h, w)

            # ════ STATE: TPOSE ════════════════════════════════════════════
            if state == "TPOSE":
                node.publish_state("CALIBRATION")
                is_t = (pose_ok and rv_wr > 0.5 and rv_el > 0.5 and lv_wr > 0.5
                        and abs(re[1] - rs[1]) < 0.12) if pose_ok else False
                pct = int(len(c_upper) * 100 / CALIB_N)
                draw_progress_bar(frame, "T-POSE — stretch BOTH arms out", pct, 0, w, (0, 255, 255))
                draw_arm_target(frame, h, w, "TPOSE")
                if is_t:
                    cv2.putText(frame, "GOOD — hold still!", (14, 68),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    c_upper.append(ul)
                    c_fore.append(fl)
                    if len(c_upper) >= CALIB_N:
                        ua = float(np.median(c_upper))
                        fa = float(np.median(c_fore))
                        scale_upper = ROBOT_UPPER_ARM / ua
                        scale_fore  = ROBOT_FOREARM   / fa
                        state = "TRACKING"
                        banner_timer = BANNER_FRAMES
                        node.get_logger().info(
                            f"T-Pose ✓  upper={ua*100:.1f}cm  fore={fa*100:.1f}cm")
                        print("\n" + "="*52)
                        print("  CALIBRATION COMPLETE — Tracking is now Active!")
                        print("="*52 + "\n")
                else:
                    if c_upper: c_upper.pop()
                    if c_fore:  c_fore.pop()
                    cv2.putText(frame, "Not detected — spread arms wide", (14, 68),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 100, 255), 1)


            # ════ STATE: TRACKING ═════════════════════════════════════════
            elif state == "TRACKING":
                node.publish_state("TRACKING")

                if banner_timer > 0:
                    banner_timer -= 1
                    cv2.rectangle(frame, (0, h//2-55), (w, h//2+60), (0,0,0), -1)
                    cv2.putText(frame, "CALIBRATION COMPLETE",
                                (w//2-240, h//2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)
                    cv2.putText(frame, "Tracking Active  |  R = recalibrate",
                                (w//2-250, h//2+42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (180,255,180), 2)
                else:
                    cv2.rectangle(frame, (0,0), (w, 44), (0,0,0), -1)
                    cv2.putText(frame, "TRACKING  |  R=recalibrate | bend elbow & rotate palm!",
                                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,0), 2)

                if pose_ok:
                    m_rs = to_mujoco(rs)
                    m_re = to_mujoco(re)
                    m_rw = to_mujoco(rw)
                    
                    # Apply the calibration scaling factors to the direction vectors
                    u_vec = m_re - m_rs
                    f_vec = m_rw - m_re
                    
                    # Prevent zero division / null vectors
                    ul = np.linalg.norm(u_vec)
                    fl = np.linalg.norm(f_vec)
                    
                    if ul > 1e-4 and fl > 1e-4:
                        u_scaled = (u_vec / ul) * ROBOT_UPPER_ARM
                        f_scaled = (f_vec / fl) * ROBOT_FOREARM
                        
                        m_rs_fixed = np.zeros(3) # Shoulder is origin
                        m_re_fixed = m_rs_fixed + u_scaled
                        m_rw_fixed = m_re_fixed + f_scaled
                        
                        node.publish_joints(
                            m_rs_fixed, m_re_fixed, m_rw_fixed, last_hand_quat)
                        node.publish_gripper(last_gripper_val)

                # Live HUD
                draw_joint_hud(frame, h, w, *hud_angles)

            cv2.imshow("Mocap Camera Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                state = "TPOSE"
                c_upper.clear(); c_fore.clear()
                cur_upper_dir = None
                banner_timer = 0
                print("\n[Cal] Reset — T-POSE to start again.")

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
