"""
mocap_pub.py — ROS2 Publisher Node

Listens on UDP port 5005 for JSON data from camera_tracker.py and publishes:
  /human/skeletal_data    → geometry_msgs/PoseArray
      poses[0] = shoulder  (position: x,y,z)
      poses[1] = elbow     (position: x,y,z)
      poses[2] = hand      (position: x,y,z  |  orientation: hand quaternion)
  /mocap/state            → std_msgs/String  ("TRACKING" or "CALIBRATION")

Topic and message format matches Akshay's ocra_node.py subscriber exactly.

How to run:
  ros2 run mocap_publisher mocap_pub_node

Ensure camera_tracker.py is also running to send UDP data on port 5005.
"""

import json
import socket

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String


UDP_IP   = "0.0.0.0"   # Listen on all interfaces
UDP_PORT = 5005


class MocapPublisher(Node):
    def __init__(self):
        super().__init__('mocap_publisher')

        # ── Publisher: matches /human/skeletal_data that ocra_node.py subscribes to ──
        self.pub_skeletal = self.create_publisher(
            PoseArray, '/human/skeletal_data', 10
        )

        # ── Tracker state ("TRACKING" / "CALIBRATION") ──────────────────────
        self.pub_state = self.create_publisher(
            String, '/mocap/state', 10
        )

        # ── UDP socket (non-blocking) ────────────────────────────────────────
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)
        self.get_logger().info(
            f'Mocap Publisher listening on UDP {UDP_IP}:{UDP_PORT} ...'
        )
        self.get_logger().info(
            'Publishing on /human/skeletal_data → poses[0]=shoulder, '
            'poses[1]=elbow, poses[2]=hand+quaternion'
        )

        # ── Timer: poll UDP at 60 Hz ─────────────────────────────────────────
        self.create_timer(1.0 / 60.0, self.timer_callback)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _xyz_pose(self, xyz):
        """Pose with only position set (identity orientation)."""
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])
        p.orientation.w = 1.0
        return p

    def _xyz_pose_with_quat(self, xyz, quat):
        """
        Pose with position AND orientation (hand pose).
        quat = [qx, qy, qz, qw]
        This matches how ocra_node.py reads:
            msg.poses[2].position   → hand XYZ
            msg.poses[2].orientation → hand quaternion
        """
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])
        p.orientation.x = float(quat[0])
        p.orientation.y = float(quat[1])
        p.orientation.z = float(quat[2])
        p.orientation.w = float(quat[3])
        return p

    # ── main callback ────────────────────────────────────────────────────────

    def timer_callback(self):
        """Drain the UDP buffer and publish the latest packet."""
        payload = None

        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
                payload = json.loads(data.decode())
            except BlockingIOError:
                break
            except json.JSONDecodeError as e:
                self.get_logger().warn(f'JSON decode error: {e}')
                break

        if payload is None:
            return

        now = self.get_clock().now().to_msg()
        state_str = payload.get('state', 'UNKNOWN')

        # ── Publish state ────────────────────────────────────────────────────
        state_msg = String()
        state_msg.data = state_str
        self.pub_state.publish(state_msg)

        if state_str != 'TRACKING':
            return

        # ── Build PoseArray matching ocra_node.py's expected format ─────────
        try:
            shoulder = payload['h_shoulder']   # [x, y, z]
            elbow    = payload['h_elbow']       # [x, y, z]
            hand     = payload['h_wrist']       # [x, y, z]  (wrist = hand pos)
            quat     = payload.get('h_hand_quat', [0.0, 0.0, 0.0, 1.0])  # [qx,qy,qz,qw]
        except KeyError as e:
            self.get_logger().warn(f'Missing key in payload: {e}')
            return

        pose_array = PoseArray()
        pose_array.header.stamp    = now
        pose_array.header.frame_id = 'mocap_world'
        pose_array.poses = [
            self._xyz_pose(shoulder),              # poses[0] → shoulder
            self._xyz_pose(elbow),                 # poses[1] → elbow
            self._xyz_pose_with_quat(hand, quat),  # poses[2] → hand + quaternion
        ]

        self.pub_skeletal.publish(pose_array)

        self.get_logger().debug(
            f'shoulder={shoulder} elbow={elbow} hand={hand} quat={quat}'
        )

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MocapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

