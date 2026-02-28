import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from scipy.optimize import Bounds
import numpy as np
import jax.numpy as jnp

from . import rx200_kinematics as rx_kine
from scipy.optimize import minimize

LOOP_RATE = 10
ALPHA     = 0.67
BETA      = 0.33
GAMMA    =  2*ALPHA  # Additional weight for hand position error, not in original paper but helps convergence

JOINT_NAMES = [
    'waist',
    'shoulder',
    'elbow',
    'wrist_angle',
    'wrist_rotate'
]

# keep_feasible=True prevents SLSQP stepping outside bounds during line search
BOUNDS = Bounds(
    lb=[-3.1416, -1.8849, -1.8849, -1.7453, -3.1416],
    ub=[ 3.1416,  1.9722,  1.6231,  2.1467,  3.1416],
    keep_feasible=True
)


class OCRASimNode(Node):
    def __init__(self):
        super().__init__('ocra_sim_controller')

        # ── Subscribers ───────────────────────────────────────────
        self.target_sub = self.create_subscription(
            PoseArray,
            '/human/skeletal_data',
            self.human_callback,
            1
        )

        self.robot_sub = self.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.robot_state_callback,
            10
        )

        # ── Publisher ─────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            JointTrajectory,
            '/rx200/arm_controller/joint_trajectory',
            10
        )

        self.current_joints     = np.zeros(5)
        self.last_solution      = np.zeros(5)
        self.latest_target_flat = None
        self.first_solve        = True

        # Rate limiter — prevents flooding the controller
        self.last_publish_time    = self.get_clock().now()
        self.min_publish_interval = 1.0 / LOOP_RATE  # match loop rate exactly

        # ── Timer: single control loop at LOOP_RATE ───────────────
        self.timer = self.create_timer(1.0 / LOOP_RATE, self.control_loop)

        # ── JAX JIT warmup — prevents NaN on first call ───────────
        self._warmup_jit()

        self.get_logger().info("OCRA Sim Controller Node Initialized, waiting for human data...")

    def _warmup_jit(self):
        dummy_q      = jnp.zeros(5)
        dummy_target = jnp.array([0.0, 0.0, 0.3,
                                   0.2, 0.0, 0.2,
                                   0.35, 0.0, 0.1,
                                   0.0, 0.0, 0.0, 1.0])
        dummy_w = jnp.array([ALPHA, BETA, GAMMA])
        _ = rx_kine.loss_and_grad_fn(dummy_q, dummy_target, dummy_w)
        self.get_logger().info("JAX JIT warmup complete.")

    # ── Callbacks ─────────────────────────────────────────────────

    def robot_state_callback(self, msg):
        # Name-based lookup — safe regardless of joint order in message
        name_to_pos = dict(zip(msg.name, msg.position))
        self.current_joints = np.array([
            name_to_pos.get(n, 0.0) for n in JOINT_NAMES
        ], dtype=np.float64)

    def human_callback(self, msg):
        if len(msg.poses) < 3:
            self.get_logger().warn("Skeletal data has fewer than 3 keypoints, ignoring.")
            return

        shoulder = np.array([msg.poses[0].position.x,
                             msg.poses[0].position.y,
                             msg.poses[0].position.z])
        elbow    = np.array([msg.poses[1].position.x,
                             msg.poses[1].position.y,
                             msg.poses[1].position.z])
        hand     = np.array([msg.poses[2].position.x,
                             msg.poses[2].position.y,
                             msg.poses[2].position.z])
        quat = np.array([
            msg.poses[2].orientation.x,
            msg.poses[2].orientation.y,
            msg.poses[2].orientation.z,
            msg.poses[2].orientation.w
        ])
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        self.latest_target_flat = np.concatenate([shoulder, elbow, hand, quat])

    # ── Control loop ──────────────────────────────────────────────

    def control_loop(self):
        if self.latest_target_flat is None:
            return

        x0       = self.current_joints if self.first_solve else self.last_solution
        max_iter = 50 if self.first_solve else 10

        # Wrap JAX output → plain float64 numpy for scipy
        def loss_fn(x):
            val, grad = rx_kine.loss_and_grad_fn(
                jnp.array(x),
                jnp.array(self.latest_target_flat),
                jnp.array([ALPHA, BETA, GAMMA])
            )
            val_np  = float(val)
            grad_np = np.array(grad, dtype=np.float64)
            # Sanitize any residual NaN/Inf
            if not np.isfinite(val_np):
                val_np = 1e6
            grad_np = np.where(np.isfinite(grad_np), grad_np, 0.0)
            return val_np, grad_np

        res = minimize(
            fun=loss_fn,
            x0=np.array(x0, dtype=np.float64),
            method='SLSQP',
            jac=True,
            bounds=BOUNDS,
            options={'maxiter': max_iter, 'ftol': 1e-4}
        )

        if res.success or 'iteration' in res.message.lower():
            self.last_solution = res.x
            self.first_solve   = False
            self._publish_trajectory(res.x)
        else:
            self.get_logger().warn(f"Optimization failed: {res.message}",
                                   throttle_duration_sec=1.0)

    def _publish_trajectory(self, joint_positions):
        # Rate limiter — skip if called too soon
        now = self.get_clock().now()
        dt  = (now - self.last_publish_time).nanoseconds / 1e9
        if dt < self.min_publish_interval:
            return

        self.last_publish_time = now

        msg              = JointTrajectory()
        msg.header.stamp = now.to_msg()
        msg.joint_names  = JOINT_NAMES

        point               = JointTrajectoryPoint()
        point.positions     = joint_positions.tolist()
        point.velocities    = [0.0] * 5
        point.accelerations = [0.0] * 5

        # 200ms: enough for controller to execute, short enough for responsiveness
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)

        msg.points = [point]
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OCRASimNode()

    try:
        rclpy.spin(node)          # single threaded — no race conditions
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()