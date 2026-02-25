import rclpy
import rclpy.node as Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray
import numpy as np
import time

import rx200_kinematics as rx_kine
from scipy.optimize import minimize

LOOP_RATE = 25
ALPHA = 0.67
BETA = 0.33

JOINT_LIMITS = [
    (-3.1416, 3.1416),   # Waist
    (-1.8849, 1.9722),   # Shoulder
    (-1.8849, 1.6231),   # Elbow
    (-1.7453, 2.1467),   # Wrist Angle
    (-3.1416, 3.1416)    # Wrist Rotate
]

class OCRANode(Node):
    def __init__(self):
        super().__init__('ocra_controller')

        self.cb = ReentrantCallbackGroup()

        self.target_sub = self.create_subscription(
            PoseArray,
            '/human/skeletal_data',
            self.human_callback,
            1,
            callback_group = self.cb_group
        )

        self.robot_sub = self.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.robot_state_callback,
            10,
            callback_group = self.cb_group
        )

        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/rx200/commands/joint_group',
            10
        )

        self.current_joints = np.zeros(5)
        self.last_solution = np.zeros(5)
        self.latest_target_flat = None
        self.timer = self.create_timer(
            1/LOOP_RATE,
            self.control_loop,
            callback_group = self.cb_group
        )

        self.get_logger().info("OCRA Controller Node Initialized, waiting for human data...")
    
    def robot_state_callback(self, msg):
        self.current_joints = np.array(msg.position[:5])
    
    def human_callback(self, msg):
        if len(msg.poses) < 3:
            self.get_logger().warn("Received skeletal data with less than 3 keypoints, ignoring.")
            return
        
        # Extract shoulder, elbow, hand positions and hand rotation (quaternion)
        shoulder = np.array([msg.poses[0].position.x, msg.poses[0].position.y, msg.poses[0].position.z])
        elbow = np.array([msg.poses[1].position.x, msg.poses[1].position.y, msg.poses[1].position.z])
        hand = np.array([msg.poses[2].position.x, msg.poses[2].position.y, msg.poses[2].position.z])
        
        quat = np.array([
            msg.poses[2].orientation.x, 
            msg.poses[2].orientation.y, 
            msg.poses[2].orientation.z, 
            msg.poses[2].orientation.w])

        self.latest_target_flat = np.concatenate([shoulder, elbow, hand, quat])

        def control_loop(self):
            if self.latest_target_flat is None:
                return
            
            x0 = self.last_solution if np.any(self.last_solution) else self.current_joints

            res = minimize(
                fun=rx_kine.loss_and_grad_fn,
                x0=x0,
                args=(self.latest_target_flat, [ALPHA, BETA]),
                method='SLSQP',
                jac=True, 
                bounds=JOINT_LIMITS,
                options={'maxiter': 5, 'ftol': 1e-3, 'disp': False}
            )
            
            if res.success:
                self.last_solution = res.x
                cmd_msg = Float64MultiArray()
                cmd_msg.data = res.x.tolist()
                self.cmd_pub.publish(cmd_msg)
            else:
                self.get_logger().warn(f"Optimization failed: {res.message}")

def main(args=None):
    rclpy.init(args=args)
    ocra_node = OCRANode()

    executor = MultiThreadedExecutor()
    executor.add_node(ocra_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ocra_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
