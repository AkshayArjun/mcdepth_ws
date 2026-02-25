import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

M_HOME = jnp.array([
    [1.0, 0.0, 0.0, 0.408575],  
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.30457],  
    [0.0, 0.0, 0.0, 1.0]
])

S_LIST = jnp.array([
    [0.0, 0.0, 1.0,  0.0,       0.0,       0.0],      # Joint 1
    [0.0, 1.0, 0.0, -0.10457,   0.0,       0.0],      # Joint 2
    [0.0, 1.0, 0.0, -0.30457,   0.0,       0.05],     # Joint 3
    [0.0, 1.0, 0.0, -0.30457,   0.0,       0.25],     # Joint 4
    [1.0, 0.0, 0.0,  0.0,       0.30457,   0.0]       # Joint 5
])

M_ELBOW = jnp.array([
    [1.0, 0.0, 0.0, 0.05],     
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.30457],
    [0.0, 0.0, 0.0, 1.0]
])

# ======== Helper functions for forward kinematics ========
# Lie algebra se(3) and its exponential map to SE(3)

@jit
def skew(v):
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

@jit
def matrix_exp_se3(S, theta):
    omega = S[:3]
    v = S[3:]

    #add def of rodrigue's formula here for future reference
    omega_mat = skew(omega)
    R = jnp.eye(3) + jnp.sin(theta)*omega_mat + (1 - jnp.cos(theta))*(omega_mat @ omega_mat)
    v_mat = (jnp.eye(3)*theta + 
             (1 - jnp.cos(theta))*omega_mat +
             (theta - jnp.sin(theta))*(omega_mat @ omega_mat))
    p = v_mat @ v

    T = jnp.eye(4)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(p)
    return T

# ===== ocra function terms ========
# the fucntion is defined as ϵs​=l∑i=1m​si​+∑j=1n​tj​​ where si is the distance from the elbow to 
# the line segment between the shoulder and the hand, and tj is the distance from the hand 
# to the target hand position and orientation.
# =======================================

# =======================================
# im doing this by definign a dista_point_to_seg function that will give me the distance of 
# a point to a line segement
# then ill find the minimum of the disatnces to get the closesnt segement and use that in the
# loss function 
# =======================================

@jit
def dist_point_to_segment(p, a, b):
    segment_vec = b - a
    point_vec = p - a
    seg_len_sq = jnp.dot(segment_vec, segment_vec)

    t = jnp.where(seg_len_sq > 1e-6, jnp.dot(point_vec, segment_vec) / seg_len_sq, 0.0)
    t = jnp.clip(t, 0.0, 1.0)

    closest_point = a + t * segment_vec
    return jnp.linalg.norm(p - closest_point)

@jit
def get_min_distance_to_chain(point, chain_joints):
    d1 = dist_point_to_segment(point, chain_joints[0], chain_joints[1])
    d2 = dist_point_to_segment(point, chain_joints[1], chain_joints[2])

    return jnp.minimum(d1, d2)

# ======== Forward kinematics ========

@jit
def forward_kinematics(joint_angles):
    # compute the forward kinematics and return the elbox-xyz, hand_xyz, and hand-rotation

    T = jnp.eye(4)

    T = T @ matrix_exp_se3(S_LIST[0], joint_angles[0])
    T = T @ matrix_exp_se3(S_LIST[1], joint_angles[1])
    
    T_elbow = T @ M_ELBOW
    elbow_pos = T_elbow[:3, 3]

    T = T @ matrix_exp_se3(S_LIST[2], joint_angles[2])
    T = T @ matrix_exp_se3(S_LIST[3], joint_angles[3])
    T = T @ matrix_exp_se3(S_LIST[4], joint_angles[4])

    T_hand = T @ M_HOME
    hand_pos = T_hand[:3, 3]
    hand_rot = T_hand[:3, :3]

    return elbow_pos, hand_pos, hand_rot

@jit
def ocra_loss(joint_angles, target_flat, weights):
    #joint angles : [5]
    #target_flat : flattened array of [shoulde[3] relbow_pos(3), hand_pos(3), hand_rot(4)]
    #weights : alpha, beta for the loss terms
    
    t_shoulder = target_flat[:3]
    t_elbow = target_flat[3:6]
    t_hand = target_flat[6:9]
    t_quat = target_flat[9:13]

    alpha = weights[0]
    beta = weights[1]
    
    ROBOT_BASE = jnp.array([0.0, 0.0, 0.0])  # Assuming the robot base is at the origin
    r_elbow, r_hand, r_rot = forward_kinematics(joint_angles)
    
    robot_chain = jnp.stack ([ROBOT_BASE, r_elbow, r_hand])
    human_chain = jnp.stack ([t_shoulder, t_elbow, t_hand])

    d_h_elb = get_min_distance_to_chain(t_elbow, robot_chain)
    d_h_hnd = get_min_distance_to_chain(t_hand, robot_chain)

    d_r_hnd = get_min_distance_to_chain(r_hand, human_chain)
    d_r_elb = get_min_distance_to_chain(r_elbow, human_chain)

    #this is the skeletal error
    skel_err = (d_h_elb**2 + d_h_hnd**2) + (d_r_elb**2 + d_r_hnd**2)
    
    #now orrientation error acc to ocra
    tr = jnp.trace(r_rot)
    s = jnp.sqrt(jnp.maximum(1.0 + tr, 1e-6)) * 2.0 
    w = 0.25 * s
    x = (r_rot[2, 1] - r_rot[1, 2]) / s
    y = (r_rot[0, 2] - r_rot[2, 0]) / s
    z = (r_rot[1, 0] - r_rot[0, 1]) / s
    r_quat = jnp.array([x, y, z, w])
    
    # Error = 1 - <q1, q2>^2
    dot_prod = jnp.dot(r_quat, t_quat)
    orient_err = 1.0 - (dot_prod ** 2)
    
    return alpha * skel_err + beta * orient_err

loss_and_grad_fn = value_and_grad(ocra_loss)



