import mujoco
import numpy as np

def apply_pos_control(model, data, qd):
    # Use only the actuated joints
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_qpos_addr = model.jnt_qposadr[joint_id]
        data.ctrl[i] = qd[joint_qpos_addr]

def transformation(p, R):
    p.reshape(3, 1)
    R.reshape(3, 3)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T

def get_body_full_transformation(model, data, body_id):
    p = data.xpos[body_id]
    R = data.xmat[body_id].reshape(3, 3)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    return T

def get_body_rot_jac_local_frame(model, data, body_id):
    jacp = np.zeros((3, model.nv)) # 3 x nv translational jac
    jacr = np.zeros((3, model.nv)) # 3 x nv rotational jac

    # Compute Jac in world frame
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Transform to body frame
    R_wb = data.xmat[body_id].reshape(3, 3)
    R_bw = R_wb.T

    return R_bw @ jacr

def get_body_trans_jac_local_frame(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Compute Jac in world frame
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Transform to body frame
    R_wb = data.xmat[body_id].reshape(3, 3)
    R_bw = R_wb.T

    return R_bw @ jacp

def get_body_jac_local_frame(model, data, body_id):
    jacp = np.zeros((3, model.nv)) # 3 x nv translational jac
    jacr = np.zeros((3, model.nv)) # 3 x nv rotational jac

    # Compute Jac in world frame
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

    # Transform to body frame
    R_wb = data.xmat[body_id].reshape(3, 3)
    R_bw = R_wb.T
    J_b = np.vstack((R_bw @ jacr, R_bw @ jacp))

    return J_b

def get_current_joint_range(model, qk):
    # Get the min and max joint limits
    lower_limits = model.jnt_range[1:, 0]
    upper_limits = model.jnt_range[1:, 1]

    # Available joint range of movement

    limit_size = lower_limits.shape[0]
    nv = model.nv

    d_min = np.full(nv, -np.inf)
    d_max = np.full(nv, np.inf)

    d_min[nv - limit_size:] = lower_limits - qk[-limit_size:]
    d_max[nv - limit_size:] = upper_limits - qk[-limit_size:]

    return d_min, d_max
