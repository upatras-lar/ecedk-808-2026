import mujoco
import mujoco.viewer
import mujoco_utils
from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
from scipy.spatial.transform import Rotation as R
import proxsuite
import math_utils

class Talos:
    def __init__(self):
        # Load the robot and set him to the walk pose
        self.model = load_robot_description("talos_mj_description", variant="scene_position")
        self.data = mujoco.MjData(self.model)
        walk_pose_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "walk_pose")
        mujoco.mj_resetDataKeyframe(self.model, self.data, walk_pose_id)
        mujoco.mj_forward(self.model, self.data)

        self.model.vis.quality.shadowsize = 0

        self.left_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_6_link")
        self.right_foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_right_6_link")
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_1_link")

    def inverse_kinematics(self, data, feet_targets, feet_weights, com_target, com_weight, step = 0.001, max_iters = 5000):
        qp_dim = self.model.nv
        qp_eq = 0
        qp_ineq = self.model.nv
        qp = proxsuite.proxqp.dense.QP(qp_dim, qp_eq, qp_ineq)

        it = 0

        qk = data.qpos.copy()

        solution = qk.copy()

        feet_ids = [self.left_foot_id, self.right_foot_id]

        while True:
            # Calculate feet errors and Jacobians
            errors = []
            Jacobians = []
            for foot_id, foot_target in zip(feet_ids, feet_targets):
                T_wb = mujoco_utils.get_body_full_transformation(self.model, data, foot_id)
                T_wb_inv = math_utils.invert_transformation(T_wb)
                foot_error = math_utils.log_transformation(T_wb_inv @ foot_target)
                errors.append(foot_error)

                Jacobians.append(mujoco_utils.get_body_jac_local_frame(self.model, data, foot_id))

            # Calculate CoM error and Jacobian
            com = data.subtree_com[0]
            errors.append((com_target - com).reshape(3, 1))

            J_b_com = np.zeros((3, self.model.nv))
            mujoco.mj_jacSubtreeCom(self.model, data, J_b_com, 0)
            Jacobians.append(J_b_com)

            # Stack errors and jacobians
            ##### ENTER CODE HERE #####

            ###########################

            # Check for convergence
            if np.all(error < 1e-5) or it == max_iters:
                break

            # Setup the QP problem
            ##### ENTER CODE HERE #####

            ###########################
            W = np.dot(weights, J)
            error = weights @ error

            ##### ENTER CODE HERE #####

            ###########################
            ##### ENTER CODE HERE #####

            ###########################

            # Solve

            if it == 0:
                qp.init(Q, q, None, None, C, d_min, d_max)
            else:
                qp.update(Q, q, None, None, C, d_min, d_max)
            qp.solve()

            v = np.copy(qp.results.x)

            mujoco.mj_integratePos(self.model, qk, v, step)
            solution = qk.copy()
            data.qpos[:] = qk
            mujoco.mj_forward(self.model, data)

            it += 1
        
        return solution
    
    def march(self, n_steps, travel_distance, time_step, theta):
        data_sim = mujoco.MjData(self.model)
        walk_pose_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "walk_pose")
        mujoco.mj_resetDataKeyframe(self.model, data_sim, walk_pose_id)
        mujoco.mj_forward(self.model, data_sim)

        walk_trajectory = []

        Ryaw = math_utils.Ryaw(theta / (n_steps * 5.))

        # Find the x, y targets for the given step length
        dx_target = travel_distance * np.cos(theta)
        dy_target = travel_distance * np.sin(theta)

        dx = dx_target / n_steps
        dy = dy_target / n_steps

        for i in range(n_steps):
            left_foot_pos = data_sim.xpos[self.left_foot_id].copy()

            # Move right leg
            walk_trajectory.extend(self.move_leg(data_sim, dx, dy, time_step , Ryaw, False))
            # Move left leg
            walk_trajectory.extend(self.move_leg(data_sim, dx, dy, time_step, Ryaw, True))
        
        return walk_trajectory
    
    def turn(self, data, theta, steps, time_step):
        traj = []

        # Throw away uneccessary turns
        theta = math_utils.normalize_angle(theta)

        Ryaw = math_utils.Ryaw(theta / (steps * 3.)) # because for every spline we have 3 points

        for i in range(steps):
            left_foot_pos = data.xpos[self.left_foot_id].copy()
            right_foot_pos = data.xpos[self.right_foot_id].copy()
            com_target = data.subtree_com[0].copy()

            left_arc_points = math_utils.generate_foot_arc(left_foot_pos, left_foot_pos, 0.1, 3)
            right_arc_points = math_utils.generate_foot_arc(right_foot_pos, right_foot_pos, 0.1, 3)

            if (theta > 0):
                print(f"Now we turn left (counterclockwise)")

                # Move left foot first

                c = 0

                for point in left_arc_points:
                    T_wd_left_foot = mujoco_utils.transformation(point, Ryaw.copy() @ data.xmat[self.left_foot_id].copy().reshape(3, 3))
                    T_wd_right_foot = mujoco_utils.transformation(right_foot_pos, data.xmat[self.right_foot_id].copy().reshape(3, 3))

                    feet_targets = [T_wd_left_foot, T_wd_right_foot]

                    # Weights

                    w_feet = [np.full(6, 20.), np.full(6, 20.)]
                    w_com = np.full(3, 5.)

                    q_start = data.qpos.copy()

                    sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

                    # Interpolate trajectory
                    c += 1
                    q_end = sol.copy()
                    traj.extend(math_utils.interpolate_traj(q_start, q_end, int(time_step/ 0.003)))
                
                # Move right foot
                
                c = 0

                for point in right_arc_points:
                    T_wd_left_foot = mujoco_utils.transformation(left_foot_pos, data.xmat[self.left_foot_id].copy().reshape(3, 3))
                    T_wd_right_foot = mujoco_utils.transformation(point, Ryaw.copy() @ data.xmat[self.right_foot_id].copy().reshape(3, 3))

                    feet_targets = [T_wd_left_foot, T_wd_right_foot]

                    # Weights

                    w_feet = [np.full(6, 20.), np.full(6, 20.)]
                    w_com = np.full(3, 5.)

                    q_start = data.qpos.copy()

                    sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

                    # Interpolate trajectory
                    c += 1
                    q_end = sol.copy()
                    traj.extend(math_utils.interpolate_traj(q_start, q_end, int(time_step/ 0.003)))

            else:
                print(f"Now we turn right (clockwise)")
                # Move right foot
                
                c = 0

                for point in right_arc_points:
                    T_wd_left_foot = mujoco_utils.transformation(left_foot_pos, data.xmat[self.left_foot_id].copy().reshape(3, 3))
                    T_wd_right_foot = mujoco_utils.transformation(point, Ryaw.copy() @ data.xmat[self.right_foot_id].copy().reshape(3, 3))

                    feet_targets = [T_wd_left_foot, T_wd_right_foot]

                    # Weights

                    w_feet = [np.full(6, 20.), np.full(6, 20.)]
                    w_com = np.full(3, 5.)

                    q_start = data.qpos.copy()

                    sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

                    # Interpolate trajectory
                    c += 1
                    q_end = sol.copy()
                    traj.extend(math_utils.interpolate_traj(q_start, q_end, int(time_step/ 0.003)))

                # Move left foot first

                c = 0

                for point in left_arc_points:
                    T_wd_left_foot = mujoco_utils.transformation(point, Ryaw.copy() @ data.xmat[self.left_foot_id].copy().reshape(3, 3))
                    T_wd_right_foot = mujoco_utils.transformation(right_foot_pos, data.xmat[self.right_foot_id].copy().reshape(3, 3))

                    feet_targets = [T_wd_left_foot, T_wd_right_foot]

                    # Weights

                    w_feet = [np.full(6, 20.), np.full(6, 20.)]
                    w_com = np.full(3, 5.)

                    q_start = data.qpos.copy()

                    sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

                    # Interpolate trajectory
                    c += 1
                    q_end = sol.copy()
                    traj.extend(math_utils.interpolate_traj(q_start, q_end, int(time_step/ 0.003)))
        return traj

    def move_leg(self, data, dx, dy, swing_phase, Ryaw ,move_left = True):
        left_foot_pos = data.xpos[self.left_foot_id].copy()
        right_foot_pos = data.xpos[self.right_foot_id].copy()
        com_target = data.subtree_com[0].copy()

        if move_left:
            # com_target[0] = com_target[0] + 0.5 * (right_foot_pos[0] - com_target[0])
            # com_target[1] = com_target[1] + 0.5 * (right_foot_pos[1] - com_target[1])
            com_target[0] = right_foot_pos[0]
            com_target[1] = right_foot_pos[1]
            left_foot_target = left_foot_pos.copy()
            left_foot_target[0] += dx
            left_foot_target[1] += dy
            arc_points = math_utils.generate_foot_arc(left_foot_pos, left_foot_target, 0.1, 5)
        else:
            # com_target[0] = com_target[0] + 0.5 * (left_foot_pos[0] - com_target[0])
            # com_target[1] = com_target[1] + 0.5 * (left_foot_pos[1] - com_target[1])
            com_target[0] = left_foot_pos[0]
            com_target[1] = left_foot_pos[1]
            right_foot_target = right_foot_pos.copy()
            right_foot_target[0] += dx
            right_foot_target[1] += dy
            arc_points = math_utils.generate_foot_arc(right_foot_pos, right_foot_target, 0.1, 5)

        c = 0

        traj = []

        for point in arc_points:

            if move_left:
                T_wd_left_foot = mujoco_utils.transformation(point, Ryaw.copy() @ data.xmat[self.left_foot_id].copy().reshape(3, 3))
                T_wd_right_foot = mujoco_utils.transformation(right_foot_pos, data.xmat[self.right_foot_id].copy().reshape(3, 3))
            else:
                T_wd_left_foot = mujoco_utils.transformation(left_foot_pos, data.xmat[self.left_foot_id].copy().reshape(3, 3))
                T_wd_right_foot = mujoco_utils.transformation(point, Ryaw.copy() @ data.xmat[self.right_foot_id].copy().reshape(3, 3))

            feet_targets = [T_wd_left_foot, T_wd_right_foot]

            # Weights

            w_feet = [np.full(6, 20.), np.full(6, 20.)]
            w_com = np.full(3, 5.)

            q_start = data.qpos.copy()

            sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

            # Interpolate trajectory
            c += 1
            q_end = sol.copy()
            traj.extend(math_utils.interpolate_traj(q_start, q_end, int(swing_phase/ 0.005)))
        
        return traj
    
    def shift_com(self, data, x, y, shift_phase):
        left_foot_pos = data.xpos[self.left_foot_id].copy()
        right_foot_pos = data.xpos[self.right_foot_id].copy()
        com_target = data.subtree_com[0].copy()
        # com_target[0] = x
        # com_target[1] = y
        com_target[0] = com_target[0] + 0.5 * (x - com_target[0])
        com_target[1] = com_target[1] + 0.5 * (y - com_target[1])


        T_wd_left_foot = mujoco_utils.transformation(left_foot_pos, data.xmat[self.left_foot_id].copy().reshape(3, 3))
        T_wd_right_foot = mujoco_utils.transformation(right_foot_pos, data.xmat[self.right_foot_id].copy().reshape(3, 3))

        feet_targets = [T_wd_left_foot, T_wd_right_foot]

        # Weights

        w_feet = [np.full(6, 100.), np.full(6, 100.)]
        w_com = np.full(3, 200.)

        q_start = data.qpos.copy()

        sol = self.inverse_kinematics(data, feet_targets, w_feet, com_target, w_com)

        # Interpolate traj
        traj = math_utils.interpolate_traj(q_start, sol, int(shift_phase / 0.001))

        return traj
    
    def walk(self, n_steps, step_length, left_swing_time, right_swing_time, shift_time, theta):

        data_sim = mujoco.MjData(self.model)
        walk_pose_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "walk_pose")
        mujoco.mj_resetDataKeyframe(self.model, data_sim, walk_pose_id)
        mujoco.mj_forward(self.model, data_sim)

        walk_trajectory = []

        Ryaw = math_utils.Ryaw(theta / n_steps)

        # Find the x, y targets for the given step length
        dx_target = step_length * np.cos(theta)
        dy_target = step_length * np.sin(theta)

        dx = dx_target / n_steps
        dy = dy_target / n_steps
        
        for i in range(n_steps):
            left_foot_pos = data_sim.xpos[self.left_foot_id].copy()

            # Move right leg
            walk_trajectory.extend(self.shift_com(data_sim, left_foot_pos[0], left_foot_pos[1], shift_time))
            walk_trajectory.extend(self.move_leg(data_sim, dx, dy, right_swing_time , Ryaw, False))

            right_foot_pos = data_sim.xpos[self.right_foot_id].copy()

            # Move left leg
            walk_trajectory.extend(self.shift_com(data_sim, right_foot_pos[0], right_foot_pos[1], shift_time))
            walk_trajectory.extend(self.move_leg(data_sim, dx, dy, left_swing_time, Ryaw, True))
        
        return walk_trajectory
    
    def visualize_traj(self, traj):
        with mujoco.viewer.launch_passive(self.model, self.data) as vis:
            # Set camera to look at the front side of the robot
            vis.cam.lookat[:] = [0, 0, 1.0]
            vis.cam.distance = 4.0
            vis.cam.azimuth = 180
            vis.cam.elevation = -20

            while vis.is_running():
                for i in range(0, len(traj)):
                    self.data.qpos[:] = traj[i]
                    mujoco.mj_forward(self.model, self.data)
                    vis.sync()
                mujoco.mj_forward(self.model, self.data)
                vis.sync()
    
    def position_control(self, traj):
        with mujoco.viewer.launch_passive(self.model, self.data) as vis:
            # Set camera to look at the front side of the robot
            vis.cam.lookat[:] = [0, 0, 1.0]
            vis.cam.distance = 4.0
            vis.cam.azimuth = 180
            vis.cam.elevation = -20

            while vis.is_running():
                for q in traj:
                    mujoco_utils.apply_pos_control(self.model, self.data, q)
                    mujoco.mj_step(self.model, self.data)
                    vis.sync()
                mujoco.mj_step(self.model, self.data)
                vis.sync()
    
    def simulate(self, traj):
        reward = 0.
        R_before = self.data.xmat[self.torso_id].reshape(3, 3).copy()
        init_pos = self.data.xpos[self.torso_id][:2].copy()
        # Find the geoms for each foot, to check for contacts later
        left_foot_geoms  = np.where(self.model.geom_bodyid == self.left_foot_id)[0].tolist()
        right_foot_geoms = np.where(self.model.geom_bodyid == self.right_foot_id)[0].tolist()
        for q in traj:
            mujoco_utils.apply_pos_control(self.model, self.data, q)
            mujoco.mj_step(self.model, self.data)
            # Check for fall
            torso_pos = self.data.xpos[self.torso_id].copy()
            if (torso_pos[2] < 0.6):
                break
            # For every step that you don't fall +1
            reward += 1.
            # If you pitch your torso, you get a penalty
            R_after = self.data.xmat[self.torso_id].reshape(3, 3).copy()
            R_rel = R_before.T @ R_after
            roll, pitch, yaw = R.from_matrix(R_rel).as_euler('xyz')
            reward -= abs(pitch) * 1.
            R_before = R_after.copy()
            # We don't want the feet touching each other
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if (c.geom[0] in left_foot_geoms and c.geom[1] in right_foot_geoms) or (c.geom[1] in left_foot_geoms and c.geom[0] in right_foot_geoms):
                    reward -= 1.

        # The futher you go, the more rewarded you get
        final_pos = self.data.xpos[self.torso_id][:2].copy()
        reward += np.linalg.norm(final_pos - init_pos) * 10.

        behavior = final_pos.copy()

        return reward, behavior
