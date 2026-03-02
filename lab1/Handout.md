# Robotic Systems I – Lab 1  
## Mobile Robot Motion Planning with RRT
### Introduction

In *Introduction to Robotics*, we learned how to build robotic systems using ROS2, focusing on communication, sensing, and software architecture. In this course, we shift our focus toward implementing the core algorithms that make robots intelligent.

In this first lab, we will implement a **sampling-based motion planner (RRT)** and use it to solve a **randomly generated maze** with a differential-drive mobile robot.

More specifically, we will:

- Work in a MuJoCo simulation environment containing a randomly generated maze
- Implement the missing components of a Rapidly-exploring Random Tree (RRT) planner
- Grow the RRT to explore the configuration space
- Find a collision-free path from a start pose to a goal pose
- Implement a controller to drive the robot along the planned path

Every time you run the simulation, a new maze is generated.  
Your implementation must therefore work **robustly**, not just for a single environment.

This lab connects motion planning and control:

1. First, you compute *where* the robot should go (planning).
2. Then, you control *how* the robot moves to follow that plan (control).

If everything goes well, by the end of this lab, you will have built a complete pipeline:

Maze → RRT Exploration → Path Extraction → Robot Control → Goal Reached

### Robot Model

We use a small differential-drive robot called **TurtleBot3 Waffle Pi**.

The configuration of the robot is:

q = [x, y, θ]

Where:
- x, y → planar position
- θ → heading angle

Control inputs:
- v → linear velocity
- ω → angular velocity

### Provided Files

As you can see, you are provided with two python files in the `src` directory: `create_maze.py` and `waffle_rrt.py`. The first one randomly generates a new maze each time you run the simulation using Prim's algorithm. **You will not need to modify this file during the lab.**

You will work with the `waffle_rrt.py` file. In this file, we have the RRT structure, the controller and we load the main simulation environment. You must complete the missing parts of the code in this file.

## Part I – RRT Motion Planning

### RRT Algorithm

RRT builds a tree in configuration space by:

1. Sampling a random configuration
2. Finding the nearest existing node
3. Extending toward the sample
4. Adding the new node if collision-free
5. Repeating until the goal is reached

In the `waffle_rrt.py` file we have there is a function called `rrt_plan_se2`. This function implements the main RRT logic. Study it carefully and then look at the TODO's after the code block. They will help you fill the blanks.

```python
def rrt_plan_se2(
    model: mujoco.MjModel,
    x_start: np.ndarray,
    x_goal: np.ndarray,
    total_bounds_xy: np.ndarray,
    maze_bounds_xy: np.ndarray,
    hall_bounds_xy: np.ndarray,
    step_size: float = 0.1,
    goal_radius_xy: float = 0.1,
    goal_radius_theta: float = 0.1,
    max_iters: int = 20000,
    w_theta: float = 0.2,
    maze_sample_rate = 0.8,
    goal_sample_rate = 0.05,
    rng: np.random.Generator | None = None
):
    total_bounds_xy = np.array(total_bounds_xy)
    maze_bounds_xy = np.array(maze_bounds_xy)
    hall_bounds_xy = np.array(hall_bounds_xy)
    if rng is None:
        rng = np.random.default_rng(seed=None)

    nodes = [RRTNode(x_start.copy(), parent=-1)]

    # Separate data for collision checking
    data_check = mujoco.MjData(model)

    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id < 0:
        floor_geom_id = None

    robot_body_in_subtree = build_base_subtree_body_mask(model, base_body_name="base")

    qpos_template = data_check.qpos.copy()
    mujoco.mj_forward(model, data_check)

    def sample_state(maze_sample_rate, goal_sample_rate):
        if rng.uniform() < goal_sample_rate:
            return x_goal.copy()
        # bounds = total_bounds_xy
        if rng.uniform() < maze_sample_rate:
            bounds = maze_bounds_xy
        else:
            bounds = hall_bounds_xy
        ##### ENTER CODE HERE #####
        
        ###########################
        return np.array([x, y, th], dtype = float)

    def nearest_index(x_sample):
        xs = np.stack([n.x for n in nodes], axis=0)  # (N,3)
        ##### ENTER CODE HERE #####
        
        ###########################
        dth = np.array([wrap_to_pi(t - x_sample[2]) for t in xs[:, 2]])
        d2 = dx*dx + dy*dy + w_theta * (dth*dth)
        return int(np.argmin(d2))

    def steer(x_near, x_sample):
        # Move a fixed step in the XY direction toward sample
        dx = x_sample[0] - x_near[0]
        dy = x_sample[1] - x_near[1]
        dist = math.hypot(dx, dy)

        if dist < 1e-9:
            # no translation; just rotate a bit toward sample theta
            th_new = x_near[2] + clamp(wrap_to_pi(x_sample[2] - x_near[2]), -0.5, 0.5)
            return np.array([x_near[0], x_near[1], wrap_to_pi(th_new)], dtype = float)

        ux = dx / dist
        uy = dy / dist
        ##### ENTER CODE HERE #####
        
        ###########################

        # clamp within bounds
        x_new = clamp(x_new, total_bounds_xy[0, 0], total_bounds_xy[0, 1])
        y_new = clamp(y_new, total_bounds_xy[1, 0], total_bounds_xy[1, 1])

        th_new = math.atan2(uy, ux)
        return np.array([x_new, y_new, th_new], dtype = float)

    goal_node_idx = None

    for iter in range(max_iters):
        ##### ENTER CODE HERE #####
        
        ###########################

        success = connect(
            model=model,
            data=data_check,
            qpos_template=qpos_template,
            x_from=x_near,
            x_to=x_new,
            robot_body_in_subtree=robot_body_in_subtree,
            floor_geom_id=floor_geom_id,
            step_m=0.1
        )

        if (iter % 100 == 0):
            print(f"Iteration: {iter} ,  RRT tree length: {len(nodes)}")

        if success:
            nodes.append(RRTNode(x_new, parent=i_near))
            i_new = len(nodes) - 1

            in_goal_xy = np.linalg.norm(x_new[0:2] - x_goal[0:2]) <= goal_radius_xy
            in_goal_th = abs(wrap_to_pi(x_new[2] - x_goal[2])) <= goal_radius_theta
            if in_goal_xy and in_goal_th:
                print(f"Break at iteration: {iter} with distance {np.linalg.norm(x_new[0:2] - x_goal[0:2])} and angle distance {abs(wrap_to_pi(x_new[2] - x_goal[2]))}")
                goal_node_idx = i_new
                break

    if goal_node_idx is None:
        return None, np.stack([n.x for n in nodes], axis = 0), np.array([n.parent for n in nodes], dtype = int)

    # Backtrack path
    rrt_path = []
    idx = goal_node_idx
    while idx != -1:
        rrt_path.append(nodes[idx].x.copy())
        idx = nodes[idx].parent
    rrt_path.reverse()
    rrt_path = np.array(rrt_path)
    rrt_nodes = np.stack([n.x for n in nodes], axis = 0)
    rrt_parents = np.array([n.parent for n in nodes], dtype = int)

    return rrt_path, rrt_nodes, rrt_parents
```
> - **Student TODO I:** As you can see this functions contains other helper functions (yes we can do that in python!). One of them is called `sample_state`. Our RRT uses this method at every step to sample a random state. Using the `rng` generator already created above, sample `x`, `y` and `theta` uniformly at random. `x` and `y` should be within the bounds contained in `bounds` and `theta` should be between `-π, π`.
> - **Student TODO II:** The `nearest_index` function takes as an argument the sampled state and finds which node of the RRT tree is closest to the new sampled state. The array `x_s` contains all the nodes currently in our RRT tree. It is of size (N, 3) where N is the total number of nodes. In order to find which of these nodes is closest to the new sampled node `x_sample`, we need to determine the distance both in x and in y. Create two arrays dx and dy both of size (N, 1). `dx` should be the distance between the nodes of the RRT graph and the new sampled point in the x direction and `dy` in the y direction.
> - **Student TODO III:** The `steer` function is used to move one step towards the direction of the new sampled state from the nearest node. `x_near` is a 3 dimensional vector representing the nearest node of the tree (aka whrere we are know). We want to take a step in the direction of the new sampled state. `ux` and `uy` are the velocities in the x and y directions respectively. `step_size` is the size of the step we want to take in that direction. Calculate the point we will end up if we take one step from where we are now with these velocities. The new coordinates are called `x_new` and `y_new`.
> - **Student TODO IV:** Now in the main loop of the RRT algorithm, we repeat the same steps until we find the goal or we exceed our max iterations First call `sample_state` method to sample a random state called `x_sample`. Then give that sampled state as an argument to the `nearest_index` method to get back the index of the closest node (`i_near`) of the RRT graph to the new state. Get the position of that nearest node using the `i_near` in the `nodes` object. Finally, take a step in the direction of the sampled node using `steer` method. The new point should be called `x_new`.

### Connect

So far we have implemented the main RRT logic, but we haven't taken into account one very important thing: can we actually connect the new state with the closest RRT node? How do we know that is a move our robot can perform or that there are no obstacles in between? To make sure that the two points can actually be connected, we use the `connect` method.

```python
def connect(model,
            data,
            qpos_template,
            x_from,
            x_to,
            robot_body_in_subtree,
            floor_geom_id,
            step_m = 0.1):
    """
    Discretize the tree edge in SE(2) and use Mujoco contacts to test for collision
    """

    p0 = x_from[:2]
    p1 = x_to[:2]

    seg = p1 - p0
    length = float(np.linalg.norm(seg))

    if length < 1e-9:
        n = 2
    else:
        n = max(2, int(math.ceil(length / step_m)) + 1)

    th0 = float(x_from[2])
    th1 = float(x_to[2])
    dth = wrap_to_pi(th1 - th0)

    for i in range(n):
        t = i / (n - 1)

        x = float(p0[0] + t * seg[0])
        y = float(p0[1] + t * seg[1])
        th = wrap_to_pi(th0 + t * dth)

        # Reset qpos to a template so wheels etc remain sane
        data.qpos[:] = qpos_template

        data.qpos[0] = x
        data.qpos[1] = y
        data.qpos[2] = qpos_template[2]
        data.qpos[3:7] = yaw_to_quat(th)

        ##### ENTER CODE HERE #####
        
        ###########################

        if in_contact(model, data, robot_body_in_subtree, floor_geom_id):
            return False
        
    return True
```

> - **Student TODO I:** This method takes the two points that we want to connect breaks the path between them into small segments and for each segment does forward kinematics for the robot (to make sure that it is a viable move) and also uses the `in_contact` method to check if the robot is in contact with an obstacle. With mujoco we can use the `mj_forward` method with `model` and `data` as arguments to do forward kinematics.

### In contact

As we saw previously, the `connect` method uses the `in_contact` method to check for collisions. In MuJoCo we can check for collisions between any two objects in the scene at any point during the simulation.

```python
def in_contact(model: mujoco.MjModel,
               data: mujoco.MjData,
               robot_body_in_subtree: np.ndarray,
               floor_geom_id: int | None = None) -> bool:
    """
    Returns True if robot collides with ANY obstacle geom.
    """

    for k in range(data.ncon):
        ##### ENTER CODE HERE #####
        
        ###########################
        g1 = c.geom1
        g2 = c.geom2

        # Ignore floor contacts
        if floor_geom_id is not None and (g1 == floor_geom_id or g2 == floor_geom_id):
            continue

        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]

        # Check robot vs non-robot collision
        ##### ENTER CODE HERE #####
        
        ###########################

    return False
```

> - **Student TODO I:** In MuJoCo at any point the total number of contacts is `data.ncon` and the information for each contact is stored in the `data.contact` array. As we iterate over all contacts, we want the variable `c` to be the current contact with index `k`
> - **Student TODO II:** The two bodies involved in the contact are `b1` and `b2`. The array `robot_body_in_subtree` contains `True` if the body that is given as an index is belongs to the robot. Gιven this information, we would like our `in_contact` function to only return True if the contact was between the robot and an obstacle.

### Following the waypoints

After RRT has found the path to the goal, we use the `follow_waypoints` method to try and follow the found path as best we can.

```python
def follow_waypoints_se2(model, data, viewer, waypoints: np.ndarray):
    act_left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
    act_right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right")
    if act_left < 0 or act_right < 0:
        raise RuntimeError("Could not find actuators wheel_left / wheel_right.")

    L = 0.288
    r = 0.033
    W_MIN, W_MAX = -7.88, 7.88

    # gains
    Kp_dist = 1.0
    Kp_yaw_pos = 3.0   
    Kp_yaw_final = 4.0

    v_max = 5.
    omega_max = 2.5

    wp_tol = 0.15
    final_xy_tol = 0.10
    final_th_tol = 0.12

    ##### ENTER CODE HERE #####
        
    ###########################
    mujoco.mj_forward(model, data)

    wp_i = 0

    while viewer.is_running():
        ##### ENTER CODE HERE #####
        
        ###########################

        target = waypoints[wp_i]
        tx, ty, tth = float(target[0]), float(target[1]), float(target[2])

        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)

        # advance waypoint
        if wp_i < len(waypoints) - 1 and dist < wp_tol:
            wp_i += 1
            continue

        # final stopping condition
        if wp_i == len(waypoints) - 1:
            eyaw_final = wrap_to_pi(tth - yaw)
            if dist < final_xy_tol and abs(eyaw_final) < final_th_tol:
                data.ctrl[act_left] = 0.0
                data.ctrl[act_right] = 0.0
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(dt)
                continue

        yaw_des = math.atan2(dy, dx) if dist > 1e-6 else tth
        eyaw = wrap_to_pi(yaw_des - yaw)

        v = clamp(Kp_dist * dist, 0.0, v_max)
        if wp_i == len(waypoints) - 1 and dist < 0.4:
            v *= 0.5

        if wp_i == len(waypoints) - 1 and dist < 0.15:
            eyaw_final = wrap_to_pi(tth - yaw)
            omega = clamp(Kp_yaw_final * eyaw_final, -omega_max, omega_max)
            v = min(v, 0.08)  # creep while aligning
        else:
            omega = clamp(Kp_yaw_pos * eyaw, -omega_max, omega_max)
        
        ##### ENTER CODE HERE #####
        
        ###########################

        ##### ENTER CODE HERE #####
        
        ###########################

        ##### ENTER CODE HERE #####
        
        ###########################
        viewer.sync()
        time.sleep(dt)
```
> - **Student TODO I:** In this function we will run our simulation. First let's get our simulation timestep. We can find it in `model.opt.timestep`. Let's put it in a variable named `dt`.
> - **Student TODO II:** Now in the main simulation loop, let's get the current position and orientation of the robot base. We can access the current state of our floating base in `data.qpos`. Remember the first three agruments are position and the next four are orientation (quaternion). Let's name our x position `x` and our y position `y`. We can transform our four quaternion parts (`qw`, `qx`, `qy` and `qz`) to yaw angle using the `quat_to_yaw`. We'll store that in the variable `yaw`
> - **Student TODO III:** We can calculate the speed that we need to apply at each wheel using the formula: $\omega_{left} = \frac{v}{r} - \frac{L \omega}{2r}$ and $\omega_{right} = \frac{v}{r} + \frac{L \omega}{2r}$. Then we will use the `clamp` method to make sure these velocities stay within the limit of `(W_MIN, W_MAX)`.
> - **Student TODO IV:** Now that we calculated the desired velocities for both wheels, let's apply them to our actuators. We must put the values of the controls that we calculated in `data.ctrl` using the indexes `act_left` for the left wheel and `act_right` for the right wheel.
> - **Student TODO V:** After giving the controls, we need to advance our simulation step. We can do that by calling mujoco's `mj_step` method with our `model` and `data` as arguments.

### Run the maze!

If everything went well when you run the script you should first see the maze that RRT is going to explore. If it succeeds, you will see the RRT graph plotted inside the maze and finally the robot navigating the maze in the MuJoCo environment.

In main, you can change hyper-parameters such as the number of max iterations for RRT, how big the maze should be, how many exits it should have etc.