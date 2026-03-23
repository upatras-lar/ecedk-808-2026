# Robotic Systems I – Lab 2
## State Estimation with Kalman Filters
### Introduction

So far in this course, we have focused on **how robots move and plan**.  
In this lab, we shift our focus to another fundamental question:

> *How does a robot know where it is?*

In real-world systems, measurements are **noisy**, incomplete, and sometimes misleading.  
To deal with this, we use **state estimation algorithms**.

The lab is divided into two parts:

- Part I – **Kalman Filter**. We will estimate the state of a **mass-spring-damper system**.
- Part II – **Extended Kalman Filter**. We will estimate the state of a **cart-pole system**, which is nonlinear.

## Part I – Kalman Filter (Linear Systems)


### System Description

We consider a **damped harmonic oscillator**:

- State:
  $x = [position, velocity]$

- We can **only measure position**.

- The system evolves according to a **linear discrete-time model**:

$$x_{k+1} = A x_k + B u_k + w_k$$

$$z_k = C x_k + v_k$$

Where:

- $w_k$: process noise  
- $v_k$: measurement noise  

### Kalman Filter Algorithm

At each timestep, the filter performs two steps:

#### 1. Prediction

Given the previous estimation $\mu_{k-1}$ and covariance $\Sigma_{k-1}$ along with the control $u_k$, we make a prediction about the new values of the estimation ($\bar{\mu}_k$) and covariance ($\bar{\Sigma}_k$).

$$\bar{\mu}_k = A_k \mu_{k-1} + B_k u_k$$

$$\bar{\Sigma}_k = A_k \Sigma_{k-1} A_k^T + R_k$$

#### 2. Correction

In this step using the measurement $z_k$ we try to correct our previous predictions $\mu_k$ and $\Sigma_k$.

$$K_k = \bar{\Sigma}_k C_k^T (C_k \bar{\Sigma}_t C_k^T + Q_t)^{-1}$$

$$\mu_k = \bar{\mu}_k + K_k(z_k - C_k \bar{\mu}_k)$$

$$\Sigma_k = (I - K_k C_k)\bar{\Sigma}_k$$

```python
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Parameters

m = 1.0       # mass
c = 0.4       # damping coefficient
k = 2.0       # spring constant

dt = 0.01     # sampling time
T = 50.0      # total simulation time
N = int(T / dt)


# Discrete-time model

# exact
A = np.array([
    [0.0, 1.0],
    [-k/m, -c/m]
])
A_inv = (m/k) * np.array([[-c/m, -1.0], [k/m, 0.0]])
B = np.array([
    [0],
    [1/m]
])

A_d = expm(A*dt)
B_d = A_inv @ (A_d - np.eye(2)) @ B

C = np.array([[1.0, 0.0]])   # measure position only

# Process noise covariance
##### ENTER CODE HERE #####

###########################

# Measurement noise covariance
##### ENTER CODE HERE #####

###########################

#  Input signal

##### ENTER CODE HERE #####

###########################

# Simulate true system

x_true = np.zeros((N + 1, 2))
z = np.zeros(N + 1)
u = np.zeros(N + 1)

# Initial true state: displaced, zero velocity
x_true[0, :] = np.array([1.0, 0.0])

rng = np.random.default_rng(42)

for k_step in range(N):
    t = k_step * dt
    u_t = control_input(t)
    u[k_step] = u_t

    ##### ENTER CODE HERE #####

    ###########################

# measurement noise
for k_step in range(N + 1):
    v_k = rng.normal(loc=0.0, scale=np.sqrt(Q[0, 0]))
    ##### ENTER CODE HERE #####

    ###########################

# Kalman filter

mu = np.zeros((N + 1, 2))
Sigma = np.zeros((N + 1, 2, 2))

# Initial estimate
mu[0, :] = np.array([0.0, 0.0])

# Initial covariance
Sigma[0, :, :] = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

I = np.eye(2)

for k_step in range(1, N + 1):
    # Prediction step
    ##### ENTER CODE HERE #####

    ###########################

    # Correction step
    ##### ENTER CODE HERE #####

    ###########################

# Plot results

time = np.linspace(0.0, T, N + 1)

plt.figure(figsize=(10, 5))
plt.plot(time, x_true[:, 0], label="True position", c="blue")
plt.scatter(time, z, s=2, label="Measured position", alpha=0.3, c="green")
plt.plot(time, mu[:, 0], label="Estimated position", alpha=0.8, c="red")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.title("Kalman Filter for Damped Harmonic Oscillator")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, x_true[:, 1], label="True velocity", c="blue")
plt.plot(time, mu[:, 1], label="Estimated velocity", alpha=0.8, c="red")
plt.xlabel("Time [s]")
plt.ylabel("Velocity")
plt.title("Velocity Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Animation: true system vs KF estimate

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-1.5 * np.max(np.abs(x_true[:,0])), 1.5 * np.max(np.abs(x_true[:,0])))
ax.set_ylim(-0.2, 0.2)
ax.set_xlabel("Position")
ax.set_yticks([])
ax.set_title("Damped Harmonic Oscillator: True vs Estimated")

# Rectangle size
mass_width = 0.1
mass_height = 0.05

# True mass rectangle
mass_true = Rectangle(
    (x_true[0, 0] - mass_width/2, -mass_height/2),
    mass_width,
    mass_height,
    color='blue',
    label='True'
)
ax.add_patch(mass_true)

# Estimated mass rectangle
mass_est = Rectangle(
    (mu[0, 0] - mass_width/2, -mass_height/2),
    mass_width,
    mass_height,
    color='orange',
    alpha=0.8,
    label='Estimate'
)
ax.add_patch(mass_est)

# Line connecting equilibrium to true/estimated mass
line_true, = ax.plot([0, x_true[0, 0]], [0, 0], 'b--', alpha=0.5)
line_est, = ax.plot([0, mu[0, 0]], [0, 0], 'orange', alpha=0.5)

# Time label
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def init():
    mass_true.set_x(x_true[0, 0] - mass_width/2)
    mass_est.set_x(mu[0, 0] - mass_width/2)
    line_true.set_data([0, x_true[0, 0]], [0, 0])
    line_est.set_data([0, mu[0, 0]], [0, 0])
    time_text.set_text('')
    return mass_true, mass_est, line_true, line_est, time_text

frame_skip = 1  # show every frame

def update(frame):
    # Update true
    mass_true.set_x(x_true[frame, 0] - mass_width/2)
    line_true.set_data([0, x_true[frame, 0]], [0, 0])
    
    # Update estimate
    mass_est.set_x(mu[frame, 0] - mass_width/2)
    line_est.set_data([0, mu[frame, 0]], [0, 0])
    
    # Update time
    time_text.set_text(f"t = {frame*dt:.2f}s")
    
    return mass_true, mass_est, line_true, line_est, time_text

ani = FuncAnimation(
    fig, update, frames=range(0, N+1, frame_skip),
    init_func=init, blit=True, interval=dt*1000*frame_skip
)

# Pause/Resume with spacebar
paused = False
def on_key(event):
    global paused
    if event.key == ' ':
        if paused:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        paused = not paused

fig.canvas.mpl_connect('key_press_event', on_key)

ax.legend()
plt.tight_layout()
plt.show()
```
> - **Student TODO I:** We will start by creating the process/motion noise array `R` and measurement noise array `Q`. Let's start by giving values only on the diagonal entries. 1e-3 is good to start with, we can play with it later.
> - **Student TODO II:** Let's also define a function that will give as our control signal. We will call it `control_input`. It takes the current time `t` as an argument and returns the value of the control at that time. you can choose to give it a constant control, or a time variant (like a sinusoidal).
> - **Student TODO III:** Given the control function that we implemented in the previous step and the initial starting point of the system, we would like to calculate what the ground truth `x_true` is at every time step. We already now the system (through the matrices `A_d` and `B_d`).
> - **Student TODO IV:** Now that we have calculated the ground truth `x_true` at each timestep and since we know the matrix `C`, we can "make" our observations array `z`. Don't forget to add the measurement noise!
> - **Student TODO V:** In the prediction phase of the Kalman filter algorithm, let's calculate our guesses `mu_bar` and `Sigma_bar` using the equations earlier in this section.
> - **Student TODO VI:** In the correction phase of the algorithm, let's first calculate the Kalman gain `K` and then make our corrected estimates `mu` and `Sigma` for this timestep.

## Part II – Extended Kalman Filter (Nonlinear Systems)

The standard Kalman Filter only works for linear systems. However, most robotic systems are nonlinear. To handle this, we use the **Extended Kalman Filter (EKF)**.

### System Description – Cart-Pole

We now consider a **cart-pole system**:

- State:
  x = [p, p_dot, theta, theta_dot]

Where:

- p: cart position  
- theta: pole angle  

Unlike the previous exercise here, the system dynamics are nonlinear.

$$x_{k+1} = f(x_k, u_k)$$

We linearize using a **Jacobian**

### EKF Algorithm

#### 1. Prediction

We use the non-linear system dynamics to with our previous estimation $\mu_{k-1}$ and current control $u_k$ to compute a new prediction for the state:

$$\bar{\mu}_k = f(\bar{\mu}_{k-1}, u_k)$$

We also use the previous covariance $\Sigma_{k-1}$ and the Jacobian of the dynamics with respect to the state ($F_t = \frac{\partial f}{\partial x}$) to predict the covariance:

$$\bar{\Sigma}_k = F_k \bar{\Sigma}_{k-1} F_k^T + R_k$$

### 2. Correction

Now using our measuremnt we can compute the Kalman gain and correct our previouse "guesses":

$$K_k = \bar{\Sigma}_k H_k^T (H_k \bar{\Sigma}_k H_k^T + Q_k)^{-1}$$

$$\mu = \bar{\mu}_k + K_k(z_k - H_k \bar{\mu}_k)$$

$$\Sigma = (I - K_k H_k)\bar{\Sigma}_k$$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Parameters

M = 5.0       # cart mass
m = 0.2       # pole mass
l = 0.5       # pole COM length
g = 9.81      # gravity

dt = 0.01
T = 50.0
N = int(T / dt)

# Measurement model

H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

# Process noise covariance
##### ENTER CODE HERE #####

###########################

# Measurement noise covariance
##### ENTER CODE HERE #####

###########################

# Input signal

# def control_input(t):
#     return 0.5 * np.sin(0.5 * t)

def control_input(t):
    return 0.

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Nonlinear dynamics
# x = [p, p_dot, theta, theta_dot]
def dynamics(x, u):
    p, p_dot, theta, theta_dot = x

    s = np.sin(theta)
    c = np.cos(theta)
    D = M + m * s**2

    p_ddot = (u + m * s * (l * theta_dot**2 + g * c)) / D

    theta_ddot = (
        -u * c
        - m * l * theta_dot**2 * s * c
        - (M + m) * g * s
    ) / (l * D)

    return np.array([p_dot, p_ddot, theta_dot, theta_ddot])

# Analytic Jacobian of continuous dynamics

def df_dx(x, u):
    p, p_dot, theta, omega = x

    s = np.sin(theta)
    c = np.cos(theta)
    D = M + m * s**2
    D_theta = 2.0 * m * s * c

    N2 = u + m * s * (l * omega**2 + g * c)
    N2_theta = m * (c * l * omega**2 + g * (c**2 - s**2))
    df2_dtheta = (N2_theta * D - N2 * D_theta) / (D**2)
    df2_domega = (2.0 * m * l * s * omega) / D

    N4 = -u * c - m * l * omega**2 * s * c - (M + m) * g * s
    N4_theta = (
        u * s
        - m * l * omega**2 * (c**2 - s**2)
        - (M + m) * g * c
    )
    df4_dtheta = (N4_theta * D - N4 * D_theta) / (l * D**2)
    df4_domega = (-2.0 * m * omega * s * c) / D

    A = np.array([
        [0.0, 1.0,         0.0,         0.0],
        [0.0, 0.0, df2_dtheta,  df2_domega],
        [0.0, 0.0,         0.0,         1.0],
        [0.0, 0.0, df4_dtheta,  df4_domega]
    ])

    return A

# Runge-Kutta 4th order (RK4) integration

##### ENTER CODE HERE #####

###########################

# Jacobian of discrete dynamics using RK4

def F(x, u):
    f1 = dynamics(x, u)
    f2 = dynamics(x+dt/2.*f1, u)
    f3 = dynamics(x+dt/2.*f2, u)
    I = np.eye(x.shape[0])
    df1_dx = df_dx(x, u)
    df2_dx = df_dx(x+dt/2.*f1, u) @ (I + dt/2.*df1_dx)
    df3_dx = df_dx(x+dt/2.*f2, u) @ (I + dt/2.*df2_dx)
    df4_dx = df_dx(x+dt*f3, u) @ (I + dt*df3_dx)
    F = I + dt/6.*(df1_dx + 2.*df2_dx + 2.*df3_dx + df4_dx)
    return F

# Simulate true system

x_true = np.zeros((N + 1, 4))
z = np.zeros((N + 1, 2))
u = np.zeros(N + 1)

x_true[0, :] = np.array([0., 0., np.pi/4, 0.])

rng = np.random.default_rng(42)

for k_step in range(N):
    t = k_step * dt
    u_t = control_input(t)
    u[k_step] = u_t

    ##### ENTER CODE HERE #####

    ###########################

for k_step in range(N + 1):
    v_k = rng.multivariate_normal(mean=np.zeros(2), cov=Q)
    ##### ENTER CODE HERE #####

    ###########################

# EKF

mu = np.zeros((N + 1, 4))
Sigma = np.zeros((N + 1, 4, 4))

mu[0, :] = np.array([0.0, 0.0, 0.0, 0.0])
Sigma[0, :, :] = np.diag([1.0, 1.0, 1.0, 1.0])

I = np.eye(4)

for k_step in range(1, N + 1):
    # Prediction
    ##### ENTER CODE HERE #####

    ###########################

    # Correction
    ##### ENTER CODE HERE #####

    ###########################

# Plot results

time = np.linspace(0.0, T, N + 1)

plt.figure(figsize=(10, 5))
plt.plot(time, x_true[:, 0], label="True cart position", c="blue")
plt.scatter(time, z[:, 0], s=1, label="Measured cart position", alpha=0.3, c="green")
plt.plot(time, mu[:, 0], label="Estimated cart position", alpha=0.6, c="red")
plt.xlabel("Time [s]")
plt.ylabel("Cart position")
plt.title("EKF for Cart-Pole: Cart Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, x_true[:, 1], label="True cart velocity", c="blue")
plt.plot(time, mu[:, 1], label="Estimated cart velocity", alpha=0.8, c="red")
plt.xlabel("Time [s]")
plt.ylabel("Cart velocity")
plt.title("EKF for Cart-Pole: Cart Velocity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, wrap_to_pi(x_true[:, 2]), label="True pole angle", c="blue")
plt.scatter(time, wrap_to_pi(z[:, 1]), s=1, label="Measured pole angle", alpha=0.3, c="green")
plt.plot(time, wrap_to_pi(mu[:, 2]), label="Estimated pole angle", alpha=0.8, c="red")
plt.xlabel("Time [s]")
plt.ylabel("Pole angle [rad]")
plt.title("EKF for Cart-Pole: Pole Angle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time, x_true[:, 3], label="True pole angular velocity", c="blue")
plt.plot(time, mu[:, 3], label="Estimated pole angular velocity", alpha = 0.8, c="red")
plt.xlabel("Time [s]")
plt.ylabel("Pole angular velocity [rad/s]")
plt.title("EKF for Cart-Pole: Pole Angular Velocity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Animation: true system vs EKF estimate

fig, ax = plt.subplots(figsize=(10, 6))

# Visualization parameters
cart_width = 0.4
cart_height = 0.2
pole_length = l * 2.0
y_cart = 0.0

# Limits
all_cart_positions = np.concatenate([x_true[:, 0], mu[:, 0]])
x_min = np.min(all_cart_positions) - 1.0
x_max = np.max(all_cart_positions) + 1.0

ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.3 * pole_length, 1.3 * pole_length)
ax.set_aspect('equal')
ax.set_xlabel("Position")
ax.set_yticks([])
ax.set_title("Cart-Pole: True vs EKF Estimate")

# Ground line
ax.axhline(0.0, color='black', linewidth=1)

# TRUE SYSTEM (blue)
cart_true = Rectangle(
    (x_true[0, 0] - cart_width/2, y_cart - cart_height/2),
    cart_width,
    cart_height,
    color='blue',
    label='True'
)
ax.add_patch(cart_true)

pole_true, = ax.plot([], [], 'b-', linewidth=3)
pivot_true, = ax.plot([], [], 'bo')

# ESTIMATED SYSTEM (orange)
cart_est = Rectangle(
    (mu[0, 0] - cart_width/2, y_cart - cart_height/2),
    cart_width,
    cart_height,
    color='orange',
    alpha=0.7,
    label='Estimate'
)
ax.add_patch(cart_est)

pole_est, = ax.plot([], [], color='orange', linewidth=3, alpha=0.7)
pivot_est, = ax.plot([], [], 'o', color='orange', alpha=0.7)

# Lines showing cart position difference
line_true, = ax.plot([0, x_true[0,0]], [0, 0], 'b--', alpha=0.3)
line_est, = ax.plot([0, mu[0,0]], [0, 0], color='orange', alpha=0.3)

# Time label
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def cartpole_points(state):
    p, _, theta, _ = state

    x_pivot = p
    y_pivot = y_cart

    x_tip = x_pivot + pole_length * np.sin(theta)
    y_tip = y_pivot - pole_length * np.cos(theta)

    return x_pivot, y_pivot, x_tip, y_tip

def init():
    pole_true.set_data([], [])
    pivot_true.set_data([], [])

    pole_est.set_data([], [])
    pivot_est.set_data([], [])

    time_text.set_text('')

    return (
        cart_true, pole_true, pivot_true,
        cart_est, pole_est, pivot_est,
        line_true, line_est,
        time_text
    )

def update(frame):
    # Update true
    x_pivot, y_pivot, x_tip, y_tip = cartpole_points(x_true[frame, :])
    cart_true.set_x(x_true[frame, 0] - cart_width/2)
    pole_true.set_data([x_pivot, x_tip], [y_pivot, y_tip])
    pivot_true.set_data([x_pivot], [y_pivot])

    line_true.set_data([0, x_true[frame,0]], [0, 0])

    # Update estimate
    x_pivot_e, y_pivot_e, x_tip_e, y_tip_e = cartpole_points(mu[frame, :])
    cart_est.set_x(mu[frame, 0] - cart_width/2)
    pole_est.set_data([x_pivot_e, x_tip_e], [y_pivot_e, y_tip_e])
    pivot_est.set_data([x_pivot_e], [y_pivot_e])
    line_est.set_data([0, mu[frame,0]], [0, 0])

    # Update time
    time_text.set_text(f"t = {frame * dt:.2f}s")

    return (
        cart_true, pole_true, pivot_true,
        cart_est, pole_est, pivot_est,
        line_true, line_est,
        time_text
    )

frame_skip = 5

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, N+1, frame_skip),
    init_func=init,
    blit=True,
    interval=dt * 1000 * frame_skip
)

# Pause/resume (spacebar)
paused = False
def on_key(event):
    global paused
    if event.key == ' ':
        if paused:
            ani.event_source.start()
        else:
            ani.event_source.stop()
        paused = not paused

fig.canvas.mpl_connect('key_press_event', on_key)

ax.legend()
plt.tight_layout()
plt.show()
```
> - **Student TODO I:** We will start by creating the process/motion noise array `R` and measurement noise array `Q`. Let's start by giving values only on the diagonal entries. 1e-5 is good to start with, we can play with it later.
> - **Student TODO II:** Unlike the previous exercise, the cartpole system is highly non-linear. Using a simple Euler integration will likely blow up this time. So we will implement the 4th order Runge and Kutta integration method. Let's make a function called `runge_kutta_4th_order` that takes `xk`, `uk` and `dt` as arguments and returns the next state using the 4th order runge and kutta method.
> - **Student TODO III:** Given the integration function that we created in the previous step and the initial starting point of the system, we would like to calculate what the ground truth `x_true` is at every time step.
> - **Student TODO IV:** Now that we have calculated the ground truth `x_true` at each timestep and since we know the matrix `H`, we can "make" our observations array `z`. Don't forget to add the measurement noise!
> - **Student TODO V:** In the prediction phase of the EKF algorithm, let's calculate our guesses `mu_bar` and `Sigma_bar` using the equations earlier in this section.
> - **Student TODO VI:** In the correction phase of the algorithm, let's first calculate the Kalman gain `K` and then make our corrected estimates `mu` and `Sigma` for this timestep.