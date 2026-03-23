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