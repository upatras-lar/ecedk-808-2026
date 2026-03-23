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