import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


# ---------- Damped Harmonic Oscillator ----------
print("\nTest Damped Harmonic Oscillator:\n")

# Parameters

m = 1.0       # mass
c = 0.2       # damping coefficient
k = 2.0       # spring constant

dt = 0.05     # sampling time
T = 50.0      # total simulation time
N = int(T / dt)

A = np.array([
    [0.0, 1.0],
    [-k/m, -c/m]
])

B = np.array([
    [0],
    [1/m]
])

# Discrete-time model

A_inv = (m/k) * np.array([[-c/m, -1.0], [k/m, 0.0]])
sqrt = np.sqrt(complex(c**2 - 4.0*k*m))
l1 = (-c-sqrt)/(2.0*m)
l2 = (-c+sqrt)/(2.0*m)

A_d = np.real_if_close(m/sqrt * np.array([[l2 * np.exp(l1*dt) - l1 * np.exp(l2*dt), np.exp(l2*dt) - np.exp(l1*dt)], [l1 * l2 * (np.exp(l1*dt) - np.exp(l2*dt)), l2 * np.exp(l2*dt) - l1 * np.exp(l1*dt)]]))
B_d = A_inv @ (A_d - np.eye(2)) @ B

A_d_approx = np.array([
    [1.0, dt],
    [-(k/m) * dt, 1.0 - (c/m) * dt]
])

B_d_approx = np.array([
    [0.0],
    [dt / m]
])

print("A_d (mine) = \n", A_d)
print("A_d (exact) = \n", expm(A*dt))
print("Aapprox_d = \n", A_d_approx)
print("B_d = \n", B_d)
print("Bapprox_d = \n", B_d_approx)


# ---------- Cart-Pole ----------
print("\nTest Cart-Pole:\n")

# Parameters

M = 5.0       # cart mass
m = 0.2       # pole mass
l = 0.5       # pole COM length
g = 9.81      # gravity

dt = 0.01
T = 30.0
N = int(T / dt)

def df_dx_1(x, u):
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

def df_dx_2(x, u):
    p, p_dot, theta, omega = x

    s = np.sin(theta)
    s2 = np.sin(2*theta)
    c = np.cos(theta)
    D = M + m * s**2
    
    df2_dtheta = (m*(M-m*s**2)*(g+l*omega**2*c)-2*m*M*g*s**2-m*s2*u) / D**2
    df2_domega = (2*m*l*s*omega) / D
    df4_dtheta = (-(m*l*omega**2+g*(m+M)*c)*(M-m*s**2)+2*m*l*M*omega**2*s**2+s*(M+m+m*c**2)*u) / (l*D**2)
    df4_domega = -(m*s2*omega) / D

    A = np.array([
        [0.0, 1.0,         0.0,         0.0],
        [0.0, 0.0, df2_dtheta,  df2_domega],
        [0.0, 0.0,         0.0,         1.0],
        [0.0, 0.0, df4_dtheta,  df4_domega]
    ])

    return A

x = 1000*np.random.random((4,))
u = 1000*np.random.random()
print("A_error = \n", df_dx_1(x, u) - df_dx_2(x, u))
