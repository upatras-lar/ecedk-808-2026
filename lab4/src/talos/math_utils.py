import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_traj(q_start, q_end, n_steps):
    alphas = np.linspace(0., 1., n_steps+1)#[1:] # don't repeat the first pose again
    return [q_start + a *(q_end - q_start) for a in alphas]

def generate_foot_arc(start, end, height = 0.05, n_points=20):
    t = np.array([0., 0.5, 1.])

    mid = (start + end) / 2.
    mid[2] += height

    x = [start[0], mid[0], end[0]]
    y = [start[1], mid[1], end[1]]
    z = [start[2], mid[2], end[2]]

    spline_x = CubicSpline(t, x)
    spline_y = CubicSpline(t, y)
    spline_z = CubicSpline(t, z)

    samples = np.linspace(0., 1., n_points)
    trajectory = np.stack([spline_x(samples), spline_y(samples), spline_z(samples)], axis=1)

    return trajectory

def Ryaw(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.array([
        [cos, -sin, 0.],
        [sin, cos, 0.],
        [0., 0., 1.]
    ])


def hat(vec):
    v = vec.reshape((3,))
    return np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ])

def unhat(mat):
    return np.array([[mat[2, 1], mat[0, 2], mat[1, 0]]]).T

def J_l_inv(q, epsilon = 1e-8):
    n = np.linalg.norm(q)
    if n < epsilon:
        return np.eye(3)
    n_sq = n * n
    n_3 = n_sq * n
    c = np.cos(n)
    s = np.sin(n)
    hat_q = hat(q)
    hat_q_sq = hat_q @ hat_q
    JL_inv = np.eye(3) - 0.5 * hat_q + hat_q_sq * (1./n_sq - (1. + c) / (2. * n * s))
    return JL_inv

def log_rotation(R):
    theta = np.arccos(max(-1., min(1., (np.trace(R) - 1.) / 2.)))

    if np.isclose(theta, 0.):
        return np.zeros((3, 1))
    elif np.isclose(theta, np.pi):
        r00 = R[0, 0]
        r11 = R[1, 1]
        r22 = R[2, 2]

        r02 = R[0, 2]
        r12 = R[1, 2]

        r01 = R[0, 1]
        r21 = R[2, 1]

        r10 = R[1, 0]
        r20 = R[2, 0]

        if not np.isclose(r22, -1.):
            multiplier = theta / np.sqrt(2. * (1. + r22))
            return multiplier * np.array([[r02, r12, 1. + r22]]).T
        elif not np.isclose(r11, -1.):
            multiplier = theta / np.sqrt(2. * (1. + r11))
            return multiplier * np.array([[r01, 1. + r11, r21]]).T
        elif not np.isclose(r00, -1.):
            multiplier = theta / np.sqrt(2. * (1. + r00))
            return multiplier * np.array([[1. + r00, r10, r20]]).T

    mat = R - R.T
    r = unhat(mat)

    return theta / (2. * np.sin(theta)) * r

def invert_transformation(T):
    R = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.block([
        [R.T, -R.T @ p.reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1,1))]
        ])

    return T_inv

def log_transformation(T):
    R = T[:3, :3]
    p = T[:3, 3]

    phi = log_rotation(R)
    rho = J_l_inv(phi) @ p

    return np.vstack((phi, rho.reshape(3, 1)))

def normalize_angle(theta):
    # Normalize the angle to a range from -pi to pi
    return np.arctan2(np.sin(theta), np.cos(theta))