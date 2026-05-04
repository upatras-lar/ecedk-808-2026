from __future__ import annotations

import numpy as np
from ekf_slam import EKF_SLAM


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def motion_model_callback(
    mu: np.ndarray,
    vx: float,
    vy: float,
    omega: float,
    dt: float,
    velocities_in_body_frame: bool = True,
):
    x, y, theta = np.asarray(mu, dtype=float).reshape(3)

    F = np.eye(3)

    if velocities_in_body_frame:
        c = np.cos(theta)
        s = np.sin(theta)

        dx_world = (c * vx - s * vy) * dt
        dy_world = (s * vx + c * vy) * dt

        x_pred = x + dx_world
        y_pred = y + dy_world

        F[0, 2] = (-s * vx - c * vy) * dt
        F[1, 2] = (c * vx - s * vy) * dt
    else:
        x_pred = x + vx * dt
        y_pred = y + vy * dt

    theta_pred = wrap_to_pi(theta + omega * dt)

    return np.array([x_pred, y_pred, theta_pred], dtype=float), F


def make_T_cm_from_body_xy(x_body: float, y_body: float) -> np.ndarray:
    T_cm = np.eye(4)
    T_cm[0, 3] = x_body
    T_cm[1, 3] = y_body
    return T_cm


def propagate_true_state(mu: np.ndarray, u_body: np.ndarray, dt: float) -> np.ndarray:
    mu_next, _ = motion_model_callback(
        mu=mu,
        vx=float(u_body[0]),
        vy=float(u_body[1]),
        omega=float(u_body[2]),
        dt=dt,
        velocities_in_body_frame=True,
    )
    return mu_next


def simulate_marker_detections(
    true_pose: np.ndarray,
    landmarks_world: np.ndarray,
    rng: np.random.Generator,
    max_range: float = 4.0,
    fov_rad: float = np.deg2rad(160.0),
    range_std: float = 0.03,
    bearing_std: float = np.deg2rad(2.0),
):
    """
    Simulate marker detections as EKF_SLAM expects them:

        detection = (marker_id, T_cm_meas)

    Since T_bc = identity in this test, T_cm_meas contains the marker position
    in the robot/body frame.
    """
    x, y, theta = true_pose
    c = np.cos(theta)
    s = np.sin(theta)

    # World-to-body rotation is R(theta).T
    R_bw = np.array(
        [
            [c, s],
            [-s, c],
        ],
        dtype=float,
    )

    detections = []

    for marker_id, landmark_w in enumerate(landmarks_world):
        delta_w = landmark_w - np.array([x, y])
        delta_b = R_bw @ delta_w

        r = float(np.linalg.norm(delta_b))
        bearing = float(np.arctan2(delta_b[1], delta_b[0]))

        if r > max_range:
            continue

        if abs(bearing) > fov_rad / 2.0:
            continue

        noisy_r = r + rng.normal(0.0, range_std)
        noisy_bearing = bearing + rng.normal(0.0, bearing_std)

        noisy_x_body = noisy_r * np.cos(noisy_bearing)
        noisy_y_body = noisy_r * np.sin(noisy_bearing)

        T_cm_meas = make_T_cm_from_body_xy(noisy_x_body, noisy_y_body)
        detections.append((marker_id, T_cm_meas))

    return detections


def landmark_rmse(slam: EKF_SLAM, landmarks_world: np.ndarray) -> float:
    mu = slam.mu
    seen = slam.marker_seen

    errors = []

    for marker_id, landmark_true in enumerate(landmarks_world):
        if not seen[marker_id]:
            continue

        sl = slam.landmark_slice(marker_id)
        landmark_est = mu[sl.start : sl.stop]
        errors.append(np.linalg.norm(landmark_est - landmark_true))

    if not errors:
        return np.inf

    return float(np.sqrt(np.mean(np.square(errors))))


def run_ekf_slam_simulation(seed: int = 7, do_print: bool = True):
    rng = np.random.default_rng(seed)

    landmarks_world = np.array(
        [
            [2.0, 1.0],
            [4.0, 0.0],
            [3.0, 2.5],
            [0.5, 3.0],
            [-1.0, 2.0],
            [1.5, -1.5],
        ],
        dtype=float,
    )

    slam = EKF_SLAM(
        max_markers=len(landmarks_world),
        T_bc=np.eye(4),
        motion_model_callback=motion_model_callback,
    )

    # Slightly tune noise for the synthetic test.
    slam.R = np.diag([0.02**2, 0.02**2, np.deg2rad(1.0) ** 2])
    slam.Q = np.diag([0.04**2, np.deg2rad(2.0) ** 2])

    true_pose = np.array([0.0, 0.0, 0.0], dtype=float)

    dt = 0.1
    n_steps = 300

    for k in range(n_steps):
        # A gentle curved trajectory that repeatedly observes landmarks.
        u_true = np.array(
            [
                0.25,
                0.03 * np.sin(0.05 * k),
                0.20 * np.sin(0.025 * k),
            ],
            dtype=float,
        )

        true_pose = propagate_true_state(true_pose, u_true, dt)

        # Odometry/control noise.
        u_odom = u_true + rng.normal(
            loc=0.0,
            scale=np.array([0.01, 0.01, np.deg2rad(0.5)]),
            size=3,
        )

        detections = simulate_marker_detections(
            true_pose=true_pose,
            landmarks_world=landmarks_world,
            rng=rng,
            max_range=4.0,
            fov_rad=np.deg2rad(170.0),
            range_std=0.03,
            bearing_std=np.deg2rad(2.0),
        )

        slam.step(u_body=u_odom, detections=detections, dt=dt)

        mu = slam.mu
        Sigma = slam.Sigma

        # Basic numerical checks at every step.
        assert np.all(np.isfinite(mu)), f"Non-finite state at step {k}"
        assert np.all(np.isfinite(Sigma)), f"Non-finite covariance at step {k}"
        assert np.allclose(Sigma, Sigma.T, atol=1e-8), f"Covariance not symmetric at step {k}"

        eigvals = np.linalg.eigvalsh(Sigma)
        assert eigvals.min() > -1e-7, f"Covariance not PSD at step {k}: min eig={eigvals.min()}"

    est_pose = slam.mu[:3]
    pose_xy_error = float(np.linalg.norm(est_pose[:2] - true_pose[:2]))
    pose_theta_error = abs(wrap_to_pi(float(est_pose[2] - true_pose[2])))
    map_rmse = landmark_rmse(slam, landmarks_world)

    num_seen = int(np.count_nonzero(slam.marker_seen))

    if do_print:
        print()
        print("=== EKF-SLAM simulation result ===")
        print(f"True pose:      {true_pose}")
        print(f"Estimated pose: {est_pose}")
        print(f"Pose xy error:  {pose_xy_error:.3f} m")
        print(f"Pose yaw error: {np.rad2deg(pose_theta_error):.2f} deg")
        print(f"Seen markers:   {num_seen}/{len(landmarks_world)}")
        print(f"Map RMSE:       {map_rmse:.3f} m")

    assert num_seen >= 4, "Too few landmarks were initialized"
    assert pose_xy_error < 0.50, f"Pose xy error too large: {pose_xy_error:.3f} m"
    assert pose_theta_error < np.deg2rad(15.0), (
        f"Pose theta error too large: {np.rad2deg(pose_theta_error):.2f} deg"
    )
    assert map_rmse < 0.75, f"Landmark map RMSE too large: {map_rmse:.3f} m"

    return slam, true_pose, landmarks_world


def test_ekf_slam_runs_without_ros_stack():
    run_ekf_slam_simulation(seed=7, do_print=False)


if __name__ == "__main__":
    run_ekf_slam_simulation(seed=7, do_print=True)