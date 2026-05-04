
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import warnings
from .helpers import (
    DEFAULT_MAX_MARKERS,
    LM_DIM,
    ROBOT_DIM,
    lm_slice,
    pose2d_to_Twb,
    wrap_slam_state_angles,
    wrap_to_pi,
)

ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]
MotionModelCallback = Callable[[ArrayF, float, float, float, float, bool], Tuple[ArrayF, ArrayF]]
Detection = Tuple[int, ArrayF]


class EKF:

    def __init__(
        self,
        mu: Optional[npt.ArrayLike] = None,
        Sigma: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        Q: Optional[npt.ArrayLike] = None,
        motion_model_callback: Optional[MotionModelCallback] = None,
    ) -> None:
        self._mu = np.zeros(3, dtype=float) if mu is None else np.array(mu, dtype=float).reshape(3)
        self._mu[2] = wrap_to_pi(self._mu[2])
        self._Sigma = np.eye(3, dtype=float) if Sigma is None else np.array(Sigma, dtype=float).reshape(3, 3)
        self._R = np.diag([0.05, 0.05, 0.02]) if R is None else np.array(R, dtype=float).reshape(3, 3)
        self._Q = np.diag([0.2, 0.2]) if Q is None else np.array(Q, dtype=float).reshape(2, 2)
        if motion_model_callback is None:
            raise ValueError("motion_model_callback is required (no default callback is provided).")
        self._motion_model_callback = motion_model_callback

    def set_motion_model_callback(self, callback: MotionModelCallback) -> None:
        """
        Set custom motion model callback:
            callback(mu, vx, vy, omega, dt, velocities_in_body_frame) -> (mu_pred, F)
        """
        if callback is None:
            raise ValueError("motion_model_callback cannot be None.")
        self._motion_model_callback = callback

    @property
    def motion_model_callback(self) -> MotionModelCallback:
        return self._motion_model_callback

    @property
    def mu(self) -> ArrayF:
        return self._mu.copy()

    @property
    def Sigma(self) -> ArrayF:
        return self._Sigma.copy()

    @property
    def R(self) -> ArrayF:
        return self._R.copy()

    @R.setter
    def R(self, value: npt.ArrayLike) -> None:
        self._R = np.array(value, dtype=float).reshape(3, 3)

    @property
    def Q(self) -> ArrayF:
        return self._Q.copy()

    @Q.setter
    def Q(self, value: npt.ArrayLike) -> None:
        self._Q = np.array(value, dtype=float).reshape(2, 2)

    # Backward-compatible aliases.
    @property
    def state(self) -> ArrayF:
        return self.mu

    @property
    def covariance(self) -> ArrayF:
        return self.Sigma

    @property
    def process_noise(self) -> ArrayF:
        return self.R

    @process_noise.setter
    def process_noise(self, value: npt.ArrayLike) -> None:
        self.R = value

    @property
    def measurement_noise(self) -> ArrayF:
        return self.Q

    @measurement_noise.setter
    def measurement_noise(self, value: npt.ArrayLike) -> None:
        self.Q = value

    def prediction_step(
        self,
        vx: float,
        vy: float,
        omega: float,
        dt: float,
        velocities_in_body_frame: bool = True,
    ) -> None:
        self._mu, F_t = self._motion_model_callback(
            mu=self._mu,
            vx=vx,
            vy=vy,
            omega=omega,
            dt=dt,
            velocities_in_body_frame=velocities_in_body_frame,
        )
        self._Sigma = F_t @ self._Sigma @ F_t.T + self._R

    def correction_step(self, measured_x: float, measured_y: float) -> None:
        z = np.array([float(measured_x), float(measured_y)], dtype=float)
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)

        innovation_t = z - (H @ self._mu)
        S = H @ self._Sigma @ H.T + self._Q
        K = self._Sigma @ H.T @ np.linalg.solve(S, np.eye(2))

        self._mu = self._mu + K @ innovation_t
        self._mu[2] = wrap_to_pi(self._mu[2])

        I = np.eye(3, dtype=float)
        self._Sigma = (I - K @ H) @ self._Sigma

    def reset(self, mu: Optional[npt.ArrayLike] = None, Sigma: Optional[npt.ArrayLike] = None) -> None:
        self._mu = np.zeros(3, dtype=float) if mu is None else np.array(mu, dtype=float).reshape(3)
        self._mu[2] = wrap_to_pi(self._mu[2])
        self._Sigma = np.eye(3, dtype=float) if Sigma is None else np.array(Sigma, dtype=float).reshape(3, 3)


class EKF_SLAM:
    """
    EKF-SLAM for omnidirectional base + planar landmarks (world x, y).

    Measurement (body-frame): z = [r, atan2(p_y^b, p_x^b)], with
      p_b = T_bc @ T_cm @ [0,0,0,1]^T marker origin expressed in robot base/body frame.

    Predicted from map: h(μ) = [sqrt(dx^2+dy^2), atan2(dy,dx) - θ]
      with dx = m_x - x, dy = m_y - y (Probabilistic Robotics convention).

    - eps_m ~ N(0, R): process noise on [x, y, theta]
    - eps_o ~ N(0, Q): measurement noise on [r, phi]; wrap bearing residual to +-pi.
    """

    MIN_RANGE = 1e-6

    def __init__(
        self,
        max_markers: int = DEFAULT_MAX_MARKERS,
        T_bc: Optional[npt.ArrayLike] = None,
        mu: Optional[npt.ArrayLike] = None,
        Sigma: Optional[npt.ArrayLike] = None,
        marker_seen: Optional[npt.ArrayLike] = None,
        R: Optional[npt.ArrayLike] = None,
        Q: Optional[npt.ArrayLike] = None,
        motion_model_callback: Optional[MotionModelCallback] = None,
    ) -> None:
        self.max_markers = int(max_markers)
        self.state_dim = ROBOT_DIM + self.max_markers * LM_DIM
        self.T_bc = np.eye(4, dtype=float) if T_bc is None else np.array(T_bc, dtype=float).reshape(4, 4)

        self._mu = np.zeros(self.state_dim, dtype=float) if mu is None else np.array(mu, dtype=float).reshape(self.state_dim)
        self._mu[2] = wrap_to_pi(self._mu[2])

        if Sigma is None:
            Sigma = np.eye(self.state_dim, dtype=float) * 1e-3
            Sigma[ROBOT_DIM:, ROBOT_DIM:] = np.eye(self.state_dim - ROBOT_DIM, dtype=float) * 1e6
        self._Sigma = np.array(Sigma, dtype=float).reshape(self.state_dim, self.state_dim)

        self._marker_seen = np.zeros(self.max_markers, dtype=bool) if marker_seen is None else np.array(marker_seen, dtype=bool).reshape(self.max_markers)
        self._R = np.diag([0.03**2, 0.03**2, np.deg2rad(2.0) ** 2]) if R is None else np.array(R, dtype=float).reshape(3, 3)
        self._Q = (
            np.diag([0.05**2, np.deg2rad(3.0) ** 2])
            if Q is None
            else np.array(Q, dtype=float).reshape(2, 2)
        )
        if motion_model_callback is None:
            raise ValueError("motion_model_callback is required (no default callback is provided).")
        self._motion_model_callback = motion_model_callback

    @property
    def motion_model_callback(self) -> MotionModelCallback:
        return self._motion_model_callback

    @property
    def mu(self) -> ArrayF:
        return self._mu.copy()

    @property
    def Sigma(self) -> ArrayF:
        return self._Sigma.copy()

    @property
    def marker_seen(self) -> ArrayB:
        return self._marker_seen.copy()

    @property
    def R(self) -> ArrayF:
        return self._R.copy()

    @R.setter
    def R(self, value: npt.ArrayLike) -> None:
        self._R = np.array(value, dtype=float).reshape(3, 3)

    @property
    def Q(self) -> ArrayF:
        return self._Q.copy()

    @Q.setter
    def Q(self, value: npt.ArrayLike) -> None:
        self._Q = np.array(value, dtype=float).reshape(2, 2)

    def set_motion_model_callback(self, callback: MotionModelCallback) -> None:
        if callback is None:
            raise ValueError("motion_model_callback cannot be None.")
        self._motion_model_callback = callback

    def landmark_slice(self, marker_index: int) -> slice:
        return lm_slice(marker_index, robot_dim=ROBOT_DIM)

    @staticmethod
    def measurement_body_from_T_cm(T_bc: npt.ArrayLike, T_cm_meas: npt.ArrayLike) -> ArrayF:
        """z = [r, phi_meas] body-frame range and bearing to marker."""
        pb = T_bc @ T_cm_meas @ np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        pbx, pby = float(pb[0]), float(pb[1])
        r = float(np.hypot(pbx, pby))
        phi = float(np.arctan2(pby, pbx))
        return np.array([r, phi], dtype=float)

    def predicted_range_bearing(self, mu: npt.ArrayLike, marker_index: int) -> ArrayF:
        """h(μ) = [range, atan2(dy,dx) - θ] consistent with measurement."""
        mu = np.array(mu, dtype=float).reshape(self.state_dim)
        x_r, y_r, th_r = mu[0], mu[1], mu[2]
        sl = self.landmark_slice(marker_index)
        mx, my = mu[sl.start], mu[sl.start + 1]
        ##### ENTER CODE HERE #####

        ###########################

    def measurement_residual(self, mu: npt.ArrayLike, marker_index: int, T_cm_meas: npt.ArrayLike) -> ArrayF:
        ##### ENTER CODE HERE #####

        ###########################

    def analytic_measurement_jacobian(self, mu: npt.ArrayLike, marker_index: int) -> ArrayF:
        """
        H = ∂h/∂μ for h = [r, atan2(dy,dx) - θ], non-zero on robot + this landmark xy only.
        """
        mu = np.array(mu, dtype=float).reshape(self.state_dim)
        x_r, y_r = mu[0], mu[1]
        sl = self.landmark_slice(marker_index)
        mx, my = mu[sl.start], mu[sl.start + 1]
        ##### ENTER CODE HERE #####

        ###########################

    def initialize_landmark(
        self,
        mu: npt.ArrayLike,
        Sigma: npt.ArrayLike,
        marker_index: int,
        T_cm_meas: npt.ArrayLike,
    ) -> Tuple[ArrayF, ArrayF]:
        T_wb = pose2d_to_Twb(mu[0], mu[1], mu[2])
        T_wc = T_wb @ self.T_bc
        ##### ENTER CODE HERE #####

        ###########################

    def prediction_step(self, u_body: np.ndarray, dt: float) -> None:
        vx, vy, omega = np.array(u_body, dtype=float).reshape(3)
        mu_robot_bar, F_r = self._motion_model_callback(
            mu=self._mu[:3],
            vx=vx,
            vy=vy,
            omega=omega,
            dt=dt,
            velocities_in_body_frame=True,
        )

        ##### ENTER CODE HERE #####

        ###########################

    def correction_step(self, detections: Iterable[Detection]) -> None:
        I = np.eye(self.state_dim, dtype=float)
        mu_bar = self._mu.copy()
        Sigma_bar = self._Sigma.copy()

        for marker_id, T_cm_meas in detections:
            if marker_id < 0 or marker_id >= self.max_markers:
                warnings.warn(
                    (
                        f"Ignoring detection with invalid marker_id={marker_id}. "
                        f"Valid range is [0, {self.max_markers - 1}]."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            if not self._marker_seen[marker_id]:
                ##### ENTER CODE HERE #####

                ###########################
                continue

            ##### ENTER CODE HERE #####

            ###########################

        self._mu = mu_bar
        self._Sigma = Sigma_bar

    def step(self, u_body: npt.ArrayLike, detections: Iterable[Detection], dt: float) -> None:
        #event based prediction
        self.prediction_step(u_body, dt)

        #event based correction
        self.correction_step(detections)