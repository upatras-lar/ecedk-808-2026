# Robotic Systems I – Lab 3
## EKF-SLAM
### Introduction

In the previous lab, we used the **Kalman Filter** and the **Extended Kalman Filter** to estimate the state of a system.

In this lab, we move one step further. Instead of only estimating where the robot is, we will also estimate where the landmarks around the robot are.

This problem is called **SLAM (Simultaneous Localization and Mapping)**. The robot needs to answer two questions at the same time:

> *Where am I?*  
> *Where are the landmarks?*

The code you complete today will be used inside a ROS stack and will run on an actual omnidirectional robot. However, for this lab you do **not** need to worry about ROS, cameras, messages, or deployment.

Your task is only to complete the missing parts of the **EKF-SLAM** code.

## EKF-SLAM System Description

We consider a mobile robot moving in a 2D world.

The robot state is:

$$x_r = [x, y, \theta]$$

Where:

- $x$: robot position in the world x-axis
- $y$: robot position in the world y-axis
- $\theta$: robot orientation

The map contains 2D landmarks. Each landmark has state:

$$m_i = [m_{i,x}, m_{i,y}]$$

The full EKF-SLAM state is the robot pose together with all landmark positions:

$$\mu = [x, y, \theta, m_{0,x}, m_{0,y}, m_{1,x}, m_{1,y}, ...]^T$$

So, unlike the EKF from last week, the state vector is now much larger. It contains both the robot and the map.

## Measurements

The robot observes visual markers. Each observation gives us a relative measurement from the robot/camera frame to the marker.

For the SLAM update, we will use a range and bearing measurement:

$$z = [r, \phi]$$

Where:

- $r$: distance from the robot to the landmark
- $\phi$: bearing angle from the robot to the landmark

For a landmark $m_i = [m_x, m_y]$ and robot pose $[x, y, \theta]$, define:

$$dx = m_x - x$$

$$dy = m_y - y$$

The predicted measurement is:

$$h(\mu) =
\begin{bmatrix}
\sqrt{dx^2 + dy^2} \\
\text{atan2}(dy, dx) - \theta
\end{bmatrix}$$

The second part is an angle, so whenever we compute an angle difference we need to wrap it to the interval $[-\pi, \pi]$.

## EKF-SLAM Algorithm

Like before, the algorithm has two main steps.

### 1. Prediction

The robot moves according to a motion model:

$$\bar{\mu}_{r,k} = f(\mu_{r,k-1}, u_k)$$

The landmarks do not move. This means that during prediction we only update the robot part of the state.

However, the covariance belongs to the full SLAM state, so we need a big Jacobian matrix $F_t$ with the robot Jacobian in the top-left block and identity for everything else.

$$\bar{\Sigma}_k = F_t \Sigma_{k-1} F_t^T + R$$

In our code, the process noise $R$ is only added to the robot part of the covariance.

### 2. Correction

When the robot sees a marker, we do one of two things:

- If this marker has **never been seen before**, we initialize its position in the map.
- If this marker has **already been seen**, we use EKF correction to improve both the robot pose and the landmark position.

For a known landmark, the correction is:

$$\nu = z - h(\bar{\mu})$$

$$S = H\bar{\Sigma}H^T + Q$$

$$K = \bar{\Sigma}H^TS^{-1}$$

$$\mu = \bar{\mu} + K\nu$$

For the covariance update we will use the **Joseph form**:

$$\Sigma = (I - KH)\bar{\Sigma}(I - KH)^T + KQK^T$$

This form is a little more expensive than the simple update, but it usually behaves better numerically.

## Code

```python
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


class EKF_SLAM:
    """
    EKF-SLAM for omnidirectional base + planar landmarks.

    The robot state is [x, y, theta].
    Each landmark state is [m_x, m_y].

    Measurement model:
        z = [range, bearing]

    Predicted measurement:
        h(mu) = [sqrt(dx^2 + dy^2), atan2(dy, dx) - theta]
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
        """h(mu) = [range, atan2(dy, dx) - theta]."""
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
        H = dh/dmu for h = [range, bearing].

        H should be zero everywhere except:
        - the robot pose part [x, y, theta]
        - the two entries of the current landmark [m_x, m_y]
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
        """
        Initialize a landmark the first time we see it.

        The measurement is given as a camera-to-marker transform.
        We convert it to the world frame and store the marker x,y in the map.
        """
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
        # event based prediction
        self.prediction_step(u_body, dt)

        # event based correction
        self.correction_step(detections)
```

> - **Student TODO I:** Start by implementing `predicted_range_bearing`. Use the robot pose and the landmark position to compute the distance from the landmark and the predicted bearing. Return them as a NumPy array.
> - **Student TODO II:** Implement `measurement_residual`. First compute the real measurement `z` from the `measurement_body_from_T_cm`. Then compute the predicted measurement `h` from `predicted_range_bearing`. The residual is `z - h`. Do not forget to wrap the bearing residual with `wrap_to_pi`.
> - **Student TODO III:** Implement `analytic_measurement_jacobian`. This is the Jacobian of the range-bearing measurement model. Remember that only the robot pose and the current landmark should have non-zero entries in `H`.
> - **Student TODO IV:** Implement `initialize_landmark`. When a marker is seen for the first time, transform it into the world frame and write its x,y position into the correct part of the SLAM state vector. Also give this landmark a reasonable initial covariance. The measurement is given as a camera-to-marker transformation `T_cm_meas`. Use the `landmark_slice` function to put the new landmark in the correct position of the SLAM state. The function should return the updated `mu` and `Sigma`.
> - **Student TODO V:** Implement the EKF-SLAM prediction step. The motion model gives the predicted robot pose and the robot Jacobian. The landmarks stay fixed, but the full covariance matrix must still be propagated using the full SLAM Jacobian. The function should update `self._mu` and `self._Sigma`.
> - **Student TODO VI:** Implement the correction step for both cases: new landmarks and already-seen landmarks. For already-seen landmarks, calculate the innovation, measurement Jacobian, innovation covariance, Kalman gain, corrected state, and corrected covariance. For new landmarks initialize them and set the `self._marker_seen` for this landmark to true and use `continue` to move to the next measurement. The function should update `self._mu` and `self._Sigma`. Don't forget to use `wrap_slam_state_angles` on mu before updating it.

## Things to Check

After completing the code, check the following:

- The robot pose has three values: `x`, `y`, and `theta`.
- Each landmark uses exactly two values in the state vector.
- A newly detected landmark is initialized only once.
- Bearing differences are wrapped to $[-\pi, \pi]$.
- The covariance matrix stays symmetric after correction.
- Only the robot block receives process noise during prediction.