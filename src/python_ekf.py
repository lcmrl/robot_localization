"""Self-contained Extended Kalman Filter (EKF) matching robot_localization's ekf.cpp.

This module intentionally has *no* ROS/Eigen dependencies and does not import
anything from this repository. It re-implements the EKF core math and the
subset-update behavior used by robot_localization:

- 15-state layout: position, orientation (RPY), linear/angular velocities,
  linear accelerations
- EKF predict step with the same transfer function and analytic Jacobian
- EKF correct step with subset updates, Joseph-form covariance update, and
  Mahalanobis gating
- Optional control handling (velocity setpoints -> bounded accelerations)
- Optional dynamic process noise scaling based on velocity norm

State vector indices are compatible with robot_localization/filter_common.hpp.

Typical usage:

    ekf = EKF()
    z = Measurement.from_subset(
        time=0.0,
        update_indices=[StateMemberX, StateMemberY],
        values=[1.0, 2.0],
        covariance=[[0.1, 0.0], [0.0, 0.1]],
    )
    ekf.process_measurement(z)

    # Later...
    ekf.predict(delta_sec=0.01)

    # Fuse another measurement
    ekf.process_measurement(z2)

"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple


# --- Constants (match include/robot_localization/filter_common.hpp) ---

StateMemberX = 0
StateMemberY = 1
StateMemberZ = 2
StateMemberRoll = 3
StateMemberPitch = 4
StateMemberYaw = 5
StateMemberVx = 6
StateMemberVy = 7
StateMemberVz = 8
StateMemberVroll = 9
StateMemberVpitch = 10
StateMemberVyaw = 11
StateMemberAx = 12
StateMemberAy = 13
StateMemberAz = 14

ControlMemberVx = 0
ControlMemberVy = 1
ControlMemberVz = 2
ControlMemberVroll = 3
ControlMemberVpitch = 4
ControlMemberVyaw = 5

STATE_SIZE = 15
POSE_SIZE = 6
TWIST_SIZE = 6

POSITION_OFFSET = StateMemberX
ORIENTATION_OFFSET = StateMemberRoll
POSITION_V_OFFSET = StateMemberVx
ORIENTATION_V_OFFSET = StateMemberVroll
POSITION_A_OFFSET = StateMemberAx

PI = math.pi
TAU = 2.0 * math.pi


# --- Small linear algebra helpers (pure Python lists) ---

Vector = List[float]
Matrix = List[List[float]]


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _identity(n: int) -> Matrix:
    out = _zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def _copy_mat(a: Matrix) -> Matrix:
    return [row[:] for row in a]


def _transpose(a: Matrix) -> Matrix:
    if not a:
        return []
    rows, cols = len(a), len(a[0])
    out = _zeros(cols, rows)
    for r in range(rows):
        for c in range(cols):
            out[c][r] = a[r][c]
    return out


def _mat_add(a: Matrix, b: Matrix) -> Matrix:
    rows, cols = len(a), len(a[0])
    out = _zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            out[r][c] = a[r][c] + b[r][c]
    return out


def _mat_sub(a: Matrix, b: Matrix) -> Matrix:
    rows, cols = len(a), len(a[0])
    out = _zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            out[r][c] = a[r][c] - b[r][c]
    return out


def _mat_mul(a: Matrix, b: Matrix) -> Matrix:
    # (r x k) * (k x c) = (r x c)
    rows, k = len(a), len(a[0])
    k2, cols = len(b), len(b[0])
    if k != k2:
        raise ValueError(f"mat_mul shape mismatch: {rows}x{k} * {k2}x{cols}")
    out = _zeros(rows, cols)
    for r in range(rows):
        a_r = a[r]
        out_r = out[r]
        for kk in range(k):
            a_val = a_r[kk]
            if a_val == 0.0:
                continue
            b_kk = b[kk]
            for c in range(cols):
                out_r[c] += a_val * b_kk[c]
    return out


def _mat_vec_mul(a: Matrix, v: Vector) -> Vector:
    rows, cols = len(a), len(a[0])
    if len(v) != cols:
        raise ValueError(f"mat_vec_mul shape mismatch: {rows}x{cols} * {len(v)}")
    out = [0.0 for _ in range(rows)]
    for r in range(rows):
        s = 0.0
        row = a[r]
        for c in range(cols):
            s += row[c] * v[c]
        out[r] = s
    return out


def _vec_add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def _vec_dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_scale(a: Matrix, s: float) -> Matrix:
    rows, cols = len(a), len(a[0])
    out = _zeros(rows, cols)
    for r in range(rows):
        for c in range(cols):
            out[r][c] = a[r][c] * s
    return out


def _submatrix(a: Matrix, rows: Sequence[int], cols: Sequence[int]) -> Matrix:
    out = _zeros(len(rows), len(cols))
    for i, r in enumerate(rows):
        a_r = a[r]
        out_i = out[i]
        for j, c in enumerate(cols):
            out_i[j] = a_r[c]
    return out


def _gauss_jordan_inverse(a: Matrix) -> Matrix:
    """Invert a square matrix using Gauss-Jordan elimination.

    Raises ValueError if singular.
    """
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("matrix must be non-empty and square")

    # Augment with identity
    aug = [row[:] + ident_row[:] for row, ident_row in zip(a, _identity(n))]

    for col in range(n):
        # Pivot
        pivot_row = col
        pivot_val = abs(aug[pivot_row][col])
        for r in range(col + 1, n):
            v = abs(aug[r][col])
            if v > pivot_val:
                pivot_val = v
                pivot_row = r

        if pivot_val < 1e-15:
            raise ValueError("matrix appears singular")

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row
        pivot = aug[col][col]
        inv_pivot = 1.0 / pivot
        row = aug[col]
        for c in range(2 * n):
            row[c] *= inv_pivot

        # Eliminate other rows
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            row_r = aug[r]
            for c in range(2 * n):
                row_r[c] -= factor * row[c]

    inv = _zeros(n, n)
    for r in range(n):
        inv[r] = aug[r][n:]
    return inv


def normalize_angle(angle_rad: float) -> float:
    """Normalize angle to [-pi, pi). Equivalent to angles::normalize_angle."""
    # Fast path
    if -PI <= angle_rad < PI:
        return angle_rad
    # Python's fmod keeps sign
    a = math.fmod(angle_rad + PI, TAU)
    if a < 0.0:
        a += TAU
    return a - PI


@dataclass
class Measurement:
    """A measurement analogous to robot_localization::Measurement.

    All vectors/matrices are full STATE_SIZE and STATE_SIZE x STATE_SIZE,
    respectively, for parity with the C++ implementation.
    """

    time: float
    measurement: Vector
    covariance: Matrix
    update_vector: List[bool]
    mahalanobis_thresh: float = float("inf")

    @staticmethod
    def from_subset(
        *,
        time: float,
        update_indices: Sequence[int],
        values: Sequence[float],
        covariance: Matrix,
        mahalanobis_thresh: float = float("inf"),
    ) -> "Measurement":
        if len(update_indices) != len(values):
            raise ValueError("update_indices and values must be same length")
        if len(covariance) != len(values) or any(len(row) != len(values) for row in covariance):
            raise ValueError("covariance must be square with size len(values)")

        full_meas = [0.0] * STATE_SIZE
        full_cov = _zeros(STATE_SIZE, STATE_SIZE)
        update_vec = [False] * STATE_SIZE

        for i, idx in enumerate(update_indices):
            if idx < 0 or idx >= STATE_SIZE:
                raise ValueError(f"bad state index: {idx}")
            full_meas[idx] = float(values[i])
            update_vec[idx] = True

        for i, idx_i in enumerate(update_indices):
            for j, idx_j in enumerate(update_indices):
                full_cov[idx_i][idx_j] = float(covariance[i][j])

        return Measurement(
            time=float(time),
            measurement=full_meas,
            covariance=full_cov,
            update_vector=update_vec,
            mahalanobis_thresh=float(mahalanobis_thresh),
        )


class EKF:
    """EKF implementation closely mirroring robot_localization's Ekf + FilterBase."""

    def __init__(self) -> None:
        self.initialized: bool = False

        self.use_control: bool = False
        self.use_dynamic_process_noise_covariance: bool = False

        # Control state
        self.control_timeout_sec: float = 0.0
        self.latest_control_time: float = 0.0
        self.latest_control: Vector = [0.0] * TWIST_SIZE
        self.control_update_vector: List[bool] = [False] * TWIST_SIZE
        self.acceleration_limits: Vector = [0.0] * TWIST_SIZE
        self.acceleration_gains: Vector = [0.0] * TWIST_SIZE
        self.deceleration_limits: Vector = [0.0] * TWIST_SIZE
        self.deceleration_gains: Vector = [0.0] * TWIST_SIZE
        self.control_acceleration: Vector = [0.0] * TWIST_SIZE

        # Filter state
        self.state: Vector = [0.0] * STATE_SIZE
        self.predicted_state: Vector = [0.0] * STATE_SIZE
        self.last_measurement_time: float = 0.0

        self.identity: Matrix = _identity(STATE_SIZE)

        # Covariances
        self.estimate_error_covariance: Matrix = _identity(STATE_SIZE)
        # Matches filter_base.cpp: estimate_error_covariance_ *= 1e-9
        self.estimate_error_covariance = _mat_scale(self.estimate_error_covariance, 1e-9)

        self.process_noise_covariance: Matrix = _zeros(STATE_SIZE, STATE_SIZE)
        self._set_default_process_noise()
        self.dynamic_process_noise_covariance: Matrix = _copy_mat(self.process_noise_covariance)

        # Transfer function and Jacobian
        self.transfer_function: Matrix = _identity(STATE_SIZE)
        self.transfer_function_jacobian: Matrix = _zeros(STATE_SIZE, STATE_SIZE)

    def _set_default_process_noise(self) -> None:
        q = self.process_noise_covariance
        q[StateMemberX][StateMemberX] = 0.05
        q[StateMemberY][StateMemberY] = 0.05
        q[StateMemberZ][StateMemberZ] = 0.06
        q[StateMemberRoll][StateMemberRoll] = 0.03
        q[StateMemberPitch][StateMemberPitch] = 0.03
        q[StateMemberYaw][StateMemberYaw] = 0.06
        q[StateMemberVx][StateMemberVx] = 0.025
        q[StateMemberVy][StateMemberVy] = 0.025
        q[StateMemberVz][StateMemberVz] = 0.04
        q[StateMemberVroll][StateMemberVroll] = 0.01
        q[StateMemberVpitch][StateMemberVpitch] = 0.01
        q[StateMemberVyaw][StateMemberVyaw] = 0.02
        q[StateMemberAx][StateMemberAx] = 0.01
        q[StateMemberAy][StateMemberAy] = 0.01
        q[StateMemberAz][StateMemberAz] = 0.015

    def reset(self) -> None:
        self.__init__()

    def set_use_dynamic_process_noise_covariance(self, enabled: bool) -> None:
        self.use_dynamic_process_noise_covariance = bool(enabled)

    def set_process_noise_covariance(self, q: Matrix) -> None:
        if len(q) != STATE_SIZE or any(len(row) != STATE_SIZE for row in q):
            raise ValueError("process noise covariance must be 15x15")
        self.process_noise_covariance = _copy_mat(q)
        self.dynamic_process_noise_covariance = _copy_mat(q)

    def set_control(
        self,
        control: Sequence[float],
        control_time: float,
    ) -> None:
        if len(control) != TWIST_SIZE:
            raise ValueError("control must be length 6 (vx, vy, vz, vroll, vpitch, vyaw)")
        self.latest_control = [float(x) for x in control]
        self.latest_control_time = float(control_time)

    def set_control_params(
        self,
        *,
        update_vector: Sequence[bool],
        control_timeout_sec: float,
        acceleration_limits: Sequence[float],
        acceleration_gains: Sequence[float],
        deceleration_limits: Sequence[float],
        deceleration_gains: Sequence[float],
    ) -> None:
        if len(update_vector) != TWIST_SIZE:
            raise ValueError("control update vector must be length 6")
        for name, arr in [
            ("acceleration_limits", acceleration_limits),
            ("acceleration_gains", acceleration_gains),
            ("deceleration_limits", deceleration_limits),
            ("deceleration_gains", deceleration_gains),
        ]:
            if len(arr) != TWIST_SIZE:
                raise ValueError(f"{name} must be length 6")

        self.use_control = True
        self.control_update_vector = [bool(x) for x in update_vector]
        self.control_timeout_sec = float(control_timeout_sec)
        self.acceleration_limits = [float(x) for x in acceleration_limits]
        self.acceleration_gains = [float(x) for x in acceleration_gains]
        self.deceleration_limits = [float(x) for x in deceleration_limits]
        self.deceleration_gains = [float(x) for x in deceleration_gains]

    def _compute_dynamic_process_noise_covariance(self, state: Vector) -> None:
        # Matches FilterBase::computeDynamicProcessNoiseCovariance
        # Scale pose noise (x..yaw) by norm of twist (vx..vyaw)
        twist = state[POSITION_V_OFFSET: POSITION_V_OFFSET + TWIST_SIZE]
        vnorm = math.sqrt(sum(v * v for v in twist))
        # velocity_matrix is vnorm * I (6x6)
        # dynamic_pose_Q = velocity_matrix * pose_Q * velocity_matrix^T
        # Since velocity_matrix is diagonal with vnorm, this is vnorm^2 * pose_Q.
        scale = vnorm * vnorm
        # Update only the 6x6 pose block (0..5, 0..5)
        for r in range(POSE_SIZE):
            for c in range(POSE_SIZE):
                self.dynamic_process_noise_covariance[r][c] = self.process_noise_covariance[r][c] * scale

    @staticmethod
    def _compute_control_acceleration(
        state: float,
        control: float,
        acceleration_limit: float,
        acceleration_gain: float,
        deceleration_limit: float,
        deceleration_gain: float,
    ) -> float:
        # Matches FilterBase::computeControlAcceleration
        error = control - state
        same_sign = abs(error) <= abs(control) + 0.01
        set_point = control if same_sign else 0.0
        decelerating = abs(set_point) < abs(state)

        limit = acceleration_limit
        gain = acceleration_gain
        if decelerating:
            limit = deceleration_limit
            gain = deceleration_gain

        accel = gain * error
        if accel > limit:
            accel = limit
        elif accel < -limit:
            accel = -limit
        return accel

    def _prepare_control(self, reference_time: float) -> None:
        self.control_acceleration = [0.0] * TWIST_SIZE
        if not self.use_control:
            return

        timed_out = (reference_time - self.latest_control_time) >= self.control_timeout_sec

        for control_ind in range(TWIST_SIZE):
            if not self.control_update_vector[control_ind]:
                continue
            state_val = self.state[control_ind + POSITION_V_OFFSET]
            control_val = 0.0 if timed_out else self.latest_control[control_ind]

            self.control_acceleration[control_ind] = self._compute_control_acceleration(
                state=state_val,
                control=control_val,
                acceleration_limit=self.acceleration_limits[control_ind],
                acceleration_gain=self.acceleration_gains[control_ind],
                deceleration_limit=self.deceleration_limits[control_ind],
                deceleration_gain=self.deceleration_gains[control_ind],
            )

    def _wrap_state_angles(self) -> None:
        self.state[StateMemberRoll] = normalize_angle(self.state[StateMemberRoll])
        self.state[StateMemberPitch] = normalize_angle(self.state[StateMemberPitch])
        self.state[StateMemberYaw] = normalize_angle(self.state[StateMemberYaw])

    @staticmethod
    def _check_mahalanobis_threshold(
        innovation: Vector,
        innovation_cov_inv: Matrix,
        n_sigmas: float,
    ) -> bool:
        if not math.isfinite(n_sigmas):
            return True
        tmp = _mat_vec_mul(innovation_cov_inv, innovation)
        squared = _vec_dot(innovation, tmp)
        threshold = n_sigmas * n_sigmas
        return squared < threshold

    def predict(self, *, reference_time: Optional[float] = None, delta_sec: float) -> None:
        """Predict step.

        Args:
            reference_time: Current time seconds (only used for control timeout).
                If None, uses last_measurement_time + delta_sec.
            delta_sec: Time step in seconds.
        """
        if delta_sec < 0.0:
            raise ValueError("delta_sec must be non-negative")
        if reference_time is None:
            reference_time = self.last_measurement_time + float(delta_sec)
        delta_sec = float(delta_sec)

        roll = self.state[StateMemberRoll]
        pitch = self.state[StateMemberPitch]
        yaw = self.state[StateMemberYaw]
        x_vel = self.state[StateMemberVx]
        y_vel = self.state[StateMemberVy]
        z_vel = self.state[StateMemberVz]
        pitch_vel = self.state[StateMemberVpitch]
        yaw_vel = self.state[StateMemberVyaw]
        x_acc = self.state[StateMemberAx]
        y_acc = self.state[StateMemberAy]
        z_acc = self.state[StateMemberAz]

        sp = math.sin(pitch)
        cp = math.cos(pitch)
        if abs(cp) < 1e-12:
            # Avoid division blow-ups; ekf.cpp relies on cp in denominator.
            cp = 1e-12 if cp >= 0.0 else -1e-12
        cpi = 1.0 / cp
        tp = sp * cpi

        sr = math.sin(roll)
        cr = math.cos(roll)

        sy = math.sin(yaw)
        cy = math.cos(yaw)

        self._prepare_control(reference_time)

        # Transfer function (set identity then fill non-invariant terms)
        F = _identity(STATE_SIZE)

        F[StateMemberX][StateMemberVx] = cy * cp * delta_sec
        F[StateMemberX][StateMemberVy] = (cy * sp * sr - sy * cr) * delta_sec
        F[StateMemberX][StateMemberVz] = (cy * sp * cr + sy * sr) * delta_sec
        F[StateMemberX][StateMemberAx] = 0.5 * F[StateMemberX][StateMemberVx] * delta_sec
        F[StateMemberX][StateMemberAy] = 0.5 * F[StateMemberX][StateMemberVy] * delta_sec
        F[StateMemberX][StateMemberAz] = 0.5 * F[StateMemberX][StateMemberVz] * delta_sec

        F[StateMemberY][StateMemberVx] = sy * cp * delta_sec
        F[StateMemberY][StateMemberVy] = (sy * sp * sr + cy * cr) * delta_sec
        F[StateMemberY][StateMemberVz] = (sy * sp * cr - cy * sr) * delta_sec
        F[StateMemberY][StateMemberAx] = 0.5 * F[StateMemberY][StateMemberVx] * delta_sec
        F[StateMemberY][StateMemberAy] = 0.5 * F[StateMemberY][StateMemberVy] * delta_sec
        F[StateMemberY][StateMemberAz] = 0.5 * F[StateMemberY][StateMemberVz] * delta_sec

        F[StateMemberZ][StateMemberVx] = -sp * delta_sec
        F[StateMemberZ][StateMemberVy] = cp * sr * delta_sec
        F[StateMemberZ][StateMemberVz] = cp * cr * delta_sec
        F[StateMemberZ][StateMemberAx] = 0.5 * F[StateMemberZ][StateMemberVx] * delta_sec
        F[StateMemberZ][StateMemberAy] = 0.5 * F[StateMemberZ][StateMemberVy] * delta_sec
        F[StateMemberZ][StateMemberAz] = 0.5 * F[StateMemberZ][StateMemberVz] * delta_sec

        F[StateMemberRoll][StateMemberVroll] = delta_sec
        F[StateMemberRoll][StateMemberVpitch] = sr * tp * delta_sec
        F[StateMemberRoll][StateMemberVyaw] = cr * tp * delta_sec

        F[StateMemberPitch][StateMemberVpitch] = cr * delta_sec
        F[StateMemberPitch][StateMemberVyaw] = -sr * delta_sec

        F[StateMemberYaw][StateMemberVpitch] = sr * cpi * delta_sec
        F[StateMemberYaw][StateMemberVyaw] = cr * cpi * delta_sec

        F[StateMemberVx][StateMemberAx] = delta_sec
        F[StateMemberVy][StateMemberAy] = delta_sec
        F[StateMemberVz][StateMemberAz] = delta_sec

        # Jacobian: derived from transfer function; matches ekf.cpp
        one_half_at_squared = 0.5 * delta_sec * delta_sec

        y_coeff = cy * sp * cr + sy * sr
        z_coeff = -cy * sp * sr + sy * cr
        dFx_dR = (y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared
        dFR_dR = 1.0 + (cr * tp * pitch_vel - sr * tp * yaw_vel) * delta_sec

        x_coeff = -cy * sp
        y_coeff = cy * cp * sr
        z_coeff = cy * cp * cr
        dFx_dP = (x_coeff * x_vel + y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (x_coeff * x_acc + y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared
        dFR_dP = (cpi * cpi * sr * pitch_vel + cpi * cpi * cr * yaw_vel) * delta_sec

        x_coeff = -sy * cp
        y_coeff = -sy * sp * sr - cy * cr
        z_coeff = -sy * sp * cr + cy * sr
        dFx_dY = (x_coeff * x_vel + y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (x_coeff * x_acc + y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared

        y_coeff = sy * sp * cr - cy * sr
        z_coeff = -sy * sp * sr - cy * cr
        dFy_dR = (y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared
        dFP_dR = (-sr * pitch_vel - cr * yaw_vel) * delta_sec

        x_coeff = -sy * sp
        y_coeff = sy * cp * sr
        z_coeff = sy * cp * cr
        dFy_dP = (x_coeff * x_vel + y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (x_coeff * x_acc + y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared

        x_coeff = cy * cp
        y_coeff = cy * sp * sr - sy * cr
        z_coeff = cy * sp * cr + sy * sr
        dFy_dY = (x_coeff * x_vel + y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (x_coeff * x_acc + y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared

        y_coeff = cp * cr
        z_coeff = -cp * sr
        dFz_dR = (y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared
        dFY_dR = (cr * cpi * pitch_vel - sr * cpi * yaw_vel) * delta_sec

        x_coeff = -cp
        y_coeff = -sp * sr
        z_coeff = -sp * cr
        dFz_dP = (x_coeff * x_vel + y_coeff * y_vel + z_coeff * z_vel) * delta_sec + (x_coeff * x_acc + y_coeff * y_acc + z_coeff * z_acc) * one_half_at_squared
        dFY_dP = (sr * tp * cpi * pitch_vel + cr * tp * cpi * yaw_vel) * delta_sec

        J = _copy_mat(F)
        J[StateMemberX][StateMemberRoll] = dFx_dR
        J[StateMemberX][StateMemberPitch] = dFx_dP
        J[StateMemberX][StateMemberYaw] = dFx_dY
        J[StateMemberY][StateMemberRoll] = dFy_dR
        J[StateMemberY][StateMemberPitch] = dFy_dP
        J[StateMemberY][StateMemberYaw] = dFy_dY
        J[StateMemberZ][StateMemberRoll] = dFz_dR
        J[StateMemberZ][StateMemberPitch] = dFz_dP
        J[StateMemberRoll][StateMemberRoll] = dFR_dR
        J[StateMemberRoll][StateMemberPitch] = dFR_dP
        J[StateMemberPitch][StateMemberRoll] = dFP_dR
        J[StateMemberYaw][StateMemberRoll] = dFY_dR
        J[StateMemberYaw][StateMemberPitch] = dFY_dP

        self.transfer_function = F
        self.transfer_function_jacobian = J

        # Choose process noise covariance
        process_noise = self.process_noise_covariance
        if self.use_dynamic_process_noise_covariance:
            self._compute_dynamic_process_noise_covariance(self.state)
            process_noise = self.dynamic_process_noise_covariance

        # (1) Apply control terms (accelerations)
        self.state[StateMemberVroll] += self.control_acceleration[ControlMemberVroll] * delta_sec
        self.state[StateMemberVpitch] += self.control_acceleration[ControlMemberVpitch] * delta_sec
        self.state[StateMemberVyaw] += self.control_acceleration[ControlMemberVyaw] * delta_sec

        self.state[StateMemberAx] = self.control_acceleration[ControlMemberVx] if self.control_update_vector[ControlMemberVx] else self.state[StateMemberAx]
        self.state[StateMemberAy] = self.control_acceleration[ControlMemberVy] if self.control_update_vector[ControlMemberVy] else self.state[StateMemberAy]
        self.state[StateMemberAz] = self.control_acceleration[ControlMemberVz] if self.control_update_vector[ControlMemberVz] else self.state[StateMemberAz]

        # (2) Project the state forward: x = F x
        self.state = _mat_vec_mul(F, self.state)
        self._wrap_state_angles()

        # (3) Project the error forward: P = J P J' + dt * Q
        JP = _mat_mul(J, self.estimate_error_covariance)
        Jt = _transpose(J)
        self.estimate_error_covariance = _mat_mul(JP, Jt)
        self.estimate_error_covariance = _mat_add(self.estimate_error_covariance, _mat_scale(process_noise, delta_sec))

    def correct(self, measurement: Measurement) -> None:
        # Determine update indices: update_vector true and measurement finite
        update_indices: List[int] = []
        for i, do_update in enumerate(measurement.update_vector):
            if not do_update:
                continue
            v = measurement.measurement[i]
            if math.isnan(v) or math.isinf(v):
                continue
            update_indices.append(i)

        if not update_indices:
            return

        m = len(update_indices)

        # Build measurement and state subsets
        z = [0.0] * m
        x_subset = [0.0] * m
        R = _zeros(m, m)
        H = _zeros(m, STATE_SIZE)

        for i, idx in enumerate(update_indices):
            z[i] = measurement.measurement[idx]
            x_subset[i] = self.state[idx]

            # Measurement covariance subset + sanity checks
            for j, jdx in enumerate(update_indices):
                R[i][j] = measurement.covariance[idx][jdx]

            if R[i][i] < 0.0:
                R[i][i] = abs(R[i][i])
            if R[i][i] < 1e-9:
                R[i][i] = 1e-9

            H[i][idx] = 1.0

        # (1) K = P H' (H P H' + R)^{-1}
        Ht = _transpose(H)
        P = self.estimate_error_covariance
        PHt = _mat_mul(P, Ht)  # (15 x m)
        S = _mat_add(_mat_mul(H, PHt), R)  # (m x m)

        try:
            S_inv = _gauss_jordan_inverse(S)
        except ValueError:
            # Regularize very slightly and try again
            S_reg = _copy_mat(S)
            for i in range(m):
                S_reg[i][i] += 1e-12
            S_inv = _gauss_jordan_inverse(S_reg)

        K = _mat_mul(PHt, S_inv)  # (15 x m)

        innovation = _vec_sub(z, x_subset)

        # Wrap angles in innovation
        for i, idx in enumerate(update_indices):
            if idx in (StateMemberRoll, StateMemberPitch, StateMemberYaw):
                innovation[i] = normalize_angle(innovation[i])

        # (2) Mahalanobis gating
        if not self._check_mahalanobis_threshold(innovation, S_inv, measurement.mahalanobis_thresh):
            return

        # (3) state = state + K * innovation
        dx = _mat_vec_mul(K, innovation)
        self.state = _vec_add(self.state, dx)

        # (4) Joseph-form covariance update
        KH = _mat_mul(K, H)  # (15 x 15)
        I_minus_KH = _mat_sub(self.identity, KH)
        tmp = _mat_mul(_mat_mul(I_minus_KH, P), _transpose(I_minus_KH))
        KRKt = _mat_mul(_mat_mul(K, R), _transpose(K))
        self.estimate_error_covariance = _mat_add(tmp, KRKt)

        self._wrap_state_angles()

    def process_measurement(self, measurement: Measurement) -> None:
        """Mimics FilterBase::processMeasurement for a single measurement."""
        if self.initialized:
            delta = measurement.time - self.last_measurement_time
            if delta > 0.0:
                self.predict(reference_time=measurement.time, delta_sec=delta)
                self.predicted_state = self.state[:]
            self.correct(measurement)
        else:
            # First measurement initializes filter using only updated elements
            for i in range(min(len(measurement.update_vector), STATE_SIZE)):
                if measurement.update_vector[i]:
                    self.state[i] = measurement.measurement[i]

            for i in range(min(len(measurement.update_vector), STATE_SIZE)):
                for j in range(min(len(measurement.update_vector), STATE_SIZE)):
                    if measurement.update_vector[i] and measurement.update_vector[j]:
                        self.estimate_error_covariance[i][j] = measurement.covariance[i][j]

            self.initialized = True

        if measurement.time >= self.last_measurement_time:
            self.last_measurement_time = measurement.time


def _demo() -> None:
    ekf = EKF()

    # Initialize with position measurement at t=0
    ekf.process_measurement(
        Measurement.from_subset(
            time=0.0,
            update_indices=[StateMemberX, StateMemberY, StateMemberYaw],
            values=[0.0, 0.0, 0.0],
            covariance=[
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
        )
    )

    # Give it a forward velocity (as state), then predict
    ekf.state[StateMemberVx] = 1.0

    for k in range(1, 6):
        t = 0.1 * k
        ekf.predict(reference_time=t, delta_sec=0.1)

    # Correct with a noisy position measurement
    ekf.process_measurement(
        Measurement.from_subset(
            time=0.6,
            update_indices=[StateMemberX, StateMemberY],
            values=[0.62, -0.02],
            covariance=[[0.04, 0.0], [0.0, 0.04]],
            mahalanobis_thresh=10.0,
        )
    )

    print("State after demo (x, y, yaw, vx):", ekf.state[StateMemberX], ekf.state[StateMemberY], ekf.state[StateMemberYaw], ekf.state[StateMemberVx])


if __name__ == "__main__":
    _demo()
