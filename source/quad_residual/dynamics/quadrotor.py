"""Batched quadrotor rigid-body model.

Used by ``sim_lite`` for CPU development. Inside Isaac Lab the rigid-body
integration is PhysX's job -- but the *actuator* model here (first-order motor
lag plus a discrete actuation delay buffer) is still applied before the wrench
is handed to PhysX, because PhysX models neither.

Frames: world ENU, body FLU. Quaternions are (w, x, y, z).
Rotor layout (X configuration, viewed from above, body x forward) -- the same
order and spin directions as PX4's "Quadrotor X":

    0: front-right, CCW      1: rear-left,  CCW
    2: front-left,  CW       3: rear-right, CW

Yaw sign: a rotor's drag reaction on the body is opposite to its spin, so the
CCW rotors (0, 1) produce negative tau_z (body z up) and the CW rotors (2, 3)
positive. When mapping onto a real airframe or a USD asset, match both the
rotor order *and* the spin directions to this table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend import quat_mul, quat_normalize, quat_rotate, xp
from .actuator import actuate


@dataclass
class QuadParams:
    """Nominal parameters. Domain randomization perturbs a copy of these."""

    mass: float = 0.85                 # kg
    inertia: tuple = (5.0e-3, 5.0e-3, 9.0e-3)   # kg m^2, diagonal
    arm_length: float = 0.15           # m, rotor to centre
    max_thrust_per_rotor: float = 5.0  # N at unit command
    torque_coeff: float = 0.016        # drag-torque / thrust ratio, m
    motor_tau: float = 0.035           # s, first-order motor time constant
    actuation_delay_steps: int = 1     # nominal physics steps of pure transport delay (0 allowed)
    drag_coeff: float = 0.10           # N per (m/s), linear body drag
    gravity: float = 9.81
    dt: float = 0.004                  # s, 250 Hz physics


# Mixing: [thrust, tau_x, tau_y, tau_z] = MIX @ rotor_thrusts
def mixing_matrix(arm_length: float, torque_coeff: float) -> np.ndarray:
    L = arm_length * (2.0 ** -0.5)      # X-layout moment arm on each axis
    k = torque_coeff
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],      # total thrust
            [-L, L, L, -L],            # roll  (tau_x)
            [-L, L, -L, L],            # pitch (tau_y)
            [-k, -k, k, k],            # yaw   (tau_z): reaction opposes spin, CW rotors positive
        ],
        dtype=np.float64,
    )


@dataclass
class QuadState:
    pos: np.ndarray        # (N, 3)
    vel: np.ndarray        # (N, 3) world
    quat: np.ndarray       # (N, 4) w-first
    omega: np.ndarray      # (N, 3) body
    motor: np.ndarray      # (N, 4) actual normalized rotor command in [0, 1]
    delay_buf: np.ndarray  # (N, D_max, 4) last D_max issued commands, oldest first

    def copy(self) -> "QuadState":
        return QuadState(*(a.copy() for a in (
            self.pos, self.vel, self.quat, self.omega, self.motor, self.delay_buf)))


class QuadrotorModel:
    """Semi-implicit Euler integration of the batched rigid body."""

    def __init__(self, params: QuadParams, num_envs: int, max_delay_steps: int | None = None):
        """``max_delay_steps`` sizes the delay buffer; domain randomization may
        then set any per-env delay in ``[0, max_delay_steps]``."""
        self.p = params
        self.n = num_envs
        base = int(params.actuation_delay_steps)
        self.max_delay = base if max_delay_steps is None else int(max_delay_steps)
        if self.max_delay < base:
            raise ValueError(f"max_delay_steps={self.max_delay} < nominal delay {base}")
        self.delay_steps = np.full(num_envs, base, dtype=np.int64)
        self._mix = mixing_matrix(params.arm_length, params.torque_coeff)
        # Per-environment parameters, broadcastable; domain randomization writes here.
        self.mass = np.full((num_envs, 1), params.mass, dtype=np.float32)
        self.inertia = np.tile(
            np.asarray(params.inertia, dtype=np.float32), (num_envs, 1))
        self.max_thrust = np.full(
            (num_envs, 1), params.max_thrust_per_rotor, dtype=np.float32)
        self.motor_tau = np.full((num_envs, 1), params.motor_tau, dtype=np.float32)
        self.drag = np.full((num_envs, 1), params.drag_coeff, dtype=np.float32)
        self.wind = np.zeros((num_envs, 3), dtype=np.float32)

    # ------------------------------------------------------------------ init
    def zero_state(self) -> QuadState:
        n = self.n
        d = self.max_delay
        quat = np.zeros((n, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        hover = self.hover_command()
        return QuadState(
            pos=np.zeros((n, 3), dtype=np.float32),
            vel=np.zeros((n, 3), dtype=np.float32),
            quat=quat,
            omega=np.zeros((n, 3), dtype=np.float32),
            motor=np.tile(hover, (1, 4)).astype(np.float32),
            delay_buf=np.tile(hover[:, None, :], (1, d, 4)).astype(np.float32),
        )

    def hover_command(self) -> np.ndarray:
        """Per-rotor normalized command that exactly cancels gravity."""
        return (self.mass * self.p.gravity) / (4.0 * self.max_thrust)

    # ---------------------------------------------------------------- step
    def step(self, state: QuadState, cmd: np.ndarray) -> QuadState:
        """Advance one physics step. ``cmd`` is the commanded rotor vector in [0, 1]."""
        B = xp(state.pos)
        p = self.p
        dt = p.dt

        # --- per-env actuation delay + first-order motor lag ------------------
        alpha = np.clip(dt / np.maximum(self.motor_tau, 1e-6), 0.0, 1.0)
        motor, buf = actuate(state.delay_buf, state.motor,
                             np.asarray(cmd, np.float32), self.delay_steps, alpha)

        # --- wrench ------------------------------------------------------------
        rotor_thrust = motor * self.max_thrust                   # (N, 4) newtons
        wrench = rotor_thrust @ self._mix.T.astype(np.float32)   # (N, 4)
        thrust = wrench[:, 0:1]
        torque = wrench[:, 1:4]

        # --- translational -----------------------------------------------------
        thrust_body = np.concatenate(
            [np.zeros_like(thrust), np.zeros_like(thrust), thrust], axis=-1)
        f_world = quat_rotate(state.quat, thrust_body)
        rel_v = state.vel - self.wind
        f_world = f_world - self.drag * rel_v
        acc = f_world / self.mass
        acc[:, 2] -= p.gravity

        vel = state.vel + dt * acc
        pos = state.pos + dt * vel          # semi-implicit: new velocity

        # --- rotational --------------------------------------------------------
        I = self.inertia
        gyro = np.cross(state.omega, I * state.omega)
        omega_dot = (torque - gyro) / I
        omega = state.omega + dt * omega_dot

        wq = np.concatenate([np.zeros((self.n, 1), dtype=state.quat.dtype), omega],
                            axis=-1)
        quat = state.quat + dt * 0.5 * quat_mul(state.quat, wq)
        quat = quat_normalize(quat)

        del B
        return QuadState(pos=pos, vel=vel, quat=quat, omega=omega,
                         motor=motor, delay_buf=buf)
