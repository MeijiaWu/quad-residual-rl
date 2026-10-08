"""Cascaded geometric controller -- the classical baseline.

Position loop (PD + feed-forward) produces a desired acceleration; that fixes
the desired thrust magnitude and the desired body-z axis. The attitude loop is
the geometric controller of Lee, Leok & McClamroch (2010), reduced to its
proportional-derivative form.

Two reasons this class exists in an RL project:

1. It is the baseline every learned policy is measured against.
2. It is the *base* of the residual policy -- the policy outputs a correction
   added to this controller's rotor command rather than the command itself,
   which is one of the three approaches the project compares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import quat_from_zaxis_yaw, quat_rotate, quat_to_rotmat_cols, vee, xp
from ..dynamics.quadrotor import mixing_matrix


@dataclass
class GeometricGains:
    """Cascade gains.

    Because the attitude torque is scaled by the inertia, the closed-loop
    angular dynamics reduce to ``theta_ddot = -kr*e_R - kw*e_omega``, so the
    gains map directly onto a second-order response:
    ``kr = wn**2`` and ``kw = 2*zeta*wn``.

    The defaults are the "fast" preset: attitude wn = 20 rad/s (yaw 8),
    position wn = 2.5 rad/s. The ~8x separation is what makes the cascade
    stable -- halving it makes the inner loop too slow to serve the outer one
    and the whole thing limit-cycles. Use :meth:`preset` / :meth:`from_bandwidth`
    rather than editing numbers by hand.
    """

    kp_pos: tuple = (6.25, 6.25, 9.0)     # wn 2.5 rad/s (3.0 on z)
    kd_pos: tuple = (4.5, 4.5, 5.4)       # zeta 0.9
    kr_att: tuple = (400.0, 400.0, 64.0)  # wn 20 rad/s (yaw 8)
    kw_att: tuple = (32.0, 32.0, 12.8)    # zeta 0.8
    tilt_limit: float = 0.7      # rad, cap on commanded tilt
    max_acc_xy: float = 8.0      # m/s^2
    ki_pos: tuple = (0.0, 0.0, 0.0)  # position integral gain; 0 = PD only (original)
    max_i_acc: float = 6.0       # m/s^2 per axis, anti-windup clamp; must cover the worst hover-thrust
                                 # mismatch in the randomization range (heavy + weak motors ~ 0.45 g)

    @classmethod
    def from_bandwidth(cls, att_wn: float, pos_wn: float | None = None,
                       integral: bool = True, att_zeta: float = 0.8,
                       pos_zeta: float = 0.9) -> "GeometricGains":
        """Gains from natural frequencies. Yaw runs at 0.4x the roll/pitch
        bandwidth, altitude at 1.2x the horizontal one; ``pos_wn`` defaults to
        ``att_wn / 8`` (the cascade separation).

        With ``integral`` the position loop becomes PID with ``ki = 0.25 wn^3``.
        For the double-integrator position plant the characteristic polynomial
        is ``s^3 + kd s^2 + kp s + ki``; Routh needs ``kd*kp > ki``, i.e.
        ``0.25 < 2*zeta`` -- satisfied with a wide margin, so the integrator
        removes steady-state offsets (unknown mass, thrust, wind, drag) without
        reshaping the transient much.
        """
        pw = att_wn / 8.0 if pos_wn is None else pos_wn
        aw = (att_wn, att_wn, 0.4 * att_wn)
        pv = (pw, pw, 1.2 * pw)
        return cls(
            kp_pos=tuple(w * w for w in pv),
            kd_pos=tuple(2 * pos_zeta * w for w in pv),
            kr_att=tuple(w * w for w in aw),
            kw_att=tuple(2 * att_zeta * w for w in aw),
            ki_pos=tuple(0.25 * w ** 3 for w in pv) if integral else (0.0, 0.0, 0.0),
        )

    @classmethod
    def preset(cls, name: str, integral: bool = True) -> "GeometricGains":
        if name not in BASELINE_PRESETS:
            raise KeyError(f"unknown baseline_gains {name!r}; have {sorted(BASELINE_PRESETS)}")
        return cls.from_bandwidth(BASELINE_PRESETS[name], integral=integral)


# Attitude natural frequency (rad/s) of each named baseline; position = /8.
#   fast   -- the original tuning. Good at nominal actuator dynamics, but it
#             limit-cycles or diverges over much of the randomized delay x
#             motor-lag range (see scripts/select_baseline_gains.py).
#   robust -- chosen by scripts/select_baseline_gains.py with a rule fixed in
#             advance: the largest candidate bandwidth that is stable at every
#             corner of the randomization range plus a margin. It is the base
#             of the residual policy and the safety fallback, so it must never
#             be the thing that crashes.
BASELINE_PRESETS = {"fast": 20.0, "robust": 8.0}


class GeometricController:
    def __init__(self, params, gains: GeometricGains | None = None, num_envs: int = 1,
                 dt: float | None = None):
        """``dt`` is the control period; it is only needed when the gains have
        an integral term (``ki_pos`` non-zero). The integrator state is per env
        and must be cleared with :meth:`reset` when an env resets."""
        self.p = params
        self.g = gains or GeometricGains()
        self.n = num_envs
        self.dt = dt
        mix = mixing_matrix(params.arm_length, params.torque_coeff)
        self._mix_inv = np.linalg.inv(mix).astype(np.float32)
        self._ki = np.asarray(self.g.ki_pos, np.float32)
        if self._ki.any() and not dt:
            raise ValueError("integral gains need the control period dt")
        self.i_pos = np.zeros((num_envs, 3), np.float32)

    def reset(self, env_ids=None, hover_ratio=None):
        """Clear the integrator for ``env_ids``.

        ``hover_ratio`` (per reset env) warm-starts the altitude integrator as a
        converged hover-thrust estimate: the ratio of the thrust command the
        aircraft really needs to hover to the one the nominal model predicts.
        Use it when the episode starts from a steady hover (motors already at
        the true hover command): an aircraft that has been hovering has a
        settled integrator, and zeroing it would drop a heavy airframe on the
        first step -- an artefact of the reset, not of the controller.

        Works on NumPy or torch: ``hover_ratio`` decides the backend the first
        time; the integrator then lives on that backend (and device).
        """
        ids = slice(None) if env_ids is None else env_ids
        if hover_ratio is not None and not isinstance(hover_ratio, np.ndarray) \
                and not isinstance(hover_ratio, (list, tuple)):
            self._ensure_backend(hover_ratio)      # torch tensor
        self.i_pos[ids] = 0.0
        if hover_ratio is not None and self._ki[2] > 0:
            B = xp(self.i_pos)
            r = hover_ratio if not isinstance(hover_ratio, (list, tuple)) else np.asarray(hover_ratio)
            r = B.const(r, self.i_pos).reshape(-1) if B.name == "numpy" else r.reshape(-1).to(self.i_pos)
            need = B.clip(self.p.gravity * (r - 1.0), -self.g.max_i_acc, self.g.max_i_acc)
            self.i_pos[ids, 2] = need / float(self._ki[2])

    def _ensure_backend(self, sample):
        """Move the integrator to ``sample``'s backend/device on first use."""
        if type(self.i_pos) is type(sample) and \
                getattr(self.i_pos, "device", None) == getattr(sample, "device", None):
            return
        B = xp(sample)
        self.i_pos = B.zeros((self.n, 3), like=sample)

    def __call__(self, state, ref, mass, inertia, max_thrust):
        """Return the normalized rotor command in [0, 1].

        ``ref`` carries ``pos``, ``vel``, ``acc`` (feed-forward) and ``yaw``.
        One implementation for NumPy (sim_lite) and torch (Isaac Lab): every
        operation goes through the backend shim, so the two cannot drift.
        """
        B = xp(state.pos)
        g = self.g
        self._ensure_backend(state.pos)
        like = state.pos

        kp = B.const(g.kp_pos, like)
        kd = B.const(g.kd_pos, like)
        ki = B.const(self._ki, like)

        # ---- position loop -> desired acceleration ---------------------------
        e_p = ref["pos"] - state.pos
        e_v = ref["vel"] - state.vel
        acc_des = kp * e_p + kd * e_v + ref["acc"]
        if self._ki.any():
            # Integrate, then clamp the *contribution* per axis (anti-windup).
            lim = B.const(g.max_i_acc / np.maximum(self._ki, 1e-9), like)
            self.i_pos = B.clip(self.i_pos + e_p * self.dt, -lim, lim)
            acc_des = acc_des + ki * self.i_pos
        acc_xy = B.clip(acc_des[:, :2], -g.max_acc_xy, g.max_acc_xy)
        acc_z = acc_des[:, 2:3] + self.p.gravity

        # ---- desired thrust and body z --------------------------------------
        # Limit tilt by flooring the vertical component relative to the lateral.
        lat = B.norm(acc_xy)
        min_vert = lat / max(float(np.tan(g.tilt_limit)), 1e-3)
        acc_z = B.maximum(acc_z, min_vert)
        acc_des = B.cat([acc_xy, acc_z], axis=-1)

        b3_des = acc_des / B.clip(B.norm(acc_des), 1e-6, None)
        cols = quat_to_rotmat_cols(state.quat)
        b3 = cols[2]
        thrust = mass * (acc_des * b3).sum(-1)[..., None]
        thrust = B.minimum(B.clip(thrust, 0.0, None), 4.0 * max_thrust)

        # ---- attitude loop ---------------------------------------------------
        d1, d2, d3 = quat_from_zaxis_yaw(b3_des, ref["yaw"])
        e_R = vee([d1, d2, d3], cols)        # 0.5 (Rdᵀ R - Rᵀ Rd)^vee
        e_w = state.omega                    # desired body rate taken as zero

        kr = B.const(g.kr_att, like)
        kw = B.const(g.kw_att, like)
        torque = -kr * inertia * e_R - kw * inertia * e_w
        torque = torque + B.cross(state.omega, inertia * state.omega)

        # ---- mix to rotors ---------------------------------------------------
        wrench = B.cat([thrust, torque], axis=-1)
        rotor_thrust = wrench @ B.const(self._mix_inv.T, like)
        return B.clip(rotor_thrust / max_thrust, 0.0, 1.0)
