"""Out-of-distribution supervisor: online actuation-lag estimate + fall-back.

Why lag. Out of distribution the policies crash where the *effective actuation
lag* ``L = delay + motor_tau`` exceeds anything seen in training (Isaac stress
test: residual 2 / 1450 crashes with L inside the training range, 66 / 300
beyond it). Pure delay and a first-order motor lag cost almost the same phase
at the attitude-loop bandwidth, so their sum is what the policy is sensitive
to, and it is also what can be identified from closed-loop data -- the two
terms individually are nearly interchangeable.

Estimator. Rotational dynamics per roll / pitch axis, ignoring the small
gyroscopic term: ``omega_dot = b * u_nom(t - d)`` filtered by ``1/(tau s + 1)``,
where ``u_nom`` is the angular acceleration the issued rotor command would
produce on the *nominal* airframe (known to the controller) and ``b`` absorbs
the unknown thrust / inertia scale. For every candidate ``(d, tau)`` on a grid
the supervisor replays the issued commands through that delay + lag at the
physics rate (the same actuator model as the plant), predicts the change of
body rate over each control interval, and fits ``b`` by least squares against
the measured change (the same noisy gyro measurement the policy sees). The
candidate with the smallest residual gives ``L_hat = d * dt + tau``. All sums
are running sums, so the estimate uses the whole episode so far at O(grid)
cost per step. ``scripts/sysid.py`` is the offline, single-signal version.

Gate. For the first ``probe_s`` seconds the classical controller flies alone
(identification window). After that the policy is enabled unless the estimate
leaves the training distribution; once tripped, an env stays on the classical
controller for the rest of the episode (latching -- no switching back and
forth). Estimation continues while the policy flies. Two gates:

* ``lag`` (default): ``L_hat > lag_max``, with ``lag_max`` the largest lag in
  the *training* distribution, ``(nominal_delay + max_extra_delay) * dt +
  max_motor_tau`` -- a property of the training config, not tuned on tests.
* ``support``: ``d_hat > max training delay`` or ``tau_hat > max training
  tau`` -- the whole training rectangle. More conservative, and with more false
  alarms, because ``d`` and ``tau`` are individually only identified to about
  one physics step / 5 ms.

Calibration (``scripts/check_supervisor.py``, seeds disjoint from every
evaluation seed): with the classical controller flying, the lag estimate is
within ~2 ms after 0.5 s, so ``probe_s = 0.5``.

Torch only (CPU or GPU); NumPy inputs are converted, outputs follow the input
type, so sim_lite and the Isaac core share this one implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..dynamics.quadrotor import QuadParams, mixing_matrix


@dataclass
class SupervisorCfg:
    enabled: bool = False
    probe_s: float = 0.5               # classical-only identification window per episode
    gate: str = "lag"                  # "lag" | "support"
    lag_max: float | None = None       # s; None -> training-distribution maximum
    delay_max: int | None = None       # physics steps (support gate)
    tau_max_train: float | None = None # s (support gate)
    max_delay_steps: int = 12          # candidate delays 0..max (physics steps)
    tau_min: float = 0.01              # candidate motor time constants (s)
    tau_max: float = 0.12
    tau_points: int = 23


def training_support(params: QuadParams, rand_cfg) -> dict:
    """Largest delay (physics steps), motor tau and effective lag the policy was
    trained on -- read from the training randomization config."""
    extra = int(rand_cfg.extra_delay_steps[1]) if rand_cfg.enabled else 0
    tau = float(rand_cfg.motor_tau_range[1]) if rand_cfg.enabled else params.motor_tau
    d = params.actuation_delay_steps + extra
    return {"lag_max": d * params.dt + tau, "delay_max": d, "tau_max_train": tau}


def training_lag_max(params: QuadParams, rand_cfg) -> float:
    return training_support(params, rand_cfg)["lag_max"]


class LagSupervisor:
    def __init__(self, cfg: SupervisorCfg, params: QuadParams, num_envs: int,
                 decimation: int, support: dict, device="cpu"):
        """``support``: :func:`training_support` of the policy's *training* config."""
        self.cfg, self.p, self.n, self.dec = cfg, params, num_envs, int(decimation)
        self.device = torch.device(device)
        if cfg.gate not in ("lag", "support"):
            raise ValueError(f"gate must be lag|support, got {cfg.gate!r}")
        pick = lambda k: float(getattr(cfg, k) if getattr(cfg, k) is not None else support[k])
        self.lag_max, self.delay_max, self.tau_max = (pick("lag_max"), pick("delay_max"),
                                                     pick("tau_max_train"))
        f = dict(device=self.device, dtype=torch.float32)

        # command (N,4) -> nominal roll/pitch angular acceleration (N,2)
        mix = torch.as_tensor(mixing_matrix(params.arm_length, params.torque_coeff), **f)
        inertia = torch.as_tensor(params.inertia, **f)
        self.to_acc = (mix[1:3] * params.max_thrust_per_rotor / inertia[:2, None]).T  # (4,2)

        delays = torch.arange(cfg.max_delay_steps + 1, device=self.device)
        taus = torch.linspace(cfg.tau_min, cfg.tau_max, cfg.tau_points, **f)
        dd, tt = torch.meshgrid(delays.float(), taus, indexing="ij")
        self.cand_delay = dd.reshape(-1).long()                            # (G,)
        self.cand_tau = tt.reshape(-1)
        self.cand_lag = self.cand_delay.float() * params.dt + self.cand_tau
        self.alpha = (params.dt / self.cand_tau).clamp(max=1.0)[None, :, None]  # (1,G,1)
        G = self.cand_delay.numel()

        # Within a control interval, physics sub-step i applies the command
        # issued d physics steps earlier, i.e. control step k + floor((i - d) / dec).
        i = torch.arange(self.dec, device=self.device)
        off = torch.div(i[None, :] - self.cand_delay[:, None], self.dec, rounding_mode="floor")
        self.hist_len = int(-off.min().item()) + 1
        self.hist_idx = (self.hist_len - 1 + off)                          # (G, dec) into history

        self.hist = torch.zeros((num_envs, self.hist_len, 2), **f)          # oldest first
        self.filt = torch.zeros((num_envs, G, 2), **f)
        self.sxx = torch.zeros((num_envs, G, 2), **f)
        self.sxy = torch.zeros((num_envs, G, 2), **f)
        self.syy = torch.zeros((num_envs, 2), **f)
        self.prev_omega = torch.zeros((num_envs, 2), **f)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.tripped = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.lag_hat = torch.full((num_envs,), float("nan"), **f)
        self.delay_hat = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.tau_hat = torch.full((num_envs,), float("nan"), **f)
        self.probe_steps = int(math.ceil(cfg.probe_s / (params.dt * self.dec) - 1e-9))

    # ----------------------------------------------------------------- helpers
    def _t(self, x):
        return torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x,
                               dtype=torch.float32, device=self.device)

    def _ids(self, ids):
        if ids is None:
            return torch.arange(self.n, device=self.device)
        return torch.as_tensor(np.asarray(ids) if not torch.is_tensor(ids) else ids,
                               dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------- API
    def reset(self, env_ids=None, omega_meas=None):
        """New episode for ``env_ids``. Episodes start from a steady hover, where
        the differential (roll/pitch) command and every filter state are zero.
        ``omega_meas``: the episode's first body-rate measurement (full (N,3)
        array or rows for ``env_ids``); if not given, the next ``update`` only
        records the measurement."""
        ids = self._ids(env_ids)
        for buf in (self.hist, self.filt, self.sxx, self.sxy, self.syy):
            buf[ids] = 0.0
        self.steps[ids] = 0
        self.tripped[ids] = False
        self.lag_hat[ids] = float("nan")
        if omega_meas is None:
            self.fresh[ids] = True
        else:
            om = self._t(omega_meas)[..., :2]
            self.prev_omega[ids] = om[ids] if om.shape[0] == self.n else om
            self.fresh[ids] = False

    def update(self, cmd, omega_meas):
        """After a control step: ``cmd`` (N,4) the rotor command that was issued,
        ``omega_meas`` (N,3) the body-rate measurement taken after it."""
        cmd, om = self._t(cmd), self._t(omega_meas)[:, :2]
        live = ~self.fresh
        # A fresh env's ``cmd`` belongs to the previous episode: feed zero (hover).
        u = (cmd @ self.to_acc) * live[:, None].float()                    # (N,2)
        self.hist = torch.cat([self.hist[:, 1:], u[:, None]], dim=1)
        # replay the interval through every candidate delay + lag
        dt = self.p.dt
        x = torch.zeros_like(self.filt)
        for i in range(self.dec):
            applied = self.hist[:, self.hist_idx[:, i]]                    # (N,G,2)
            self.filt = self.filt + self.alpha * (applied - self.filt)
            x = x + dt * self.filt
        y = om - self.prev_omega                                           # (N,2)
        w = live[:, None, None].float()
        self.sxx += w * x * x
        self.sxy += w * x * y[:, None]
        self.syy += w[:, 0] * y * y
        self.prev_omega = om
        self.steps += live.long()
        self.fresh[:] = False

        sse = (self.syy[:, None] - self.sxy ** 2 / self.sxx.clamp_min(1e-12)).sum(-1)  # (N,G)
        best = sse.argmin(-1)
        self.lag_hat = self.cand_lag[best]
        self.delay_hat = self.cand_delay[best]
        self.tau_hat = self.cand_tau[best]
        if self.cfg.gate == "lag":
            out = self.lag_hat > self.lag_max + 1e-6
        else:
            out = (self.delay_hat > self.delay_max) | (self.tau_hat > self.tau_max + 1e-6)
        self.tripped |= (self.steps >= self.probe_steps) & out

    def allow(self):
        """(N,) bool: may the policy act this control step?"""
        return (self.steps >= self.probe_steps) & ~self.tripped
