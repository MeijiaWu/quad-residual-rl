"""Domain randomization and the sim-to-real noise/latency model.

Split into three groups because they close different parts of the reality gap:

* **parameter**  -- mass, inertia, thrust constant, motor time constant.
  Covers manufacturing spread and payload changes.
* **latency**    -- extra actuation delay steps, drawn per env per episode and
  written into ``model.delay_steps``. The single most commonly underestimated
  term; a policy trained at zero delay generally fails on hardware regardless
  of how well the rest matches.
* **observation** -- additive sensor noise and bias on the state the policy sees.

``scale`` lets a curriculum ramp every range from 0 to 1. At 0 every range
collapses onto the *nominal* value (``QuadParams``, no wind, no extra delay,
no noise); at 1 it is exactly the configured ``[lo, hi]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomizationCfg:
    mass_range: tuple = (0.75, 1.05)          # kg
    inertia_scale: tuple = (0.8, 1.25)        # multiplier on nominal
    thrust_scale: tuple = (0.85, 1.15)        # multiplier on max thrust
    motor_tau_range: tuple = (0.02, 0.06)     # s
    drag_range: tuple = (0.05, 0.20)
    extra_delay_steps: tuple = (0, 3)         # inclusive
    wind_range: tuple = (0.0, 3.0)            # m/s, constant per episode
    gust_std: float = 0.4                     # m/s per step, OU-ish jitter
    obs_noise_pos: float = 0.01               # m
    obs_noise_vel: float = 0.05               # m/s
    obs_noise_omega: float = 0.02             # rad/s
    obs_bias_pos: float = 0.02                # m, constant per episode
    enabled: bool = True


def _lerp(lo, hi, scale, nominal):
    """Shrink ``[lo, hi]`` onto ``nominal`` as ``scale`` goes to 0."""
    return nominal + (lo - nominal) * scale, nominal + (hi - nominal) * scale


class DomainRandomizer:
    def __init__(self, cfg: RandomizationCfg, num_envs: int, seed: int = 0):
        self.cfg = cfg
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.scale = 1.0
        self.obs_bias = np.zeros((num_envs, 3), np.float32)
        self.wind = np.zeros((num_envs, 3), np.float32)

    def set_scale(self, scale: float) -> None:
        """Curriculum hook: 0 = nominal physics, 1 = full randomization."""
        self.scale = float(np.clip(scale, 0.0, 1.0))

    def _u(self, rng_pair, nominal, size):
        lo, hi = _lerp(*rng_pair, self.scale, nominal)
        return self.rng.uniform(lo, hi, size=size).astype(np.float32)

    def apply(self, model, params, env_ids=None):
        """Write randomized parameters into ``model`` for the given envs."""
        if not self.cfg.enabled:
            return
        ids = slice(None) if env_ids is None else env_ids
        k = self.n if env_ids is None else len(env_ids)
        c = self.cfg

        model.mass[ids] = self._u(c.mass_range, params.mass, (k, 1))
        model.inertia[ids] = (np.asarray(params.inertia, np.float32)
                              * self._u(c.inertia_scale, 1.0, (k, 1)))
        model.max_thrust[ids] = (params.max_thrust_per_rotor
                                 * self._u(c.thrust_scale, 1.0, (k, 1)))
        model.motor_tau[ids] = self._u(c.motor_tau_range, params.motor_tau, (k, 1))
        model.drag[ids] = self._u(c.drag_range, params.drag_coeff, (k, 1))
        model.delay_steps[ids] = (int(params.actuation_delay_steps)
                                  + self.sample_extra_delay(k))

        speed = self._u(c.wind_range, 0.0, (k, 1))
        ang = self.rng.uniform(0, 2 * np.pi, size=(k, 1)).astype(np.float32)
        self.wind[ids] = np.concatenate(
            [speed * np.cos(ang), speed * np.sin(ang), np.zeros_like(speed)], -1)
        model.wind[ids] = self.wind[ids]

        self.obs_bias[ids] = self.rng.normal(
            0.0, c.obs_bias_pos * self.scale, size=(k, 3)).astype(np.float32)

    def gust(self, model):
        """Per-step wind jitter around the episode's mean wind."""
        if not self.cfg.enabled or self.scale <= 0:
            return
        j = self.rng.normal(0.0, self.cfg.gust_std * self.scale,
                            size=(self.n, 3)).astype(np.float32)
        j[:, 2] *= 0.3
        model.wind = self.wind + j

    def sample_extra_delay(self, k: int) -> np.ndarray:
        """Extra delay steps per env, integers in the curriculum-scaled range."""
        lo, hi = _lerp(*self.cfg.extra_delay_steps, self.scale, 0.0)
        lo_i, hi_i = int(np.floor(lo + 0.5)), int(np.floor(hi + 0.5))
        return self.rng.integers(lo_i, hi_i + 1, size=k)

    @property
    def max_extra_delay(self) -> int:
        """Buffer size the plant needs to honour every possible draw."""
        return int(self.cfg.extra_delay_steps[1]) if self.cfg.enabled else 0

    def corrupt_obs(self, pos, vel, omega):
        if not self.cfg.enabled or self.scale <= 0:
            return pos, vel, omega
        c, s = self.cfg, self.scale
        pos = pos + self.obs_bias + self.rng.normal(
            0, c.obs_noise_pos * s, pos.shape).astype(np.float32)
        vel = vel + self.rng.normal(
            0, c.obs_noise_vel * s, vel.shape).astype(np.float32)
        omega = omega + self.rng.normal(
            0, c.obs_noise_omega * s, omega.shape).astype(np.float32)
        return pos, vel, omega
