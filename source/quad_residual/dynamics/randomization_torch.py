"""Torch twin of :class:`randomization.DomainRandomizer` for the Isaac Lab env.

Same configuration object, same distributions, same curriculum semantics
(``scale`` 0 collapses every range onto the nominal ``QuadParams``, 1 is the
configured range). Random streams differ from the NumPy one -- the two envs
are compared statistically, not sample by sample (see the tests).
"""

from __future__ import annotations

import math

import torch

from .randomization import RandomizationCfg, _lerp


class TorchDomainRandomizer:
    def __init__(self, cfg: RandomizationCfg, params, num_envs: int, device, seed: int = 0):
        self.cfg, self.p, self.n, self.device = cfg, params, num_envs, device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.scale = 1.0
        f = dict(device=device, dtype=torch.float32)
        p = params
        self.mass = torch.full((num_envs, 1), p.mass, **f)
        self.inertia = torch.tensor(p.inertia, **f).repeat(num_envs, 1)
        self.max_thrust = torch.full((num_envs, 1), p.max_thrust_per_rotor, **f)
        self.motor_tau = torch.full((num_envs, 1), p.motor_tau, **f)
        self.drag = torch.full((num_envs, 1), p.drag_coeff, **f)
        self.delay_steps = torch.full((num_envs,), int(p.actuation_delay_steps),
                                      device=device, dtype=torch.long)
        self.wind_mean = torch.zeros((num_envs, 3), **f)
        self.wind = torch.zeros((num_envs, 3), **f)
        self.obs_bias = torch.zeros((num_envs, 3), **f)

    # ------------------------------------------------------------------ curriculum
    def set_scale(self, scale: float) -> None:
        self.scale = float(min(max(scale, 0.0), 1.0))

    @property
    def max_extra_delay(self) -> int:
        return int(self.cfg.extra_delay_steps[1]) if self.cfg.enabled else 0

    # ------------------------------------------------------------------ sampling
    def _u(self, pair, nominal, k):
        lo, hi = _lerp(*pair, self.scale, nominal)
        r = torch.rand((k, 1), generator=self.gen, device=self.device)
        return lo + (hi - lo) * r

    def _normal(self, std, shape):
        return std * torch.randn(shape, generator=self.gen, device=self.device)

    def apply(self, env_ids):
        """Resample physical parameters, wind and sensor bias for ``env_ids``."""
        if not self.cfg.enabled:
            return
        c, p, k = self.cfg, self.p, len(env_ids)
        self.mass[env_ids] = self._u(c.mass_range, p.mass, k)
        self.inertia[env_ids] = torch.tensor(p.inertia, device=self.device) * self._u(c.inertia_scale, 1.0, k)
        self.max_thrust[env_ids] = p.max_thrust_per_rotor * self._u(c.thrust_scale, 1.0, k)
        self.motor_tau[env_ids] = self._u(c.motor_tau_range, p.motor_tau, k)
        self.drag[env_ids] = self._u(c.drag_range, p.drag_coeff, k)
        lo, hi = _lerp(*c.extra_delay_steps, self.scale, 0.0)
        lo_i, hi_i = int(math.floor(lo + 0.5)), int(math.floor(hi + 0.5))
        self.delay_steps[env_ids] = int(p.actuation_delay_steps) + torch.randint(
            lo_i, hi_i + 1, (k,), generator=self.gen, device=self.device)
        speed = self._u(c.wind_range, 0.0, k)
        ang = 2 * math.pi * torch.rand((k, 1), generator=self.gen, device=self.device)
        self.wind_mean[env_ids] = torch.cat(
            [speed * torch.cos(ang), speed * torch.sin(ang), torch.zeros_like(speed)], -1)
        self.wind[env_ids] = self.wind_mean[env_ids]
        self.obs_bias[env_ids] = self._normal(c.obs_bias_pos * self.scale, (k, 3))

    def gust(self):
        """Per-physics-step wind jitter around each episode's mean wind."""
        if not self.cfg.enabled or self.scale <= 0:
            self.wind = self.wind_mean.clone()
            return
        j = self._normal(self.cfg.gust_std * self.scale, (self.n, 3))
        j[:, 2] *= 0.3
        self.wind = self.wind_mean + j

    def corrupt_obs(self, pos, vel, omega):
        if not self.cfg.enabled or self.scale <= 0:
            return pos, vel, omega
        c, s = self.cfg, self.scale
        return (pos + self.obs_bias + self._normal(c.obs_noise_pos * s, pos.shape),
                vel + self._normal(c.obs_noise_vel * s, vel.shape),
                omega + self._normal(c.obs_noise_omega * s, omega.shape))
