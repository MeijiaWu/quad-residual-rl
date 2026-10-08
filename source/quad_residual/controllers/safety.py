"""Deployment safety layer.

A learned policy is not trusted on hardware without this. Three mechanisms,
in increasing order of severity:

1. **Rate and magnitude limiting** -- caps how far and how fast the command can
   move, which also bounds the jerk the airframe sees.
2. **Anomaly monitoring** -- watches tilt, body rate, speed and tracking error.
   Any one out of bounds for ``trip_steps`` consecutive steps trips the guard.
3. **Fallback** -- once tripped, the classical controller takes over for the
   rest of the episode. Latching rather than chattering back and forth is
   deliberate: a policy that just failed should not get the aircraft back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyCfg:
    cmd_min: float = 0.0
    cmd_max: float = 1.0
    max_cmd_rate: float = 8.0      # per second, per rotor
    max_tilt: float = 1.05         # rad, ~60 deg
    max_rate: float = 12.0         # rad/s
    max_speed: float = 12.0        # m/s
    max_pos_err: float = 4.0       # m
    trip_steps: int = 3
    enabled: bool = True


class SafetyLayer:
    def __init__(self, cfg: SafetyCfg, num_envs: int, dt: float):
        self.c = cfg
        self.n = num_envs
        self.dt = dt
        self.prev_cmd = None
        self.bad_steps = np.zeros(num_envs, np.int32)
        self.tripped = np.zeros(num_envs, bool)

    def reset(self, env_ids=None, init_cmd=None):
        """Clear the monitor for ``env_ids``.

        ``init_cmd`` (per reset env, shape (k, 4)) seeds the rate limiter. It
        must be the command the aircraft is actually holding -- the hover
        command after a reset. Seeding with 0 would clamp the first commands of
        every new episode to a slow ramp up from zero thrust and drop the
        aircraft.
        """
        ids = slice(None) if env_ids is None else env_ids
        self.bad_steps[ids] = 0
        self.tripped[ids] = False
        if init_cmd is not None:
            if self.prev_cmd is None:
                self.prev_cmd = np.zeros((self.n, 4), np.float32)
            self.prev_cmd[ids] = init_cmd
        elif self.prev_cmd is not None:
            # No seed given: forget history so the next call passes unclamped.
            self.prev_cmd = None

    def monitor(self, state, ref_pos):
        """Return the boolean anomaly flag per environment."""
        b3_z = 1.0 - 2.0 * (state.quat[:, 1] ** 2 + state.quat[:, 2] ** 2)
        tilt = np.arccos(np.clip(b3_z, -1.0, 1.0))
        rate = np.linalg.norm(state.omega, axis=-1)
        speed = np.linalg.norm(state.vel, axis=-1)
        err = np.linalg.norm(state.pos - ref_pos, axis=-1)
        bad = ((tilt > self.c.max_tilt) | (rate > self.c.max_rate)
               | (speed > self.c.max_speed) | (err > self.c.max_pos_err))
        self.bad_steps = np.where(bad, self.bad_steps + 1, 0)
        self.tripped |= self.bad_steps >= self.c.trip_steps
        return bad

    def __call__(self, cmd, fallback_cmd, state=None, ref_pos=None):
        """Clip, rate-limit, and substitute the fallback where tripped."""
        if not self.c.enabled:
            return np.clip(cmd, self.c.cmd_min, self.c.cmd_max)
        if state is not None and ref_pos is not None:
            self.monitor(state, ref_pos)
        out = np.where(self.tripped[:, None], fallback_cmd, cmd)
        out = np.clip(out, self.c.cmd_min, self.c.cmd_max)
        if self.prev_cmd is None:
            self.prev_cmd = np.full_like(out, 0.0)
            self.prev_cmd[:] = out
        else:
            step = self.c.max_cmd_rate * self.dt
            out = np.clip(out, self.prev_cmd - step, self.prev_cmd + step)
        self.prev_cmd = out.copy()
        return out.astype(np.float32)
