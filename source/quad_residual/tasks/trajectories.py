"""Reference trajectory generators (batched, backend-agnostic).

Every generator returns the dict the controller and the reward expect:
``pos``, ``vel``, ``acc`` (feed-forward) and ``yaw``. Having ``acc`` available
analytically matters -- a baseline denied feed-forward looks artificially bad
next to a learned policy, which makes the comparison meaningless.
"""

from __future__ import annotations

import numpy as np

from ..backend import xp


class Trajectory:
    name = "base"

    def sample(self, t, rng=None):
        raise NotImplementedError

    @staticmethod
    def _zeros(t):
        B = xp(t)
        return B.zeros(tuple(t.shape) + (3,), like=t if B.name == "torch" else None)


class Hover(Trajectory):
    name = "hover"

    def __init__(self, height=1.5):
        self.height = height

    def sample(self, t, rng=None):
        pos = self._zeros(t)
        pos[..., 2] = self.height
        return {"pos": pos, "vel": self._zeros(t), "acc": self._zeros(t),
                "yaw": xp(t).zeros_like(t)}


class Lemniscate(Trajectory):
    """Figure-of-eight: excites all three axes and reverses curvature."""

    name = "lemniscate"

    def __init__(self, a=1.5, height=1.5, period=8.0, z_amp=0.3):
        self.a, self.h, self.T, self.z = a, height, period, z_amp

    def sample(self, t, rng=None):
        B = xp(t)
        w = 2.0 * np.pi / self.T
        s, c = B.sin(w * t), B.cos(w * t)
        s2, c2 = B.sin(2 * w * t), B.cos(2 * w * t)
        pos = B.stack([self.a * s, self.a * s2 / 2.0, self.h + self.z * c], -1)
        vel = B.stack([self.a * w * c, self.a * w * c2, -self.z * w * s], -1)
        acc = B.stack([-self.a * w**2 * s, -2 * self.a * w**2 * s2,
                       -self.z * w**2 * c], -1)
        if B.name == "numpy":
            pos, vel, acc = (x.astype(np.float32) for x in (pos, vel, acc))
        return {"pos": pos, "vel": vel, "acc": acc, "yaw": B.zeros_like(t)}


class RandomWaypoints(Trajectory):
    """Piecewise-constant setpoints -- the step-response stress case.

    Each env gets its own random walk of waypoints, redrawn on every reset:
    hover at the start point for the first ``dwell`` seconds, then a step to a
    new waypoint every ``dwell`` seconds. Step length is uniform in
    [0.5, 1] x ``max_step`` in a random 3-D direction, kept inside ``box``
    around the start point -- so a single step can never exceed the env's
    position-error termination limit by construction (it would be a crash
    produced by the test, not by the controller).

    Reference velocity and acceleration are zero: there is no feed-forward to
    exploit, which is the point of a step test.
    """

    name = "waypoints"

    def __init__(self, box=(2.0, 2.0, 1.0), height=1.5, dwell=3.0, max_step=1.5,
                 n_wp=16, seed=0):
        self.box = np.asarray(box, np.float32)
        self.h, self.dwell, self.max_step, self.n_wp = height, dwell, max_step, n_wp
        self._rng = np.random.default_rng(seed)
        self._wp = None                               # (num_envs, n_wp, 3)

    def reset(self, env_ids, num_envs, rng=None):
        r = rng or self._rng
        if not isinstance(env_ids, np.ndarray):           # torch index tensor
            env_ids = np.asarray(env_ids.cpu() if hasattr(env_ids, "cpu") else env_ids)
        if self._wp is None or self._wp.shape[0] != num_envs:
            self._wp = None
            env_ids = np.arange(num_envs)
            self._wp = np.zeros((num_envs, self.n_wp, 3), np.float32)
        env_ids = np.asarray(env_ids)
        k = env_ids.size
        start = np.tile(np.float32([0.0, 0.0, self.h]), (k, 1))
        wp = np.empty((k, self.n_wp, 3), np.float32)
        wp[:, 0] = start
        for j in range(1, self.n_wp):
            d = r.normal(size=(k, 3))
            d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
            step = d * r.uniform(0.5, 1.0, size=(k, 1)) * self.max_step
            wp[:, j] = np.clip(wp[:, j - 1] + step, start - self.box, start + self.box)
        self._wp[env_ids] = wp
        self._wp_t = None                              # torch mirror is stale

    def sample(self, t, rng=None):
        """``t`` may be NumPy or torch. The table is always generated with the
        NumPy RNG (so both envs draw identical walks for the same seed) and
        mirrored to ``t``'s device when ``t`` is a tensor."""
        B = xp(t)
        n = t.shape[0]
        if self._wp is None or self._wp.shape[0] != n:
            self.reset(np.arange(n), n, rng)
        if B.name == "numpy":
            idx = np.minimum((t / self.dwell).astype(np.int64), self.n_wp - 1)
            pos = self._wp[np.arange(n), idx]
        else:
            torch = B.lib
            if getattr(self, "_wp_t", None) is None or self._wp_t.device != t.device:
                self._wp_t = torch.as_tensor(self._wp, device=t.device)
            idx = torch.clamp((t / self.dwell).long(), max=self.n_wp - 1)
            pos = self._wp_t[torch.arange(n, device=t.device), idx]
        return {"pos": pos, "vel": self._zeros(t), "acc": self._zeros(t),
                "yaw": B.zeros_like(t)}


REGISTRY = {c.name: c for c in (Hover, Lemniscate, RandomWaypoints)}


def make(name: str, **kw) -> Trajectory:
    if name not in REGISTRY:
        raise KeyError(f"unknown trajectory {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
