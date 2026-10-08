"""Tiny array-backend shim.

The physics, the baseline controller and the metrics are written once and run
on either NumPy (``sim_lite``, CPU, no GPU needed) or torch (Isaac Lab, GPU,
thousands of parallel environments). Only the handful of operations whose
names or signatures differ between the two libraries are wrapped here.

Everything else -- arithmetic, broadcasting, indexing, ``.sum(-1)`` -- is
already spelled identically in both, so the rest of the code is plain array
code with no backend awareness at all.
"""

from __future__ import annotations

import numpy as np


class _NumpyBackend:
    name = "numpy"
    lib = np

    stack = staticmethod(np.stack)
    sqrt = staticmethod(np.sqrt)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    exp = staticmethod(np.exp)
    abs = staticmethod(np.abs)
    where = staticmethod(np.where)
    zeros_like = staticmethod(np.zeros_like)
    ones_like = staticmethod(np.ones_like)

    maximum = staticmethod(np.maximum)
    minimum = staticmethod(np.minimum)
    arccos = staticmethod(np.arccos)

    @staticmethod
    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    @staticmethod
    def const(x, like):
        """A constant (scalar / list / array) as an array matching ``like``."""
        return np.asarray(x, dtype=like.dtype)

    @staticmethod
    def any_nonzero(x):
        return bool(np.any(x != 0))

    @staticmethod
    def cat(arrays, axis=-1):
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def cross(a, b):
        return np.cross(a, b)

    @staticmethod
    def atan2(y, x):
        return np.arctan2(y, x)

    @staticmethod
    def norm(x, axis=-1, keepdims=True):
        return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))

    @staticmethod
    def zeros(shape, like=None):
        dtype = like.dtype if like is not None else np.float32
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def asarray(x, like=None):
        dtype = like.dtype if like is not None else np.float32
        return np.asarray(x, dtype=dtype)


class _TorchBackend:
    name = "torch"

    def __init__(self, torch):
        self.lib = torch
        self.stack = torch.stack
        self.sqrt = torch.sqrt
        self.sin = torch.sin
        self.cos = torch.cos
        self.exp = torch.exp
        self.abs = torch.abs
        self.where = torch.where
        self.zeros_like = torch.zeros_like
        self.ones_like = torch.ones_like
        self.atan2 = torch.atan2
        self.maximum = torch.maximum
        self.minimum = torch.minimum
        self.arccos = torch.arccos

    def clip(self, x, lo, hi):
        t = self.lib
        if isinstance(lo, t.Tensor) or isinstance(hi, t.Tensor):
            return t.minimum(t.maximum(x, lo), hi)
        return t.clamp(x, lo, hi)

    def const(self, x, like):
        return self.lib.as_tensor(x, dtype=like.dtype, device=like.device)

    def any_nonzero(self, x):
        return bool((x != 0).any())

    def cat(self, arrays, axis=-1):
        return self.lib.cat(arrays, dim=axis)

    def cross(self, a, b):
        return self.lib.cross(a, b, dim=-1)

    def norm(self, x, axis=-1, keepdims=True):
        return self.lib.sqrt((x * x).sum(dim=axis, keepdim=keepdims))

    def zeros(self, shape, like=None):
        if like is not None:
            return self.lib.zeros(shape, dtype=like.dtype, device=like.device)
        return self.lib.zeros(shape)

    def asarray(self, x, like=None):
        if like is not None:
            return self.lib.as_tensor(x, dtype=like.dtype, device=like.device)
        return self.lib.as_tensor(x)


_NP = _NumpyBackend()
_TORCH = None


def xp(sample):
    """Return the backend matching ``sample``'s array type."""
    global _TORCH
    if isinstance(sample, np.ndarray):
        return _NP
    try:
        import torch  # noqa: WPS433  (optional dependency)
    except ImportError as exc:  # pragma: no cover
        raise TypeError(f"unsupported array type {type(sample)!r}") from exc
    if isinstance(sample, torch.Tensor):
        if _TORCH is None:
            _TORCH = _TorchBackend(torch)
        return _TORCH
    raise TypeError(f"unsupported array type {type(sample)!r}")


# ---------------------------------------------------------------- quaternions
# Convention: (w, x, y, z), scalar first, matching Isaac Lab.

def quat_normalize(q):
    B = xp(q)
    n = B.norm(q)
    return q / B.where(n < 1e-9, B.ones_like(n), n)


def quat_mul(a, b):
    B = xp(a)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return B.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate(q, v):
    """Rotate ``v`` (body -> world) by quaternion ``q``."""
    B = xp(q)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uv = B.cross(u, v)
    return v + 2.0 * (w * uv + B.cross(u, uv))


def quat_rotate_inverse(q, v):
    """Rotate ``v`` (world -> body)."""
    B = xp(q)
    qc = B.cat([q[..., 0:1], -q[..., 1:4]], axis=-1)
    return quat_rotate(qc, v)


def quat_to_rotmat_cols(q):
    """Return the three columns of R(q); column 3 is the body z axis in world."""
    B = xp(q)
    cols = []
    for i in range(3):
        e = B.zeros(q.shape[:-1] + (3,), like=q)
        e[..., i] = 1.0
        cols.append(quat_rotate(q, e))
    return cols


def quat_from_zaxis_yaw(b3_des, yaw):
    """Minimal-tilt attitude: body z aligned to ``b3_des`` with the given yaw.

    Returns the rotation as its three body-axis columns expressed in world,
    which is what the attitude error needs -- avoids a quaternion round trip.
    """
    B = xp(b3_des)
    n3 = B.norm(b3_des)
    b3 = b3_des / B.where(n3 < 1e-9, B.ones_like(n3), n3)
    c1 = B.stack([B.cos(yaw), B.sin(yaw), B.zeros_like(yaw)], axis=-1)
    b2 = B.cross(b3, c1)
    n2 = B.norm(b2)
    b2 = b2 / B.where(n2 < 1e-9, B.ones_like(n2), n2)
    b1 = B.cross(b2, b3)
    return b1, b2, b3


def vee(mat_cols_a, mat_cols_b):
    """vee(0.5 * (Aᵀ B - Bᵀ A)) for rotations given as column triples.

    With ``A = R_d`` and ``B = R`` this is Lee et al.'s attitude error
    ``e_R = 0.5 (R_dᵀ R - Rᵀ R_d)^vee`` (checked numerically in the tests)."""
    B = xp(mat_cols_a[0])
    a1, a2, a3 = mat_cols_a
    b1, b2, b3 = mat_cols_b
    # (Bᵀ A)_ij = b_i · a_j
    m21 = (a3 * b2).sum(-1) - (b3 * a2).sum(-1)
    m02 = (a1 * b3).sum(-1) - (b1 * a3).sum(-1)
    m10 = (a2 * b1).sum(-1) - (b2 * a1).sum(-1)
    return 0.5 * B.stack([m21, m02, m10], axis=-1)
