"""Actuator model: per-environment transport delay + first-order motor lag.

The single definition used by the CPU plant (``quadrotor.py``), by system
identification (``scripts/sysid.py``) and -- through the torch twin below -- by
the Isaac Lab env, which has to apply it by hand because PhysX models neither.

Delay semantics: with ``delay_steps[i] = d`` the command reaching the motor at
step k is the one issued at step ``k - d``; ``d = 0`` means no delay. The
buffer holds the last ``D_max`` issued commands, oldest first, so every env can
have its own ``d`` in ``[0, D_max]`` while sharing one array.

Shapes: ``delay_buf (N, D_max, C)``, ``motor/cmd (N, C)``, ``delay_steps (N,)``
integer, ``alpha (N, 1)`` or scalar. ``C`` is 4 rotors in the plant, 1 in sysid.
"""

from __future__ import annotations

import numpy as np


def actuate(delay_buf, motor, cmd, delay_steps, alpha):
    """NumPy version. Returns ``(new_motor, new_delay_buf)``."""
    d_max = delay_buf.shape[1]
    ext = np.concatenate([delay_buf, cmd[:, None, :]], axis=1)      # (N, D_max+1, C)
    rows = np.arange(ext.shape[0])
    applied = ext[rows, d_max - np.asarray(delay_steps)]             # (N, C)
    new_motor = np.clip(motor + alpha * (applied - motor), 0.0, 1.0)
    return new_motor.astype(motor.dtype), ext[:, 1:, :]


def actuate_torch(delay_buf, motor, cmd, delay_steps, alpha):
    """Torch twin of :func:`actuate` for the Isaac Lab env (same semantics)."""
    import torch

    d_max = delay_buf.shape[1]
    ext = torch.cat([delay_buf, cmd.unsqueeze(1)], dim=1)
    rows = torch.arange(ext.shape[0], device=ext.device)
    applied = ext[rows, d_max - delay_steps.long()]
    new_motor = (motor + alpha * (applied - motor)).clamp(0.0, 1.0)
    return new_motor, ext[:, 1:, :]
