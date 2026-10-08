"""Observation, reward and termination -- one implementation for both envs.

``sim_lite`` (NumPy, CPU) and the Isaac Lab env (torch, GPU) both call these
functions; they go through the backend shim, so the task cannot silently differ
between the two. ``cfg`` is anything with the reward / termination fields of
``sim_lite.env.EnvCfg`` (the Isaac cfg has the same names).

State conventions (both envs): position and linear velocity in the world frame
(relative to the env origin), quaternion (w, x, y, z) body -> world, angular
velocity in the body frame.
"""

from __future__ import annotations

from .backend import xp

OBS_DIM = 20
ACT_DIM = 4


def observation(ref, meas_pos, meas_vel, quat, meas_omega, prev_action):
    """20-dim policy observation. Measurements are the (noisy) sensed state."""
    B = xp(meas_pos)
    return B.cat(
        [
            ref["pos"] - meas_pos,     # 3  tracking error, world
            ref["vel"] - meas_vel,     # 3
            quat,                      # 4  attitude
            meas_omega,                # 3  body rate
            ref["acc"],                # 3  feed-forward the policy can exploit
            prev_action,               # 4  lets the policy reason about smoothness -> 20
        ],
        axis=-1,
    )


def reward(cfg, ref, pos, vel, omega, action, prev_action):
    """Per-env reward and its terms (true state, not measurements)."""
    B = xp(pos)
    e_pos = B.norm(ref["pos"] - pos, keepdims=False)
    e_vel = B.norm(ref["vel"] - vel, keepdims=False)
    rate = B.norm(omega, keepdims=False)
    act = B.norm(action, keepdims=False)
    smooth = B.norm(action - prev_action, keepdims=False)
    # Exponential on position keeps the gradient informative near zero error,
    # where a quadratic penalty has already flattened out.
    terms = {
        "pos": cfg.w_pos * B.exp(-2.0 * e_pos),
        "vel": -cfg.w_vel * e_vel,
        "rate": -cfg.w_rate * rate,
        "action": -cfg.w_action * act,
        "smooth": -cfg.w_smooth * smooth,
        "alive": cfg.w_alive * B.ones_like(e_pos),
    }
    total = terms["pos"]
    for k in ("vel", "rate", "action", "smooth", "alive"):
        total = total + terms[k]
    return total, terms


def termination(cfg, ref, pos, quat):
    """Crash: position error, tilt, or ground."""
    B = xp(pos)
    e = B.norm(ref["pos"] - pos, keepdims=False)
    b3_z = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    tilt = B.arccos(B.clip(b3_z, -1.0, 1.0))
    return (e > cfg.max_pos_err) | (tilt > cfg.max_tilt) | (pos[:, 2] < 0.05)
