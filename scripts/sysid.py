#!/usr/bin/env python3
"""Identify motor time constant and actuation delay from a command/response log.

Run with no arguments for a self-check on synthetic data of known parameters;
point --log at a CSV of real bench data (columns: t, cmd, measured) to identify
a real airframe. The recovered values go straight into QuadParams, and the
spread you see across runs is what should set the randomization range.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np
from quad_residual.dynamics.actuator import actuate


def simulate(tau, delay, cmd, dt):
    """Motor response to ``cmd`` for one delay and one or many time constants.

    Uses the plant's own actuator model (``dynamics/actuator.py``), so what is
    identified here is exactly what the simulator will replay. ``tau`` may be
    a scalar or a 1-D grid; the grid is simulated in one batch. Returns shape
    ``(len(cmd),)`` for scalar ``tau``, else ``(len(tau), len(cmd))``.
    """
    taus = np.atleast_1d(np.asarray(tau, np.float64))
    g, d = taus.size, int(delay)
    alpha = np.clip(dt / np.maximum(taus, 1e-6), 0.0, 1.0)[:, None]
    buf = np.zeros((g, d, 1))
    motor = np.zeros((g, 1))
    delays = np.full(g, d)
    y = np.empty((g, len(cmd)))
    for i, u in enumerate(cmd):
        motor, buf = actuate(buf, motor, np.full((g, 1), u), delays, alpha)
        y[:, i] = motor[:, 0]
    return y[0] if np.ndim(tau) == 0 else y


def identify(cmd, meas, dt, tau_grid=None, delay_grid=range(0, 8)):
    """Grid search over (delay, tau). ``delay = 0`` is a real candidate."""
    tau_grid = tau_grid if tau_grid is not None else np.linspace(0.005, 0.12, 120)
    best = (None, None, np.inf)
    for d in delay_grid:
        err = ((simulate(tau_grid, d, cmd, dt) - meas[None, :]) ** 2).mean(axis=1)
        j = int(np.argmin(err))
        if err[j] < best[2]:
            best = (tau_grid[j], d, err[j])
    return {"motor_tau": float(best[0]), "delay_steps": int(best[1]),
            "mse": float(best[2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="CSV: t,cmd,measured")
    ap.add_argument("--dt", type=float, default=0.004)
    ap.add_argument("--true-delay", type=int, default=2,
                    help="ground-truth delay for the synthetic self-check")
    args = ap.parse_args()

    if args.log:
        d = np.loadtxt(args.log, delimiter=",", skiprows=1)
        t, cmd, meas = d[:, 0], d[:, 1], d[:, 2]
        dt = float(np.median(np.diff(t)))
    else:
        dt = args.dt
        rng = np.random.default_rng(0)
        n = 2000
        cmd = (rng.random(n) < 0.01).cumsum() % 2 * 0.4 + 0.3   # random square wave
        true_tau, true_delay = 0.037, args.true_delay
        meas = simulate(true_tau, true_delay, cmd, dt)
        meas += rng.normal(0, 0.003, n)
        print(f"synthetic ground truth: tau={true_tau}s delay={true_delay} steps")

    out = identify(cmd, meas, dt)
    print(f"identified: tau={out['motor_tau']:.4f}s  "
          f"delay={out['delay_steps']} steps  mse={out['mse']:.2e}")
    return out


if __name__ == "__main__":
    main()
