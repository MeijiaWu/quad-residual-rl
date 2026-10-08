"""Pick the "robust" baseline bandwidth with a rule fixed before any RL result.

Rule: among candidate attitude bandwidths (position = attitude / 8), choose the
LARGEST one that is stable at EVERY corner of the configured randomization
range, extended by a margin, with the controller using NOMINAL parameters
(it never knows the sampled mass / inertia / thrust -- same as on hardware).

The test is stability, not accuracy: with nominal parameters and no integral
term the baseline keeps a steady-state offset under a mass or thrust mismatch,
and that offset is part of what the comparison is about, not a reason to
reject a bandwidth.

Corners: mass, inertia scale, thrust scale, motor time constant and actuation
delay each at their min / max -> 32 plants. Margin: +1 delay step and +20 % on
the largest motor time constant. Stable means: after 30 s from a 1 m offset (long enough for the slowest candidate to settle), mean body rate and mean speed over
the last 2 s below 0.05 (rad/s, m/s), position error bounded (< 1 m), and tilt
never above 1.3 rad (the env's termination limit).

The rule is deliberately about the base controller alone. It is not tuned
against the learned policy: the robust baseline is both the residual's base
and the safety fallback, so its only job is to never be what crashes.

    python scripts/select_baseline_gains.py
    python scripts/select_baseline_gains.py --set domain_rand.latency.extra_delay_steps=[0,5]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from quad_residual import config as C
from quad_residual.controllers.geometric import BASELINE_PRESETS, GeometricController, GeometricGains
from quad_residual.dynamics.quadrotor import QuadParams, QuadrotorModel

CANDIDATES = (20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0)
DELAY_MARGIN, TAU_MARGIN = 1, 1.2


def corners(rc, p: QuadParams):
    base_d = p.actuation_delay_steps
    lo_d, hi_d = rc.extra_delay_steps
    axes = [
        rc.mass_range,
        rc.inertia_scale,
        rc.thrust_scale,
        (rc.motor_tau_range[0], rc.motor_tau_range[1] * TAU_MARGIN),
        (base_d + lo_d, base_d + hi_d + DELAY_MARGIN),
    ]
    return list(itertools.product(*axes))


def check(att_wn: float, cs, p: QuadParams, integral: bool = True,
          seconds: float = 30.0, decimation: int = 5):
    """Return a boolean per corner: stable or not."""
    n = len(cs)
    d_max = max(int(c[4]) for c in cs)
    m = QuadrotorModel(p, n, max_delay_steps=d_max)
    arr = np.asarray(cs, np.float64)
    m.mass[:, 0] = arr[:, 0]
    m.inertia[:] = np.asarray(p.inertia, np.float32) * arr[:, 1:2]
    m.max_thrust[:, 0] = p.max_thrust_per_rotor * arr[:, 2]
    m.motor_tau[:, 0] = arr[:, 3]
    m.delay_steps[:] = arr[:, 4].astype(np.int64)
    m.drag[:] = p.drag_coeff

    ctrl = GeometricController(p, GeometricGains.from_bandwidth(att_wn, integral=integral), n,
                               dt=p.dt * decimation)
    nom_mass = np.full((n, 1), p.mass, np.float32)
    nom_inertia = np.tile(np.asarray(p.inertia, np.float32), (n, 1))
    nom_thrust = np.full((n, 1), p.max_thrust_per_rotor, np.float32)

    st = m.zero_state()
    st.pos[:] = np.float32([1.0, 0.0, 1.5])
    # Start every rotor at the true hover command so the delay buffer is consistent.
    hover = m.hover_command()
    st.motor[:] = hover
    st.delay_buf[:] = hover[:, None, :]
    # Same initial condition as the env: a steady hover, so the integrator
    # starts at the converged hover-thrust estimate.
    ctrl.reset(None, hover_ratio=(arr[:, 0] / p.mass) * (1.0 / arr[:, 2]))
    ref = dict(pos=np.tile(np.float32([0.0, 0.0, 1.5]), (n, 1)),
               vel=np.zeros((n, 3), np.float32), acc=np.zeros((n, 3), np.float32),
               yaw=np.zeros(n, np.float32))

    steps = int(seconds / p.dt)
    tail = int(2.0 / p.dt)
    rate_tail = np.zeros(n)
    speed_tail = np.zeros(n)
    max_tilt = np.zeros(n)
    for k in range(steps):
        if k % decimation == 0:
            cmd = ctrl(st, ref, nom_mass, nom_inertia, nom_thrust)
        st = m.step(st, cmd)
        b3z = 1.0 - 2.0 * (st.quat[:, 1] ** 2 + st.quat[:, 2] ** 2)
        tilt = np.arccos(np.clip(b3z, -1, 1))
        max_tilt = np.maximum(max_tilt, np.nan_to_num(tilt, nan=np.pi))
        if k >= steps - tail:
            rate_tail += np.linalg.norm(st.omega, axis=-1) / tail
            speed_tail += np.linalg.norm(st.vel, axis=-1) / tail
    err = np.linalg.norm(st.pos - ref["pos"], axis=-1)
    ok = (err < 1.0) & (rate_tail < 0.05) & (speed_tail < 0.05) & (max_tilt < 1.3)
    return np.nan_to_num(ok, nan=False).astype(bool), np.nan_to_num(err, nan=np.inf)


def select(rc, p: QuadParams, integral: bool = True, verbose=True):
    cs = corners(rc, p)
    chosen = None
    for wn in CANDIDATES:
        ok, err = check(wn, cs, p, integral)
        if verbose:
            bad = [cs[i] for i in np.nonzero(~ok)[0]]
            msg = (f"stable at all corners (worst steady-state offset {err.max():.2f} m)"
                   if ok.all() else None)
            msg = msg or \
                f"UNSTABLE at {len(bad)}/{len(cs)} corners, e.g. delay={int(bad[0][4])} tau={bad[0][3]*1e3:.0f}ms"
            print(f"  att wn {wn:5.1f} rad/s (pos {wn/8:.2f}): {msg}")
        if ok.all() and chosen is None:
            chosen = wn
            if not verbose:
                break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="NS.KEY=VALUE")
    args = ap.parse_args()
    conf = C.load(args.config_dir, args.set)
    rc = C.rand_cfg(conf)
    integral = bool(C.env_cfg(conf).baseline_integral)
    p = QuadParams()
    print(f"corners: delay {p.actuation_delay_steps}+{list(rc.extra_delay_steps)} (+{DELAY_MARGIN} margin), "
          f"motor tau {list(rc.motor_tau_range)} s (x{TAU_MARGIN} margin), mass/inertia/thrust at range ends; "
          f"position integral {'on' if integral else 'off'}")
    wn = select(rc, p, integral)
    print(f"\nselected robust bandwidth: {wn} rad/s  (BASELINE_PRESETS['robust'] = {BASELINE_PRESETS['robust']})")
    if wn != BASELINE_PRESETS["robust"]:
        print("-> preset differs from the rule's choice; update BASELINE_PRESETS and retrain.")
        sys.exit(1)


if __name__ == "__main__":
    main()
