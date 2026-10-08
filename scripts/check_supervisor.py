#!/usr/bin/env python3
"""Calibrate / check the OOD supervisor's lag estimator (CPU, sim_lite, seconds).

The classical controller flies (the supervisor's identification window) while
the delay range is widened to 1-10 physics steps, so the estimator sees both
in- and out-of-distribution airframes. Reported per identification window:
lag-estimate error, and for each gate the detection rate (OOD airframes
flagged) and false-alarm rate (in-distribution airframes flagged).

Seeds here (777, 778, ...) are disjoint from every evaluation seed; the probe
length is chosen from this script, never from evaluation results.

    python3 scripts/check_supervisor.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np
from quad_residual import config as C
from quad_residual.controllers.supervisor import LagSupervisor, SupervisorCfg, training_support
from quad_residual.dynamics.quadrotor import QuadParams
from quad_residual.sim_lite.env import QuadTrackEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 778])
    ap.add_argument("--trajectory", default=None)
    args = ap.parse_args()

    p = QuadParams()
    support = training_support(p, C.rand_cfg(C.load()))          # the training distribution
    windows = (0.2, 0.5, 1.0, 2.0)
    rows = {w: {"err": [], "far": [], "in": {"lag": [], "support": []},
                "out": {"lag": [], "support": []}} for w in windows}
    for seed in args.seeds:
        conf = C.load(overrides=["domain_rand.latency.extra_delay_steps=[0,9]"])
        rt = dict(num_envs=args.envs, policy_mode="baseline", seed=seed)
        if args.trajectory:
            rt["trajectory"] = args.trajectory
        env = QuadTrackEnv(C.env_cfg(conf, **rt))
        env.set_curriculum(1.0)
        env.reset()
        sup = LagSupervisor(SupervisorCfg(enabled=True, probe_s=0.0), p, args.envs,
                            env.cfg.decimation, support)
        sup.reset(None, env.meas.omega)
        d = env.model.delay_steps.copy()
        tau = env.model.motor_tau[:, 0].astype(float).copy()
        lag = d * p.dt + tau
        ood = (d > support["delay_max"]) | (tau > support["tau_max_train"] + 1e-9)
        ood_lag = lag > support["lag_max"] + 1e-6
        alive = np.ones(args.envs, bool)
        for k in range(int(round(max(windows) / env.dt))):
            _, _, term, _, _ = env.step(np.zeros((args.envs, 4), np.float32))
            alive &= ~term
            sup.update(env.last_cmd, env.meas.omega)
            t = next((w for w in windows if k + 1 == int(round(w / env.dt))), None)
            if t is not None:
                lh = sup.lag_hat.numpy()
                dh, th = sup.delay_hat.numpy(), sup.tau_hat.numpy()
                flag_lag = lh > support["lag_max"] + 1e-6
                flag_sup = (dh > support["delay_max"]) | (th > support["tau_max_train"] + 1e-6)
                r = rows[t]
                r["err"].append((lh - lag)[alive])
                r["in"]["lag"].append(flag_lag[alive & ~ood_lag])
                r["out"]["lag"].append(flag_lag[alive & ood_lag])
                r["far"].append(flag_lag[alive & (lag > support["lag_max"] + 0.005)])
                r["in"]["support"].append(flag_sup[alive & ~ood])
                r["out"]["support"].append(flag_sup[alive & ood])

    print(f"training support: lag <= {1000 * support['lag_max']:.0f} ms, delay <= "
          f"{support['delay_max']} steps, tau <= {1000 * support['tau_max_train']:.0f} ms; "
          f"{len(args.seeds)} x {args.envs} airframes, delay 1-10 steps")
    print(f"{'window':>7} {'lag MAE':>8} {'p90':>6} | {'lag gate: detect':>17} {'(>5ms out)':>10} "
          f"{'false alarm':>12} | "
          f"{'support gate: detect':>21} {'false alarm':>12}")
    for w in windows:
        r = rows[w]
        e = np.abs(np.concatenate(r["err"])) * 1000
        c = {g: {s: np.concatenate(r[s][g]).mean() for s in ("in", "out")} for g in ("lag", "support")}
        print(f"{w:>6.2f}s {e.mean():>6.1f}ms {np.percentile(e, 90):>4.1f}ms | "
              f"{c['lag']['out']:>17.3f} {np.concatenate(r['far']).mean():>10.3f} "
              f"{c['lag']['in']:>12.3f} | "
              f"{c['support']['out']:>21.3f} {c['support']['in']:>12.3f}")


if __name__ == "__main__":
    main()
