#!/usr/bin/env python3
"""Evaluate controllers on the same episodes, one episode per env.

    python scripts/eval.py                                   # baselines + residual(zero) sanity check
    python scripts/eval.py --policy results/residual_s0.zip  # + a trained policy
    python scripts/eval.py --policy results/residual_s0.zip --safety
    python scripts/eval.py --policy results/residual_s0.zip --supervisor   # OOD fall-back

Always evaluated: both classical baselines (``fast`` and ``robust`` gain
presets, see controllers/geometric.py), so every report shows the
speed-vs-robustness trade-off a fixed-gain controller has to make. With
``--policy``, the policy's interface (mode, residual scale, the baseline it sits
on) comes from the checkpoint's ``.cfg.yaml``; the evaluation conditions come
from ``configs/domain_rand.yaml`` so all checkpoints -- ablations included --
face the same distribution (``--train-dist`` to use the training one instead).

Every env runs exactly one episode. A crash is counted as a crash and nothing
after it is used; accuracy is reported over survivors and compared pairwise
(same seed -> identical disturbances) against the residual's base controller.

With no --policy the residual is driven by a zero action, which must reproduce
its base baseline exactly -- a check that the residual plumbing adds nothing.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np
from quad_residual import config as C
from quad_residual.dynamics.quadrotor import QuadParams
from quad_residual.sim_lite.env import QuadTrackEnv
from quad_residual.controllers.supervisor import training_support
from quad_residual.utils.metrics import episode_metrics, paired, summarize, table

SUMMARY_KEYS = ["episodes", "crash_rate", "rmse_settled_m_mean", "rmse_settled_m_median",
                "rmse_m_mean", "max_err_m_mean", "final_err_m_mean", "smoothness_sm_mean",
                "cmd_hf_rms_mean", "cmd_rate_p95_mean", "body_rate_mean_mean", "safety_trip_rate",
                "fallback_rate"]


def rollout(conf, args, mode, gains=None, policy=None, support=None):
    """Run ``args.episodes`` single episodes in parallel; return per-episode metrics."""
    runtime = dict(num_envs=args.episodes, policy_mode=mode, seed=args.seed,
                   safety=args.safety, episode_seconds=args.seconds)
    sup = args.supervisor and mode != "baseline"
    if sup:
        runtime.update(supervisor=True, supervisor_support=support,
                       supervisor_gate=args.gate, supervisor_probe_s=args.probe)
    if args.trajectory:
        runtime["trajectory"] = args.trajectory
    if args.nominal:
        runtime["randomize"] = False
    if gains:
        runtime["baseline_gains"] = gains
    env = QuadTrackEnv(C.env_cfg(conf, **runtime))
    env.set_curriculum(0.0 if args.nominal else 1.0)
    obs = env.reset()
    N, T = args.episodes, env.max_steps
    params = dict(env_delay_steps=env.model.delay_steps.copy().astype(float),
                  env_motor_tau=env.model.motor_tau[:, 0].astype(float).copy(),
                  env_wind=np.linalg.norm(env.model.wind, axis=-1).astype(float),
                  env_mass=env.model.mass[:, 0].astype(float).copy())

    err = np.zeros((T, N)); cmd = np.zeros((T, N, 4)); omega = np.zeros((T, N, 3))
    crashed = np.zeros(N, bool)
    tripped = np.zeros(N, bool)
    fell_back = np.zeros(N, bool)
    lag_hat = np.full(N, np.nan)
    for k in range(T):
        alive = ~crashed
        act = (policy.predict(obs, deterministic=True)[0] if policy is not None
               else np.zeros((N, 4), np.float32))
        ref = env._reference()["pos"].copy()
        obs, _, term, trunc, info = env.step(act)
        # Use the state *before* any auto-reset: env.step resets terminated AND
        # truncated envs, so env.state is already a fresh episode for those.
        fs = info["final_state"]
        ok = alive & ~term
        e = np.linalg.norm(fs.pos - ref, axis=-1)
        err[k, ok] = e[ok]
        cmd[k, ok] = env.last_cmd[ok]
        omega[k, ok] = fs.omega[ok]
        tripped |= ok & np.asarray(env.safety.tripped, bool)
        if sup:
            fell_back |= ok & env.sup.tripped.cpu().numpy()
            if k + 1 == env.sup.probe_steps:     # estimate at the gate decision
                lag_hat = np.where(ok, env.sup.lag_hat.cpu().numpy(), np.nan)
        crashed |= alive & term
    ep = episode_metrics(err, cmd, omega, crashed, env.dt, QuadParams().motor_tau,
                         tripped if args.safety else None)
    ep.update(params)
    ep["env_lag"] = params["env_delay_steps"] * QuadParams().dt + params["env_motor_tau"]
    if sup:
        ep["fallback"] = fell_back
        ep["env_lag_hat"] = lag_hat
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345,
                    help="evaluation seed; keep it different from every training seed")
    ap.add_argument("--trajectory", default=None, help="default: from config")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--policy", default=None, help="SB3 .zip checkpoint")
    ap.add_argument("--config-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="NS.KEY=VALUE")
    ap.add_argument("--train-dist", action="store_true",
                    help="evaluate under the checkpoint's training distribution instead of configs/")
    ap.add_argument("--nominal", action="store_true", help="disable randomization")
    ap.add_argument("--safety", action="store_true", help="enable the safety layer")
    ap.add_argument("--supervisor", action="store_true",
                    help="OOD supervisor: fall back to the base controller when the estimated "
                         "actuation lag exceeds the policy's training maximum")
    ap.add_argument("--gate", default="lag", choices=["lag", "support"])
    ap.add_argument("--probe", type=float, default=0.5, help="supervisor identification window (s)")
    ap.add_argument("--out", default=None, help="default: results/eval_<label>.json")
    args = ap.parse_args()

    policy, snap, label = None, None, "residual(zero)"
    if args.policy:
        from stable_baselines3 import PPO
        policy = PPO.load(args.policy, device="cpu")
        snap = C.snapshot_path(args.policy)
        if not os.path.exists(snap):
            raise SystemExit(f"no {snap}: cannot tell what interface the policy was trained with")
        label = os.path.splitext(os.path.basename(args.policy))[0]
    conf = C.load_for_eval(snap, args.config_dir, args.set, args.train_dist)
    ecfg = C.env_cfg(conf)
    mode = ecfg.policy_mode if policy is not None else "residual"
    ref_gains = ecfg.baseline_gains
    # The supervisor's threshold is the largest lag the policy was *trained* on,
    # whatever this evaluation's distribution is.
    train_conf = C.load(snapshot=snap) if snap else C.load(args.config_dir)
    support = training_support(QuadParams(), C.rand_cfg(train_conf))

    eps = {f"baseline_{g}": rollout(conf, args, "baseline", gains=g) for g in ("fast", "robust")}
    eps[label] = rollout(conf, args, mode, policy=policy, support=support)
    ref = f"baseline_{ref_gains}"

    summaries = {k: summarize(v) for k, v in eps.items()}
    pairs = {k: paired(v, eps[ref]) for k, v in eps.items() if k != ref}

    if args.supervisor:
        print(f"supervisor: gate {args.gate}, probe {args.probe} s, training support: "
              f"lag {1000 * support['lag_max']:.0f} ms, delay {support['delay_max']} steps, "
              f"tau {1000 * support['tau_max_train']:.0f} ms")
    print(f"eval: {args.episodes} episodes, seed {args.seed}, safety {'on' if args.safety else 'off'}, "
          f"baseline info '{ecfg.baseline_info}', integral {ecfg.baseline_integral}")
    print(table(summaries, [k for k in SUMMARY_KEYS if any(k in s for s in summaries.values())]))
    print(f"\npaired vs {ref} (residual base), settled RMSE on episodes both survived:")
    for k, p in pairs.items():
        print(f"  {k:22s} n={p['both_survived']:3d}  delta mean {p['delta_mean']:+.4f} m  "
              f"median {p['delta_median']:+.4f} m  win {100 * p['win_rate']:.0f}%  "
              f"(crashes: only ref {p['only_ref_crashed']}, only this {p['only_this_crashed']})")
    if policy is None:
        same = np.allclose(np.nan_to_num(eps[label]["rmse_m"]), np.nan_to_num(eps[ref]["rmse_m"]))
        print(f"\nsanity: residual(zero) == {ref}: {'OK' if same else 'MISMATCH'}")

    out = args.out or os.path.join("results", f"eval_{label.replace('(', '_').replace(')', '')}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = {
        "meta": {"policy": args.policy, "label": label, "mode": mode, "reference": ref,
                 "episodes": args.episodes, "seed": args.seed, "safety": args.safety,
                 "supervisor": args.supervisor and {"gate": args.gate, "probe_s": args.probe, **support},
                 "train_dist": args.train_dist, "overrides": args.set,
                 "trajectory": args.trajectory},
        "summary": summaries,
        "paired": pairs,
        "episodes": {k: {m: np.asarray(v).tolist() for m, v in ep.items()} for k, ep in eps.items()},
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
