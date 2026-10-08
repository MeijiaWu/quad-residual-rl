#!/usr/bin/env python3
"""Evaluate controllers in the Isaac Lab env, in the same format as eval.py.

    # an rsl_rl policy trained in Isaac
    python scripts/eval_isaac.py --rsl results/isaac/direct_s0_.../model_499.pt --mode direct
    # sim-to-sim: a policy trained in sim_lite (SB3), flown in PhysX
    python scripts/eval_isaac.py --sb3 results/direct_s0.zip
    # with the OOD supervisor (fall back to the base controller when the
    # estimated actuation lag leaves the policy's training distribution)
    python scripts/eval_isaac.py --rsl ... --mode residual --supervisor

Both classical baselines are always evaluated as well. Each controller runs in
its own process (one Isaac Sim stage per process) on the same seed, i.e. the
same sampled airframes, spawn states and waypoints. One episode per env; a
crash ends the episode's statistics. Output: results/isaac_eval/<label>.json,
readable by scripts/aggregate.py.

The two baselines do not depend on the policy, so their rollouts are cached
per (seed, episodes, trajectory, --set) in results/isaac_eval/_cache/
(``--no-cache`` to recompute, e.g. after changing the controller).
"""
import argparse, hashlib, json, os, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--trajectory", default="lemniscate")
    ap.add_argument("--mode", default=None, help="policy_mode for --rsl (direct|residual)")
    ap.add_argument("--rsl", default=None, help="rsl_rl checkpoint (model_*.pt)")
    ap.add_argument("--sb3", default=None, help="SB3 checkpoint trained in sim_lite")
    ap.add_argument("--set", action="append", default=[], metavar="domain_rand.KEY=VALUE")
    ap.add_argument("--out", default=None)
    ap.add_argument("--supervisor", action="store_true",
                    help="OOD supervisor on the policy (see controllers/supervisor.py)")
    ap.add_argument("--gate", default="lag", choices=["lag", "support"])
    ap.add_argument("--probe", type=float, default=0.5, help="supervisor identification window (s)")
    ap.add_argument("--no-cache", action="store_true", help="re-run the baselines")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_npz", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_support", default=None, help=argparse.SUPPRESS)
    return ap


class _Lenient(__import__("yaml").SafeLoader):
    """Reads Isaac Lab's dumped env.yaml without importing Isaac Lab: python
    object tags become None (only plain fields are needed here)."""


_Lenient.add_multi_constructor("", lambda loader, suffix, node: None)


def training_support_of(args):
    """Delay / tau / lag maxima of the distribution the policy was trained on."""
    import yaml
    from quad_residual import config as C
    from quad_residual.controllers.supervisor import training_support
    from quad_residual.dynamics.quadrotor import QuadParams
    if args.sb3:
        conf = C.load(snapshot=C.snapshot_path(args.sb3))
    else:
        env_yaml = os.path.join(os.path.dirname(args.rsl), "params", "env.yaml")
        with open(env_yaml) as fh:
            env = yaml.load(fh, Loader=_Lenient) or {}
        conf = C.load(overrides=list(env.get("config_overrides") or []))
    return training_support(QuadParams(), C.rand_cfg(conf))


def load_rsl_actor(ckpt, device):
    """Deterministic actor (the mean) from an rsl_rl checkpoint, rebuilt as a
    plain MLP -- avoids depending on the rsl_rl runner API, which changes
    between rsl-rl-lib versions. Training runs without observation
    normalization (empirical_normalization = False), so none is applied."""
    import torch
    import yaml
    sd = torch.load(ckpt, map_location=device)["model_state_dict"]
    with open(os.path.join(os.path.dirname(ckpt), "params", "agent.yaml")) as fh:
        agent = yaml.unsafe_load(fh)
    act = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[agent["policy"]["activation"]]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idx):
        w, b = sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
        lin.weight.data.copy_(w); lin.bias.data.copy_(b)
        layers.append(lin)
        if j < len(idx) - 1:
            layers.append(act())
    mlp = torch.nn.Sequential(*layers).eval()
    return lambda o: mlp(o)


# ------------------------------------------------------------------ worker
def worker(args, extra):
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app_args = ap.parse_args(extra)
    app_args.headless = True
    app = AppLauncher(app_args).app

    import numpy as np
    import torch
    from quad_residual.envs.quad_track_env import QuadTrackEnv
    from quad_residual.envs.quad_track_env_cfg import QuadTrackEnvCfg

    kind, _, spec = args._worker.partition(":")
    cfg = QuadTrackEnvCfg()
    cfg.scene.num_envs = args.episodes
    cfg.seed = args.seed
    cfg.trajectory = args.trajectory
    cfg.config_overrides = list(args.set)
    cfg.curriculum_ramp_steps = 0                    # full randomization
    cfg.sim.device = app_args.device or "cuda:0"
    if args._support:                                # only ever set for the policy job
        cfg.supervisor = True
        cfg.supervisor_gate = args.gate
        cfg.supervisor_probe_s = args.probe
        cfg.supervisor_support = json.loads(args._support)
    policy = None
    if kind == "baseline":
        cfg.policy_mode, cfg.baseline_gains = "baseline", spec
    elif kind == "sb3":
        from quad_residual import config as C
        snap = C.load_for_eval(C.snapshot_path(spec))
        cfg.policy_mode = snap["env"]["policy_mode"]
        cfg.residual_scale = snap["env"]["residual_scale"]
        for k, v in snap["env"]["baseline"].items():
            setattr(cfg, k, v)
    else:
        cfg.policy_mode = args.mode or "direct"

    env = QuadTrackEnv(cfg=cfg)
    if kind == "sb3":
        from stable_baselines3 import PPO
        pol = PPO.load(spec, device="cpu")
        policy = lambda o: torch.as_tensor(pol.predict(o.cpu().numpy(), deterministic=True)[0],
                                           device=env.device)
    elif kind == "rsl":
        policy = load_rsl_actor(spec, env.device)

    obs, _ = env.reset()
    obs = obs["policy"]
    N, T = args.episodes, int(env.max_episode_length)
    err = np.zeros((T, N)); cmd = np.zeros((T, N, 4)); om = np.zeros((T, N, 3))
    crashed = np.zeros(N, bool)
    fell_back = np.zeros(N, bool)
    lag_hat = np.full(N, np.nan)
    core = env.core
    params = dict(env_delay_steps=core.rand.delay_steps.cpu().numpy().astype(float),
                  env_motor_tau=core.rand.motor_tau[:, 0].cpu().numpy().astype(float),
                  env_wind=core.rand.wind_mean.norm(dim=-1).cpu().numpy().astype(float),
                  env_mass=core.rand.mass[:, 0].cpu().numpy().astype(float))
    with torch.inference_mode():
        for k in range(T):
            act = policy(obs) if policy is not None else torch.zeros((N, 4), device=env.device)
            obs, _, term, _, _ = env.step(act)
            obs = obs["policy"]
            term = term.cpu().numpy().astype(bool)
            ok = ~crashed & ~term
            err[k, ok] = env.final_err.cpu().numpy()[ok]
            cmd[k, ok] = env.final_cmd.cpu().numpy()[ok]
            om[k, ok] = env.final_omega.cpu().numpy()[ok]
            if core.sup is not None:
                fell_back |= ok & core.sup.tripped.cpu().numpy()
                if k + 1 == core.sup.probe_steps:          # estimate at the gate decision
                    lag_hat = np.where(ok, core.sup.lag_hat.cpu().numpy(), np.nan)
            crashed |= term
    extra = dict(fallback=fell_back, env_lag_hat=lag_hat) if core.sup is not None else {}
    np.savez(args._npz, err=err, cmd=cmd, omega=om, crashed=crashed,
             dt=core.dt_ctrl, tau=core.p.motor_tau, **params, **extra)
    env.close()
    app.close()


# ------------------------------------------------------------------ driver
def main():
    ap = parse()
    args, extra = ap.parse_known_args()
    if args._worker:
        return worker(args, extra)

    import numpy as np
    from quad_residual.utils.metrics import episode_metrics, paired, summarize, table

    jobs = [("baseline_fast", "baseline:fast"), ("baseline_robust", "baseline:robust")]
    label = None
    if args.rsl:
        label = "isaac_" + os.path.basename(os.path.dirname(args.rsl))
        jobs.append((label, f"rsl:{os.path.abspath(args.rsl)}"))
    if args.sb3:
        label = "sim2sim_" + os.path.splitext(os.path.basename(args.sb3))[0]
        jobs.append((label, f"sb3:{os.path.abspath(args.sb3)}"))
    out_dir = os.path.join(ROOT, "results", "isaac_eval")
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(json.dumps([args.seed, args.episodes, args.trajectory, sorted(args.set)])
                       .encode()).hexdigest()[:12]
    support = training_support_of(args) if (args.supervisor and (args.rsl or args.sb3)) else None

    from quad_residual.dynamics.quadrotor import QuadParams
    eps = {}
    for name, spec in jobs:
        is_base = spec.startswith("baseline:")
        tmp = os.path.join(cache_dir, f"_tmp_{os.getpid()}_{name}.npz")
        npz = os.path.join(cache_dir, f"{name}_{key}.npz") if is_base else tmp
        if not (is_base and os.path.exists(npz) and not args.no_cache):
            cmd = [sys.executable, __file__, "--_worker", spec, "--_npz", tmp,
                   "--episodes", str(args.episodes), "--seed", str(args.seed),
                   "--trajectory", args.trajectory] + sum([["--set", s] for s in args.set], [])
            if args.mode:
                cmd += ["--mode", args.mode]
            if support is not None and not is_base:
                cmd += ["--_support", json.dumps(support), "--gate", args.gate,
                        "--probe", str(args.probe)]
            print("  $", " ".join(cmd), flush=True)
            subprocess.run(cmd + extra, check=True)
            if is_base:
                os.replace(tmp, npz)          # atomic: parallel evaluations may share the cache
        else:
            print(f"  {name}: cached ({os.path.relpath(npz, ROOT)})", flush=True)
        d = np.load(npz)
        ep = episode_metrics(d["err"], d["cmd"], d["omega"], d["crashed"], float(d["dt"]), float(d["tau"]))
        ep.update({k: d[k] for k in ("env_delay_steps", "env_motor_tau", "env_wind", "env_mass")})
        ep["env_lag"] = d["env_delay_steps"] * QuadParams().dt + d["env_motor_tau"]
        for k in ("fallback", "env_lag_hat"):
            if k in d.files:
                ep[k] = d[k]
        eps[name] = ep
        if not is_base:
            d.close()
            os.remove(npz)

    ref = "baseline_robust"
    summaries = {k: summarize(v) for k, v in eps.items()}
    pairs = {k: paired(v, eps[ref]) for k, v in eps.items() if k != ref}
    print(table(summaries, ["episodes", "crash_rate", "rmse_settled_m_mean", "rmse_settled_m_median",
                            "cmd_hf_rms_mean", "body_rate_mean_mean", "fallback_rate"]))
    if support is not None:
        print(f"  supervisor: gate {args.gate}, probe {args.probe} s, training support lag "
              f"{1000 * support['lag_max']:.0f} ms / delay {support['delay_max']} / "
              f"tau {1000 * support['tau_max_train']:.0f} ms")
    for k, p in pairs.items():
        print(f"  {k:28s} vs robust: n={p['both_survived']} delta mean {p['delta_mean']:+.4f} m "
              f"win {100 * p['win_rate']:.0f}%")
    out = args.out or os.path.join(out_dir, f"{label or 'baselines'}.json")
    payload = {"meta": {"policy": args.rsl or args.sb3, "label": label, "mode": args.mode,
                        "reference": ref, "episodes": args.episodes, "seed": args.seed,
                        "safety": False, "train_dist": False, "overrides": args.set,
                        "trajectory": args.trajectory, "simulator": "isaac",
                        "supervisor": ({"gate": args.gate, "probe_s": args.probe, **support}
                                       if support is not None else False)},
               "summary": summaries, "paired": pairs,
               "episodes": {k: {m: np.asarray(v).tolist() for m, v in ep.items()} for k, ep in eps.items()}}
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
