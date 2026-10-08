#!/usr/bin/env python3
"""Validate the Isaac env's torch core against sim_lite -- no Isaac Sim needed.

Runs ``envs.torch_core.TorchQuadCore`` (everything the Isaac env does except
PhysX) with the same semi-implicit Euler integrator as sim_lite standing in
for PhysX, and compares crash rate and settled RMSE with sim_lite under the
same configuration. Random streams differ between the two (torch vs NumPy
generators), so the comparison is statistical: with 400 episodes the
baselines should agree within a few percent and a sim_lite-trained policy
should transfer without loss.

    python scripts/check_torch_core.py
    python scripts/check_torch_core.py --policy results/direct_s0.zip
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np
import torch

from quad_residual import config as C
from quad_residual.envs.torch_core import TorchQuadCore, rigid_body_step
from quad_residual.sim_lite.env import QuadTrackEnv
from quad_residual.utils.metrics import episode_metrics, summarize


def run_torch(conf, n, seed, mode, gains, policy=None):
    ecfg = C.env_cfg(conf, num_envs=n, policy_mode=mode, seed=seed, baseline_gains=gains)
    core = TorchQuadCore(ecfg, n, "cpu", C.rand_cfg(conf), seed=seed)
    core.set_curriculum(1.0)
    ids = torch.arange(n)
    pos, quat, vel, om = core.reset(ids)
    from types import SimpleNamespace as NS
    st = NS(pos=pos, vel=vel, quat=quat, omega=om)
    obs = core.observe(st)
    T = int(ecfg.episode_seconds / core.dt_ctrl)
    err = np.zeros((T, n)); cmd = np.zeros((T, n, 4)); omg = np.zeros((T, n, 3))
    crashed = np.zeros(n, bool)
    for k in range(T):
        if policy is not None:
            act = torch.as_tensor(policy.predict(obs.numpy(), deterministic=True)[0])
        else:
            act = torch.zeros((n, 4))
        core.control(act, st)
        for _ in range(ecfg.decimation):
            f, tq = core.wrench(st)
            st = rigid_body_step(st, f, tq, core.rand.mass, core.rand.inertia, core.p.dt, core.p.gravity)
        term = core.terminated(st).numpy()
        ok = ~crashed & ~term
        e = (st.pos - core.ref["pos"]).norm(dim=-1).numpy()
        err[k, ok] = e[ok]; cmd[k, ok] = core.cmd.numpy()[ok]; omg[k, ok] = st.omega.numpy()[ok]
        crashed |= term
        obs = core.observe(st)
    return summarize(episode_metrics(err, cmd, omg, crashed, core.dt_ctrl, core.p.motor_tau))


def run_numpy(conf, n, seed, mode, gains, policy=None):
    env = QuadTrackEnv(C.env_cfg(conf, num_envs=n, policy_mode=mode, seed=seed, baseline_gains=gains))
    env.set_curriculum(1.0)
    obs = env.reset()
    T = env.max_steps
    err = np.zeros((T, n)); cmd = np.zeros((T, n, 4)); omg = np.zeros((T, n, 3))
    crashed = np.zeros(n, bool)
    for k in range(T):
        act = policy.predict(obs, deterministic=True)[0] if policy is not None else np.zeros((n, 4), np.float32)
        ref = env._reference()["pos"].copy()
        obs, _, term, _, info = env.step(act)
        fs = info["final_state"]                 # pre-reset state
        ok = ~crashed & ~term
        e = np.linalg.norm(fs.pos - ref, axis=-1)
        err[k, ok] = e[ok]; cmd[k, ok] = env.last_cmd[ok]; omg[k, ok] = fs.omega[ok]
        crashed |= term
    return summarize(episode_metrics(err, cmd, omg, crashed, env.dt, env.p.motor_tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--policy", default=None, help="SB3 checkpoint trained in sim_lite")
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    rows = []
    for gains in ("fast", "robust"):
        conf = C.load()
        rows.append((f"baseline_{gains}", run_numpy(conf, args.episodes, args.seed, "baseline", gains),
                     run_torch(conf, args.episodes, args.seed, "baseline", gains)))
    if args.policy:
        from stable_baselines3 import PPO
        pol = PPO.load(args.policy, device="cpu")
        conf = C.load_for_eval(C.snapshot_path(args.policy))
        mode = conf["env"]["policy_mode"]
        rows.append((os.path.basename(args.policy),
                     run_numpy(conf, args.episodes, args.seed, mode, conf["env"]["baseline"]["baseline_gains"], pol),
                     run_torch(conf, args.episodes, args.seed, mode, conf["env"]["baseline"]["baseline_gains"], pol)))

    print(f"{args.episodes} episodes each, full randomization; sim_lite (NumPy) vs torch core\n")
    print(f"{'controller':22s} {'crash numpy':>12s} {'crash torch':>12s} {'RMSE numpy':>11s} {'RMSE torch':>11s} {'HF numpy':>9s} {'HF torch':>9s}")
    for name, a, b in rows:
        print(f"{name:22s} {a['crash_rate']:12.3f} {b['crash_rate']:12.3f} "
              f"{a.get('rmse_settled_m_mean', float('nan')):11.3f} {b.get('rmse_settled_m_mean', float('nan')):11.3f} "
              f"{a.get('cmd_hf_rms_mean', float('nan')):9.4f} {b.get('cmd_hf_rms_mean', float('nan')):9.4f}")


if __name__ == "__main__":
    main()
