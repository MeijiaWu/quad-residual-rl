#!/usr/bin/env python3
"""PPO (rsl_rl) on the Isaac Lab env -- thousands of parallel envs on the GPU.

Run from the Isaac Lab Python environment (Isaac Sim 5.1 + Isaac Lab 2.3):

    python scripts/train_isaac.py --mode direct --num_envs 4096 --headless
    python scripts/train_isaac.py --mode residual --seed 3 --headless
    python scripts/train_isaac.py --mode direct --set domain_rand.latency.extra_delay_steps=[0,0] --headless

Hyper-parameters come from the ``isaac`` block of configs/ppo.yaml (CLI flags
win). Logs and checkpoints (model_<iter>.pt) go to results/isaac/<run>/;
the resolved env / agent configs are written next to them.
"""
import argparse, os, sys
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))

from isaaclab.app import AppLauncher  # noqa: E402  (must come before other isaaclab imports)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="direct", choices=["direct", "residual"])
ap.add_argument("--num_envs", type=int, default=None)
ap.add_argument("--max_iterations", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--trajectory", default="lemniscate")
ap.add_argument("--set", action="append", default=[], metavar="domain_rand.KEY=VALUE",
                help="randomization override, same syntax as sim_lite")
ap.add_argument("--run_name", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

# ---- everything below needs the running simulation app --------------------
import importlib.metadata as md  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from quad_residual.envs.quad_track_env import QuadTrackEnv  # noqa: E402
from quad_residual.envs.quad_track_env_cfg import QuadTrackEnvCfg  # noqa: E402

with open(os.path.join(ROOT, "configs", "ppo.yaml")) as fh:
    H = yaml.safe_load(fh)["isaac"]


@configclass
class QuadPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = H["num_steps_per_env"]
    max_iterations = H["max_iterations"]
    save_interval = 100
    experiment_name = "quad_track"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=H["policy"]["init_noise_std"],
        actor_hidden_dims=H["policy"]["actor_hidden_dims"],
        critic_hidden_dims=H["policy"]["critic_hidden_dims"],
        activation=H["policy"]["activation"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=H["ppo"]["value_loss_coef"], use_clipped_value_loss=True,
        clip_param=H["ppo"]["clip_param"], entropy_coef=H["ppo"]["entropy_coef"],
        num_learning_epochs=H["ppo"]["num_learning_epochs"],
        num_mini_batches=H["ppo"]["num_mini_batches"],
        learning_rate=H["ppo"]["learning_rate"], schedule=H["ppo"]["schedule"],
        gamma=H["ppo"]["gamma"], lam=H["ppo"]["lam"], desired_kl=H["ppo"]["desired_kl"],
        max_grad_norm=H["ppo"]["max_grad_norm"],
    )


def main():
    env_cfg = QuadTrackEnvCfg()
    env_cfg.scene.num_envs = args.num_envs or H["num_envs"]
    env_cfg.policy_mode = args.mode
    env_cfg.trajectory = args.trajectory
    env_cfg.seed = args.seed
    env_cfg.config_overrides = list(args.set)
    env_cfg.sim.device = args.device if args.device else "cuda:0"

    agent = QuadPPORunnerCfg()
    agent.seed = args.seed
    agent.device = env_cfg.sim.device
    if args.max_iterations:
        agent.max_iterations = args.max_iterations
    # Curriculum over the first 60 % of training, as in sim_lite.
    env_cfg.curriculum_ramp_steps = int(0.6 * agent.max_iterations * agent.num_steps_per_env)
    try:  # Isaac Lab maps cfg fields that changed between rsl-rl-lib versions
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent = handle_deprecated_rsl_rl_cfg(agent, md.version("rsl-rl-lib"))
    except ImportError:
        pass

    run = args.run_name or f"{args.mode}_s{args.seed}_{datetime.now():%Y%m%d-%H%M%S}"
    log_dir = os.path.join(ROOT, "results", "isaac", run)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent)

    torch.manual_seed(args.seed)
    env = QuadTrackEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    print(f"checkpoints in {log_dir}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
