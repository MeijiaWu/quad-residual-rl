#!/usr/bin/env python3
"""PPO training on the CPU env -- for reward shaping and sanity, not final results.

    pip install stable-baselines3 gymnasium pyyaml
    python scripts/train_sim_lite.py --mode residual --steps 300000
    python scripts/train_sim_lite.py --set domain_rand.latency.extra_delay_steps=[0,0]

All settings come from configs/*.yaml (see quad_residual/config.py); CLI flags
and --set overrides win. The resolved config is saved next to the checkpoint
as <out>.cfg.yaml, and eval.py picks it up automatically.

Final training belongs on Isaac Lab with thousands of parallel envs; this path
exists so reward and observation design can be iterated without a GPU.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np
from quad_residual import config as C
from quad_residual.sim_lite.env import QuadTrackEnv


def make_adapter(cfg, total_steps, curriculum_fraction):
    """Build a real SB3 VecEnv over QuadTrackEnv (SB3 type-checks for VecEnv)."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import VecEnv

    class SB3Adapter(VecEnv):
        def __init__(self):
            self.env = QuadTrackEnv(cfg)
            self.ramp = curriculum_fraction * total_steps   # 0 -> no curriculum
            self.seen = 0
            # Start where the curriculum starts; SB3's first reset() then draws
            # every env's parameters at this scale (not at the default full scale).
            self.env.set_curriculum(0.0 if self.ramp > 0 else 1.0)
            obs_space = gym.spaces.Box(-np.inf, np.inf, (self.env.obs_dim,), np.float32)
            act_space = gym.spaces.Box(-1.0, 1.0, (self.env.act_dim,), np.float32)
            super().__init__(cfg.num_envs, obs_space, act_space)
            self._actions = None

        def reset(self):
            return self.env.reset()

        def step_async(self, actions):
            self._actions = actions

        def step_wait(self):
            obs, rew, term, trunc, info = self.env.step(self._actions)
            self.seen += self.num_envs
            if self.ramp > 0:
                self.env.set_curriculum(min(1.0, self.seen / self.ramp))
            done = term | trunc
            infos = [{} for _ in range(self.num_envs)]
            for i in np.nonzero(done)[0]:
                # SB3 bootstraps V(s_T) for time-limit truncation from these two keys.
                infos[i]["terminal_observation"] = info["final_obs"][i]
                infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
            return obs, rew, done, infos

        def close(self):
            pass

        # --- VecEnv abstract API; not used by PPO on this env ---
        def get_attr(self, attr_name, indices=None):
            return [getattr(self.env, attr_name, None)] * len(self._get_indices(indices))

        def set_attr(self, attr_name, value, indices=None):
            setattr(self.env, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            out = getattr(self.env, method_name)(*args, **kwargs)
            return [out] * len(self._get_indices(indices))

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._get_indices(indices))

    return SB3Adapter()


ACTIVATIONS = {"tanh": "Tanh", "elu": "ELU", "relu": "ReLU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None, choices=["residual", "direct"],
                    help="default: env.policy_mode from configs/env.yaml")
    ap.add_argument("--steps", type=int, default=None, help="default: ppo.sim_lite.total_steps")
    ap.add_argument("--num-envs", type=int, default=None, help="default: ppo.sim_lite.num_envs")
    ap.add_argument("--trajectory", default=None, help="default: env.trajectory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-dir", default=None, help="default: <repo>/configs")
    ap.add_argument("--set", action="append", default=[], metavar="NS.KEY=VALUE",
                    help="override a config value; repeatable")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    cfg = C.load(args.config_dir, args.set)
    # Fold CLI flags into the config so the saved snapshot is the whole truth.
    if args.mode:
        cfg["env"]["policy_mode"] = args.mode
    if args.trajectory:
        cfg["env"]["trajectory"] = args.trajectory
    if args.steps:
        cfg["ppo"]["sim_lite"]["total_steps"] = args.steps
    if args.num_envs:
        cfg["ppo"]["sim_lite"]["num_envs"] = args.num_envs
    cfg["meta"] = {"seed": args.seed}
    if cfg["env"]["policy_mode"] not in ("residual", "direct"):
        raise SystemExit(f"cannot train policy_mode={cfg['env']['policy_mode']!r}")

    hp = C.ppo_cfg(cfg)
    num_envs = hp.pop("num_envs")
    total = hp.pop("total_steps")
    frac = hp.pop("curriculum_fraction")
    std = hp.pop("init_noise_std")
    policy_kwargs = dict(
        net_arch=hp.pop("net_arch"),
        activation_fn=getattr(torch.nn, ACTIVATIONS[hp.pop("activation")]),
        log_std_init=float(np.log(std)),
    )

    env_cfg = C.env_cfg(cfg, num_envs=num_envs, seed=args.seed)
    venv = VecMonitor(make_adapter(env_cfg, total, frac))   # VecMonitor -> ep_rew_mean in logs

    model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, verbose=1,
                seed=args.seed, **hp)
    model.learn(total_timesteps=total)

    out = args.out or f"results/{env_cfg.policy_mode}_simlite"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model.save(out)
    C.save(cfg, C.snapshot_path(out))
    print(f"saved {out}.zip and {C.snapshot_path(out)}")


if __name__ == "__main__":
    main()
