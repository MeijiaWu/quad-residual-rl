"""Load ``configs/*.yaml`` into the dataclasses the code actually runs on.

Three files, three namespaces:

    env          configs/env.yaml          -> sim_lite.env.EnvCfg
    domain_rand  configs/domain_rand.yaml  -> dynamics.randomization.RandomizationCfg
    ppo          configs/ppo.yaml          -> PPO kwargs (``sim_lite`` block)

Overrides use dotted paths rooted at a namespace, values parsed as YAML::

    --set env.reward.w_smooth=0.0
    --set domain_rand.latency.extra_delay_steps=[0,0]
    --set ppo.sim_lite.curriculum_fraction=0

Unknown keys -- in a file or in an override -- raise. A config that is silently
ignored is worse than no config at all, which is the bug this module replaces.

Training writes the fully resolved config next to the checkpoint
(``<model>.cfg.yaml``); evaluation loads it so a policy is always evaluated in
the environment it was trained for (same mode, residual scale, observation).
"""

from __future__ import annotations

import copy
import os
from dataclasses import fields

import yaml

from .dynamics.randomization import RandomizationCfg
from .sim_lite.env import EnvCfg

NAMESPACES = ("env", "domain_rand", "ppo")
DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

# Keys present in the yaml for the Isaac Lab path only; accepted, not used here.
_ISAAC_ONLY = {"env": {"num_envs"}, "domain_rand": {"curriculum_steps"}}


def load(config_dir: str | None = None, overrides=(), snapshot: str | None = None) -> dict:
    """Return ``{"env": ..., "domain_rand": ..., "ppo": ...}`` as plain dicts.

    ``snapshot`` (a ``.cfg.yaml`` written by training) replaces the three files.
    ``overrides`` are applied last either way.
    """
    if snapshot:
        with open(snapshot) as fh:
            cfg = yaml.safe_load(fh)
    else:
        d = config_dir or DEFAULT_DIR
        cfg = {}
        for ns in NAMESPACES:
            with open(os.path.join(d, f"{ns}.yaml")) as fh:
                cfg[ns] = yaml.safe_load(fh) or {}
    for item in overrides:
        _apply_override(cfg, item)
    return cfg


def load_for_eval(snapshot: str | None, config_dir: str | None = None, overrides=(),
                  train_dist: bool = False) -> dict:
    """Config for evaluating a checkpoint.

    The ``env`` namespace (policy mode, residual scale, observation, baseline
    the residual sits on) comes from the checkpoint's snapshot -- that is the
    policy's interface and must match training. The *evaluation conditions*
    (``domain_rand``) come from ``configs/`` so that every checkpoint, including
    ablations trained under a different distribution, is tested under the same
    conditions. ``train_dist=True`` uses the snapshot's distribution instead.
    """
    base = load(config_dir)
    if snapshot is None:
        cfg = base
    else:
        with open(snapshot) as fh:
            snap = yaml.safe_load(fh)
        if "baseline" not in snap.get("env", {}):
            raise SystemExit(
                f"{snapshot} predates the baseline change (nominal info, robust gains, "
                "integral). Its residual was trained on a different base controller "
                "-- retrain it.")
        cfg = {"env": snap["env"], "ppo": snap.get("ppo", base["ppo"]),
               "domain_rand": snap["domain_rand"] if train_dist else base["domain_rand"]}
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        _apply_override(cfg, item)
    return cfg


def _apply_override(cfg: dict, item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like ns.key=value")
    path, raw = item.split("=", 1)
    keys = path.strip().split(".")
    if keys[0] not in NAMESPACES or len(keys) < 2:
        raise KeyError(f"override {item!r}: path must start with one of {NAMESPACES}")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"override {item!r}: no key {k!r}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"override {item!r}: no key {keys[-1]!r} "
                       f"(have {sorted(node)})")
    node[keys[-1]] = yaml.safe_load(raw)


def _flatten(section: dict, ns: str) -> dict:
    """Merge one level of nested sections (reward:, latency:, ...) into one dict."""
    flat = {}
    for k, v in section.items():
        items = v.items() if isinstance(v, dict) else [(k, v)]
        for kk, vv in items:
            if kk in flat:
                raise KeyError(f"{ns}: key {kk!r} appears twice")
            flat[kk] = vv
    return flat


def _fill(cls, values: dict, ns: str, ignore=frozenset()):
    names = {f.name: f for f in fields(cls)}
    kw = {}
    for k, v in values.items():
        if k in ignore:
            continue
        if k not in names:
            raise KeyError(f"{ns}: unknown key {k!r} for {cls.__name__}")
        default = names[k].default
        kw[k] = tuple(v) if isinstance(default, tuple) else v
    return cls(**kw)


def rand_cfg(cfg: dict) -> RandomizationCfg:
    flat = _flatten(cfg["domain_rand"], "domain_rand")
    return _fill(RandomizationCfg, flat, "domain_rand",
                 ignore=_ISAAC_ONLY["domain_rand"])


def env_cfg(cfg: dict, **runtime) -> EnvCfg:
    """Build ``EnvCfg``; ``runtime`` sets fields the yaml does not own
    (num_envs, seed, safety, episode_seconds for evaluation, ...)."""
    flat = _flatten(cfg["env"], "env")
    flat = {k: v for k, v in flat.items() if k not in _ISAAC_ONLY["env"]}
    flat.update(runtime)
    rc = rand_cfg(cfg)
    flat.setdefault("randomize", rc.enabled)
    flat["rand_cfg"] = rc
    return _fill(EnvCfg, flat, "env")


def ppo_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg["ppo"]["sim_lite"])


def save(cfg: dict, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def snapshot_path(model_path: str) -> str:
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return base + ".cfg.yaml"
