#!/usr/bin/env python3
"""Measure policy inference latency and export a deployable artifact.

The control loop runs at 50 Hz, so the policy has a 20 ms budget and in
practice should use a small fraction of it. This reports the distribution, not
just the mean -- a p99 that breaches the period is a dropped control cycle,
which the mean will happily hide.

    python scripts/bench_latency.py --policy results/residual_simlite.zip
"""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--obs-dim", type=int, default=20)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--export", default=None, help="TorchScript output path")
    args = ap.parse_args()

    budget_ms = 1000.0 / args.control_hz

    if args.policy:
        from stable_baselines3 import PPO
        model = PPO.load(args.policy, device="cpu")
        net = model.policy
        import torch
        obs = torch.zeros(1, args.obs_dim)

        def infer():
            with torch.no_grad():
                net(obs, deterministic=True)

        if args.export:
            scripted = torch.jit.trace(
                lambda x: net(x, deterministic=True)[0], obs)
            scripted.save(args.export)
            print(f"exported TorchScript -> {args.export}")
    else:
        rng = np.random.default_rng(0)
        W1 = rng.normal(0, .1, (args.obs_dim, 128)); b1 = np.zeros(128)
        W2 = rng.normal(0, .1, (128, 128)); b2 = np.zeros(128)
        W3 = rng.normal(0, .1, (128, 4)); b3 = np.zeros(4)
        obs = np.zeros((1, args.obs_dim))
        print("no --policy given: benchmarking an equivalent NumPy MLP (128,128)")

        def infer():
            h = np.tanh(obs @ W1 + b1)
            h = np.tanh(h @ W2 + b2)
            return h @ W3 + b3

    for _ in range(200):
        infer()
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        infer()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    p = lambda q: times[int(q * len(times)) - 1]
    print(f"control period      {budget_ms:8.2f} ms")
    print(f"mean                {statistics.mean(times):8.3f} ms")
    print(f"p50                 {p(0.50):8.3f} ms")
    print(f"p99                 {p(0.99):8.3f} ms")
    print(f"max                 {times[-1]:8.3f} ms")
    print(f"p99 / budget        {100 * p(0.99) / budget_ms:8.1f} %")


if __name__ == "__main__":
    main()
