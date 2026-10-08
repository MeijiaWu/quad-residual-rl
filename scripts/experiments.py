#!/usr/bin/env python3
"""Run the main comparison, the ablations and the generalization tests.

    python scripts/experiments.py list
    python scripts/experiments.py run main_residual main_direct      # train (if missing) + eval
    python scripts/experiments.py run ablations --seeds 0-9 --jobs 2
    python scripts/experiments.py run generalization                 # eval-only, reuses main checkpoints
    python scripts/experiments.py report                             # results/report.md

Every experiment = one change from the main setup. Training is skipped when the
checkpoint already exists, so runs can be resumed or extended with more seeds;
evaluation is cheap and always re-run. All evaluations use the same episodes
(eval seed 12345, 100 episodes) and the evaluation distribution in
configs/domain_rand.yaml -- an ablation trained under a different distribution
is still *tested* under the standard one, which is what makes it an ablation.

Checkpoints: results/<ckpt>_s<seed>.zip   Evaluations: results/eval/<experiment>/<label>.json
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from aggregate import aggregate  # noqa: E402

# name: dict(mode, ckpt, train=[--set ...], eval=[extra eval args], from_=checkpoints of, ref, note)
EXPERIMENTS = {
    # ---- main comparison -------------------------------------------------------
    "main_residual": dict(mode="residual", ckpt="residual", group="main",
                          note="residual RL on the robust base"),
    "main_direct": dict(mode="direct", ckpt="direct", group="main",
                        note="direct RL (rotor commands)"),
    # ---- ablations: one training change each, evaluated under standard conditions
    "abl_residual_fastbase": dict(
        mode="residual", ckpt="abl_residual_fastbase", group="ablations",
        train=["env.baseline.baseline_gains=fast"],
        note="residual on the fast (fragile) base -- does a slow base limit the residual?"),
    "abl_residual_scale05": dict(
        mode="residual", ckpt="abl_residual_scale05", group="ablations",
        train=["env.residual_scale=0.5"],
        note="residual authority 0.25 -> 0.5"),
    "abl_direct_nodelay": dict(
        mode="direct", ckpt="abl_direct_nodelay", group="ablations",
        train=["domain_rand.latency.extra_delay_steps=[0,0]"],
        note="no delay randomization in training (tested with it)"),
    "abl_direct_norand": dict(
        mode="direct", ckpt="abl_direct_norand", group="ablations",
        train=["domain_rand.enabled=false"],
        note="no domain randomization at all in training"),
    "abl_direct_nosmooth": dict(
        mode="direct", ckpt="abl_direct_nosmooth", group="ablations",
        train=["env.reward.w_smooth=0.0"],
        note="no action-smoothness penalty"),
    "abl_direct_nocurriculum": dict(
        mode="direct", ckpt="abl_direct_nocurriculum", group="ablations",
        train=["ppo.sim_lite.curriculum_fraction=0"],
        note="full randomization from the first step"),
    # ---- generalization / deployment: eval-only on the main checkpoints ---------
    "wp_residual": dict(from_="main_residual", group="generalization",
                        eval=["--trajectory", "waypoints"],
                        note="unseen task: random waypoint steps (trained on lemniscate only)"),
    "wp_direct": dict(from_="main_direct", group="generalization",
                      eval=["--trajectory", "waypoints"], note="same, direct RL"),
    "safety_residual": dict(from_="main_residual", group="generalization",
                            eval=["--safety"], note="deployment safety layer on"),
    "safety_direct": dict(from_="main_direct", group="generalization",
                          eval=["--safety"], note="deployment safety layer on"),
    # ---- out-of-distribution stress: actuation delay beyond the training range ---
    # training draws extra delay 0-3 physics steps; this tests 4-8 (total 20-36 ms)
    "stress_residual": dict(from_="main_residual", group="stress",
                            eval=["--set", "domain_rand.latency.extra_delay_steps=[4,8]"],
                            note="delay beyond the training range"),
    "stress_direct": dict(from_="main_direct", group="stress",
                          eval=["--set", "domain_rand.latency.extra_delay_steps=[4,8]"],
                          note="delay beyond the training range"),
    "stress_direct_safety": dict(from_="main_direct", group="stress",
                                 eval=["--safety", "--set", "domain_rand.latency.extra_delay_steps=[4,8]"],
                                 note="same, with the safety layer on"),
    "stress_direct_nodelay": dict(from_="abl_direct_nodelay", group="stress",
                                  eval=["--set", "domain_rand.latency.extra_delay_steps=[4,8]"],
                                  note="trained without delay randomization"),
    # ---- OOD supervisor (controllers/supervisor.py): lag estimate + fall-back ---
    "sup_residual": dict(from_="main_residual", group="supervisor", eval=["--supervisor"],
                         note="supervisor on, in distribution (cost of false alarms)"),
    "sup_direct": dict(from_="main_direct", group="supervisor", eval=["--supervisor"],
                       note="supervisor on, in distribution"),
    "stress_residual_sup": dict(from_="main_residual", group="supervisor",
                                eval=["--supervisor", "--set", "domain_rand.latency.extra_delay_steps=[4,8]"],
                                note="supervisor on, delay beyond the training range"),
    "stress_direct_sup": dict(from_="main_direct", group="supervisor",
                              eval=["--supervisor", "--set", "domain_rand.latency.extra_delay_steps=[4,8]"],
                              note="supervisor on, delay beyond the training range"),
}
GROUPS = ["main", "ablations", "generalization", "stress", "supervisor"]


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def ckpt_of(name):
    e = EXPERIMENTS[name]
    return EXPERIMENTS[e["from_"]]["ckpt"] if "from_" in e else e["ckpt"]


def sh(cmd, log=None, env=None):
    print("  $", " ".join(cmd), flush=True)
    if log:
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    else:
        r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(cmd)}" + (f" -- see {log}" if log else ""))


def run_one(name, seed, args, env):
    e = EXPERIMENTS[name]
    stem = os.path.join("results", f"{ckpt_of(name)}_s{seed}")
    if "from_" not in e and not os.path.exists(os.path.join(ROOT, stem + ".zip")):
        cmd = [sys.executable, "scripts/train_sim_lite.py", "--mode", e["mode"],
               "--seed", str(seed), "--out", stem]
        if args.steps:
            cmd += ["--steps", str(args.steps)]
        for s in e.get("train", []):
            cmd += ["--set", s]
        os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
        sh(cmd, log=os.path.join(ROOT, "results", "logs", f"{os.path.basename(stem)}.log"), env=env)
    if not os.path.exists(os.path.join(ROOT, stem + ".zip")):
        print(f"  ! {stem}.zip missing -- run the source experiment first")
        return
    out = os.path.join("results", "eval", name, f"{os.path.basename(stem)}.json")
    os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
    cmd = [sys.executable, "scripts/eval.py", "--policy", stem + ".zip", "--out", out] + e.get("eval", [])
    sh(cmd, log=os.path.join(ROOT, "results", "logs", f"eval_{name}_s{seed}.log"), env=env)


def expand(names):
    out = []
    for n in names:
        if n == "all":
            out += list(EXPERIMENTS)
        elif n in GROUPS:
            out += [k for k, e in EXPERIMENTS.items() if e["group"] == n]
        elif n in EXPERIMENTS:
            out.append(n)
        else:
            raise SystemExit(f"unknown experiment or group {n!r}; see `list`")
    # sources first, so eval-only experiments find their checkpoints
    return sorted(dict.fromkeys(out), key=lambda k: "from_" in EXPERIMENTS[k])


def cmd_run(args):
    names = expand(args.names)
    seeds = parse_seeds(args.seeds)
    env = dict(os.environ)
    if args.jobs > 1:                       # one torch thread per process when running in parallel
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    for name in names:
        print(f"\n== {name}: {EXPERIMENTS[name]['note']}  (seeds {seeds})")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda s: run_one(name, s, args, env), seeds))
    if args.report:
        cmd_report(args)


def cmd_list(_args):
    for g in GROUPS:
        print(f"[{g}]")
        for k, e in EXPERIMENTS.items():
            if e["group"] == g:
                src = f" (eval of {e['from_']})" if "from_" in e else ""
                print(f"  {k:26s} {e['note']}{src}")


def _row(name, a):
    def f(t, sign=False, nd=3):
        if not t:
            return "-"
        fmt = f"{{:{'+' if sign else ''}.{nd}f}}"
        return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"
    return (f"| {name} | {a['n']} | {f(a['crash'])} | {f(a.get('rmse'))} | "
            f"{f(a.get('delta'), True)} | {f(a.get('hf_ratio'), nd=2)} | {a['improved']}/{a['n']} |")


def cmd_report(_args):
    import glob
    lines = ["# Results", "",
             "IQM over training runs with 95% bootstrap CI (Agarwal et al. 2021). "
             "Deltas and HF ratio are against `baseline_robust` on the same episodes "
             "(accuracy on episodes both survived). Fewer than 10 runs: indicative only.", ""]
    for g in GROUPS:
        conds = {}
        rows = []
        for name, e in EXPERIMENTS.items():
            if e["group"] != g:
                continue
            files = sorted(glob.glob(os.path.join(ROOT, "results", "eval", name, "*.json")))
            if not files:
                continue
            a = aggregate(files, "robust")
            if a is None:
                continue
            rows.append(_row(name, a))
            m = a["meta"]
            cond = f"trajectory={m.get('trajectory') or 'lemniscate'}, safety={'on' if m['safety'] else 'off'}"
            conds[cond] = a["baselines"]
        if not rows:
            continue
        lines += [f"## {g}", "",
                  "| experiment | runs | crash rate | settled RMSE (m) | Δ RMSE vs robust (m) | HF ratio vs robust | improved |",
                  "|---|---|---|---|---|---|---|", *rows, ""]
        for cond, b in conds.items():
            lines.append(f"Baselines ({cond}): " + "; ".join(
                f"`{k}` crash {v['crash_rate']:.2f}, RMSE {v.get('rmse_settled_m_mean', float('nan')):.3f} m, "
                f"HF {v.get('cmd_hf_rms_mean', float('nan')):.4f}" for k, v in b.items()))
        lines.append("")
    text = "\n".join(lines)
    out = os.path.join(ROOT, "results", "report.md")
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+", help=f"experiment names, a group ({', '.join(GROUPS)}) or 'all'")
    r.add_argument("--seeds", default="0-9")
    r.add_argument("--jobs", type=int, default=1, help="seeds trained in parallel")
    r.add_argument("--steps", type=int, default=None, help="override training steps (smoke tests)")
    r.add_argument("--report", action="store_true", help="write the report afterwards")
    sub.add_parser("list")
    sub.add_parser("report")
    args = ap.parse_args()
    {"run": cmd_run, "list": cmd_list, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
