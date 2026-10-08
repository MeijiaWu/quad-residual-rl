#!/usr/bin/env python3
"""The two Isaac follow-up experiments: learning curve and OOD supervisor.

    python scripts/isaac_suite.py run curve ood     # evaluations (skips existing files)
    python scripts/isaac_suite.py report            # -> results/isaac_report.md
    python scripts/isaac_suite.py run ood --gate support   # conservative gate variant

Needs the 10 + 10 Isaac training runs in results/isaac/{direct,residual}_s{0..9}/.
Every evaluation is one ``scripts/eval_isaac.py`` call (100 episodes); the
baselines are cached, so each call starts one Isaac process. ``--jobs 2`` runs
two at a time (each process needs a few GB of GPU memory).

curve  Each saved checkpoint of every run, on the usual evaluation seed 12345.
       Question: is the residual's advantage in sample efficiency real across
       seeds, and how many samples does each method need?
ood    The final checkpoints with and without the supervisor
       (controllers/supervisor.py):
       * stress (extra delay 4-8 steps) on a NEW seed, 54321 -- the lag variable
         was chosen after looking at the seed-12345 stress results, so the
         test must not reuse those episodes;
       * in distribution, seed 12345, against the existing runs without it.

Pre-registered criteria (fixed before any supervisor result on Isaac):
  S1  stress: pooled crash rate falls by >= 80 % relative to the same policies
      without the supervisor;
  S2  in distribution: fallback rate <= 5 % and IQM settled-RMSE cost <= 0.01 m;
  S3  stress: the policies still beat `robust` on accuracy (delta CI below 0).
Learning-curve threshold: first checkpoint with delta vs `robust` <= -0.15 m
(about half of `robust`'s error).
"""
import argparse, json, os, shutil, subprocess, sys
from glob import glob

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import aggregate, boot_ci, iqm  # noqa: E402

EV = os.path.join(ROOT, "results", "isaac_eval")
RUNS = os.path.join(ROOT, "results", "isaac")
STRESS = "domain_rand.latency.extra_delay_steps=[4,8]"
SEED_MAIN, SEED_HELDOUT = 12345, 54321
SAMPLES_PER_ITER = 4096 * 24
THRESHOLD = -0.15
S1_REDUCTION, S2_FALLBACK, S2_COST = 0.80, 0.05, 0.01


def seeds_arg(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def checkpoints(mode, seed):
    files = glob(os.path.join(RUNS, f"{mode}_s{seed}", "model_*.pt"))
    return {int(os.path.basename(f)[6:-3]): f for f in files}


JOBS = []


def eval_isaac(ckpt, mode, out, seed, extra=(), dry=False):
    """Queue one evaluation unless its output exists."""
    if os.path.exists(out) or any(j[1] == out for j in JOBS):
        return
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "eval_isaac.py"), "--rsl", ckpt,
           "--mode", mode, "--seed", str(seed), "--out", out, *extra]
    JOBS.append((cmd, out))


def execute(jobs, n_jobs, dry):
    from concurrent.futures import ThreadPoolExecutor
    print(f"{len(jobs)} evaluations to run", flush=True)

    def one(i_job):
        i, (cmd, out) = i_job
        print(f"[{i + 1}/{len(jobs)}] {os.path.relpath(out, ROOT)}", flush=True)
        if dry:
            print("  $", " ".join(cmd))
            return 0
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".log", "w") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            print(f"! failed ({r.returncode}): see {os.path.relpath(out, ROOT)}.log", flush=True)
        else:
            os.remove(out + ".log")
        return r.returncode

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        codes = list(ex.map(one, enumerate(jobs)))
    if any(codes):
        sys.exit(f"{sum(c != 0 for c in codes)} evaluation(s) failed")


def ood_paths(mode, seed, gate):
    sfx = "" if gate == "lag" else f"_{gate}"
    d = os.path.join(EV, "ood")
    return {"stress_nosup": os.path.join(d, f"stress_nosup_{mode}_s{seed}.json"),
            f"stress_sup{sfx}": os.path.join(d, f"stress_sup{sfx}_{mode}_s{seed}.json"),
            "indist_nosup": os.path.join(EV, f"isaac_{mode}_s{seed}.json"),
            f"indist_sup{sfx}": os.path.join(d, f"indist_sup{sfx}_{mode}_s{seed}.json")}


# ------------------------------------------------------------------- run
def run(args):
    for mode in args.modes:
        for s in args.seeds:
            ck = checkpoints(mode, s)
            if not ck:
                print(f"! no checkpoints in results/isaac/{mode}_s{s}/ -- skipped")
                continue
            if "curve" in args.what:
                for it in args.iters:
                    it_ = it if it in ck else (max(ck) if it >= max(ck) else None)
                    if it_ is None:
                        print(f"! {mode}_s{s}: no model_{it}.pt")
                        continue
                    out = os.path.join(EV, "curve", f"{mode}_s{s}_it{it_}.json")
                    legacy = os.path.join(EV, f"curve_{mode}_s{s}_it{it_}.json")
                    if not os.path.exists(out) and os.path.exists(legacy):
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        shutil.copy(legacy, out)
                    eval_isaac(ck[it_], mode, out, SEED_MAIN, dry=args.dry_run)
            if "ood" in args.what:
                final = ck[max(ck)]
                p = ood_paths(mode, s, args.gate)
                sup = ["--supervisor", "--gate", args.gate]
                sfx = "" if args.gate == "lag" else f"_{args.gate}"
                eval_isaac(final, mode, p["stress_nosup"], SEED_HELDOUT, ["--set", STRESS], args.dry_run)
                eval_isaac(final, mode, p[f"stress_sup{sfx}"], SEED_HELDOUT,
                           ["--set", STRESS, *sup], args.dry_run)
                eval_isaac(final, mode, p["indist_nosup"], SEED_MAIN, (), args.dry_run)
                eval_isaac(final, mode, p[f"indist_sup{sfx}"], SEED_MAIN, sup, args.dry_run)
    # OOD first: fewer, and the more important result
    JOBS.sort(key=lambda j: 0 if os.sep + "curve" + os.sep not in j[1] else 1)
    execute(JOBS, args.jobs, args.dry_run)


# ---------------------------------------------------------------- report
def fmt(t, nd=3, sign=False):
    if t is None:
        return "-"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


def stat(x):
    x = [v for v in x if not np.isnan(v)]
    return (iqm(x), *boot_ci(x)) if x else (np.nan, np.nan, np.nan)


def diff_ci(a, b, reps=5000, seed=0):
    """IQM(a) - IQM(b), runs resampled independently per method."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [iqm(rng.choice(a, a.size)) - iqm(rng.choice(b, b.size)) for _ in range(reps)]
    return iqm(a) - iqm(b), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load(f):
    with open(f) as fh:
        d = json.load(fh)
    lab = d["meta"]["label"]
    ep = {k: np.array(v, float if k not in ("crashed", "fallback") else bool)
          for k, v in d["episodes"][lab].items()}
    return d, ep


def curve_report(L, modes, seeds):
    files = sorted(glob(os.path.join(EV, "curve", "*.json")))
    if not files:
        return
    iters = sorted({int(f.rsplit("_it", 1)[1][:-5]) for f in files})
    L += ["## Learning curve", "",
          f"Every saved checkpoint, eval seed {SEED_MAIN}, 100 episodes; IQM over runs [95 % CI]. "
          "Delta = settled RMSE minus `robust` on the episodes both survived.", ""]
    per = {}
    for m in modes:
        for it in iters:
            fs = [os.path.join(EV, "curve", f"{m}_s{s}_it{it}.json") for s in seeds]
            fs = [f for f in fs if os.path.exists(f)]
            if fs:
                per[m, it] = aggregate(fs)
    L += ["| iteration | samples | " + " | ".join(f"{m} delta (m)" for m in modes)
          + " | residual - direct | " + " | ".join(f"{m} HF x" for m in modes)
          + " | " + " | ".join(f"{m} crash" for m in modes) + " | runs |",
          "|" + "---|" * (3 + 3 * len(modes) + 1)]
    for it in iters:
        row = [str(it), f"{it * SAMPLES_PER_ITER / 1e6:.0f} M"]
        for m in modes:
            row.append(fmt(per[m, it]["delta"], 3, True) if (m, it) in per and "delta" in per[m, it] else "-")
        if all((m, it) in per for m in ("residual", "direct")):
            a = [r["delta"] for r in per["residual", it]["runs"]]
            b = [r["delta"] for r in per["direct", it]["runs"]]
            row.append(fmt(diff_ci(a, b), 3, True))
        else:
            row.append("-")
        for m in modes:
            row.append(fmt(per[m, it]["hf_ratio"], 2) if (m, it) in per and "hf_ratio" in per[m, it] else "-")
        for m in modes:
            row.append(f"{per[m, it]['crash'][0]:.3f}" if (m, it) in per else "-")
        row.append("/".join(str(per[m, it]["n"]) if (m, it) in per else "0" for m in modes))
        L.append("| " + " | ".join(row) + " |")
    # iterations to threshold
    L += ["", f"Iterations until delta <= {THRESHOLD} m (first checkpoint; runs that never "
          "get there count as the last iteration + 1 and are listed):", ""]
    reach = {}
    for m in modes:
        vals, never = [], []
        for s in seeds:
            hit = None
            for it in iters:
                f = os.path.join(EV, "curve", f"{m}_s{s}_it{it}.json")
                if not os.path.exists(f):
                    continue
                a = aggregate([f])
                if a and "delta" in a and a["delta"][0] <= THRESHOLD:
                    hit = it
                    break
            if hit is None:
                never.append(s)
                hit = iters[-1] + 1
            vals.append(hit)
        reach[m] = vals
        L.append(f"- {m}: IQM {fmt(stat(vals), 0)} iterations; per run {vals}"
                 + (f"; never: seeds {never}" if never else ""))
    if all(m in reach for m in ("residual", "direct")):
        d = diff_ci(reach["residual"], reach["direct"])
        L.append(f"- residual - direct: {fmt(d, 0, True)} iterations "
                 f"({'significant' if d[2] < 0 or d[1] > 0 else 'CI spans 0'})")
    L.append("")


def ood_report(L, modes, seeds, gate):
    sfx = "" if gate == "lag" else f"_{gate}"
    have = [m for m in modes if all(os.path.exists(p) for s in seeds
                                    for p in ood_paths(m, s, gate).values())]
    missing = [m for m in modes if m not in have]
    L += [f"## OOD supervisor (gate: {gate})", ""]
    if missing:
        L += [f"(incomplete for {missing} -- run `isaac_suite.py run ood" +
              ("" if gate == "lag" else f" --gate {gate}") + "`)", ""]
    for m in have:
        P = {s: ood_paths(m, s, gate) for s in seeds}
        st_n = aggregate([P[s]["stress_nosup"] for s in seeds])
        st_s = aggregate([P[s][f"stress_sup{sfx}"] for s in seeds])
        in_n = aggregate([P[s]["indist_nosup"] for s in seeds])
        in_s = aggregate([P[s][f"indist_sup{sfx}"] for s in seeds])
        pooled = {}
        for k in ("stress_nosup", f"stress_sup{sfx}"):
            c = [load(P[s][k])[1]["crashed"] for s in seeds]
            pooled[k] = (int(sum(x.sum() for x in c)), int(sum(x.size for x in c)))
        red = 1 - (pooled[f"stress_sup{sfx}"][0] / max(pooled["stress_nosup"][0], 1))
        fb_in, fb_st, cost, lagerr = [], [], [], []
        for s in seeds:
            _, a = load(P[s][f"indist_sup{sfx}"])
            _, b = load(P[s]["indist_nosup"])
            fb_in.append(a["fallback"].mean())
            both = ~a["crashed"] & ~b["crashed"]
            cost.append(float(np.mean(a["rmse_settled_m"][both] - b["rmse_settled_m"][both])))
            _, c = load(P[s][f"stress_sup{sfx}"])
            fb_st.append(c["fallback"].mean())
            for e in (a, c):
                ok = ~np.isnan(e["env_lag_hat"])
                lagerr += list(np.abs(e["env_lag_hat"][ok] - e["env_lag"][ok]))
        s1 = red >= S1_REDUCTION
        s2 = iqm(fb_in) <= S2_FALLBACK and iqm(cost) <= S2_COST
        s3 = "delta" in st_s and st_s["delta"][2] < 0
        L += [f"### {m}", "",
              "| | crash | settled RMSE delta vs robust | HF x | fallback |", "|---|---|---|---|---|",
              f"| stress, no supervisor (seed {SEED_HELDOUT}) | {fmt(st_n['crash'])} "
              f"({pooled['stress_nosup'][0]}/{pooled['stress_nosup'][1]}) | "
              f"{fmt(st_n.get('delta'), 3, True)} | {fmt(st_n.get('hf_ratio'), 2)} | - |",
              f"| stress, supervisor | {fmt(st_s['crash'])} "
              f"({pooled[f'stress_sup{sfx}'][0]}/{pooled[f'stress_sup{sfx}'][1]}) | "
              f"{fmt(st_s.get('delta'), 3, True)} | {fmt(st_s.get('hf_ratio'), 2)} | {fmt(stat(fb_st), 2)} |",
              f"| in distribution, no supervisor (seed {SEED_MAIN}) | {fmt(in_n['crash'])} | "
              f"{fmt(in_n.get('delta'), 3, True)} | {fmt(in_n.get('hf_ratio'), 2)} | - |",
              f"| in distribution, supervisor | {fmt(in_s['crash'])} | "
              f"{fmt(in_s.get('delta'), 3, True)} | {fmt(in_s.get('hf_ratio'), 2)} | {fmt(stat(fb_in), 3)} |",
              "",
              f"- in-distribution accuracy cost (supervisor - none, paired episodes): "
              f"{fmt(stat(cost), 4, True)} m",
              f"- lag estimate at the gate decision: MAE {1000 * np.mean(lagerr):.1f} ms, "
              f"p90 {1000 * np.percentile(lagerr, 90):.1f} ms",
              f"- **S1** crash reduction {100 * red:.0f} % (need >= {100 * S1_REDUCTION:.0f} %): "
              f"{'PASS' if s1 else 'FAIL'}",
              f"- **S2** fallback {100 * iqm(fb_in):.1f} % (<= {100 * S2_FALLBACK:.0f} %), cost "
              f"{iqm(cost):+.4f} m (<= {S2_COST}): {'PASS' if s2 else 'FAIL'}",
              f"- **S3** stress delta CI below 0: {'PASS' if s3 else 'FAIL'}", ""]
        # crashes by effective lag
        bins = [0, 0.060, 0.076, 0.080, 0.085, 0.090, 1.0]
        L += ["Crashes by true effective lag (delay + tau), stress, pooled over runs:", "",
              "| lag (ms) | " + " | ".join(f"{1000 * a:.0f}-{1000 * b:.0f}" if b < 1 else f">{1000 * a:.0f}"
                                         for a, b in zip(bins[:-1], bins[1:])) + " |",
              "|---|" + "---|" * (len(bins) - 1)]
        for k, name in (("stress_nosup", "no supervisor"), (f"stress_sup{sfx}", "supervisor")):
            cells = []
            for a, b in zip(bins[:-1], bins[1:]):
                cr = tot = 0
                for s in seeds:
                    _, e = load(P[s][k])
                    sel = (e["env_lag"] >= a) & (e["env_lag"] < b)
                    cr += int(e["crashed"][sel].sum())
                    tot += int(sel.sum())
                cells.append(f"{cr}/{tot}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")


def report(args):
    L = ["# Isaac follow-up: learning curve and OOD supervisor", "",
         "Generated by `scripts/isaac_suite.py report`. Criteria were fixed in the script "
         "docstring before the runs.", ""]
    curve_report(L, args.modes, args.seeds)
    for gate in ("lag", "support"):
        if gate == "lag" or glob(os.path.join(EV, "ood", "stress_sup_support_*.json")):
            ood_report(L, args.modes, args.seeds, gate)
    out = os.path.join(ROOT, "results", "isaac_report.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("what", nargs="*", default=["curve", "ood"])
    ap.add_argument("--modes", nargs="+", default=["direct", "residual"])
    ap.add_argument("--seeds", type=seeds_arg, default=list(range(10)))
    ap.add_argument("--iters", type=int, nargs="+", default=[100, 200, 300, 400, 600, 800, 999])
    ap.add_argument("--gate", default="lag", choices=["lag", "support"])
    ap.add_argument("--jobs", type=int, default=1, help="evaluations in parallel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        bad = set(args.what) - {"curve", "ood"}
        if bad:
            sys.exit(f"unknown experiment(s): {bad}")
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
