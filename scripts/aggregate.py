#!/usr/bin/env python3
"""Aggregate evaluations of several training seeds into method-level statistics.

    python scripts/aggregate.py results/eval/main_residual/*.json
    python scripts/aggregate.py --ref fast results/eval/abl_residual_fastbase/*.json

Unit of analysis = one training run, following Agarwal et al., "Deep RL at the
Edge of the Statistical Precipice" (NeurIPS 2021), Sec. 4.1 / 4.3: each run is
first reduced to one score (its mean over the evaluation episodes), then the
interquartile mean (IQM) across runs is reported with a percentile bootstrap
95% CI that resamples runs. All files must share the evaluation settings, so
the classical baselines -- which have no training randomness -- are identical
across files and are reported once.

The comparison yardstick is ``--ref`` (default ``robust``), not necessarily the
residual's own base, so that experiments with different bases stay comparable.
Accuracy is compared on the episodes both controllers survived; crashes are
reported separately.
"""
import argparse, json, sys

import numpy as np


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def boot_ci(x, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    stats = [iqm(rng.choice(x, x.size, replace=True)) for _ in range(reps)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def aggregate(files, ref="robust"):
    """Return a dict with per-run rows, method-level IQM/CI and the baselines."""
    runs, metas, base = [], [], None
    ref_key = f"baseline_{ref}"
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        m = d["meta"]
        if m["policy"] is None:
            continue
        metas.append(m)
        ep = d["episodes"]
        pol, r = ep[m["label"]], ep[ref_key]
        pc, rc = np.array(pol["crashed"], bool), np.array(r["crashed"], bool)
        a = np.array(pol["rmse_settled_m"], float)
        b = np.array(r["rmse_settled_m"], float)
        both = ~pc & ~rc
        hf = np.array(pol["cmd_hf_rms"], float)
        runs.append(dict(
            label=m["label"], crash=float(pc.mean()), ref_crash=float(rc.mean()),
            rmse=float(np.nanmean(a)) if (~pc).any() else np.nan,
            delta=float(np.mean(a[both] - b[both])) if both.any() else np.nan,
            win=float(np.mean(a[both] < b[both])) if both.any() else np.nan,
            hf=float(np.nanmean(hf)) if (~pc).any() else np.nan,
            ref_hf=float(np.nanmean(np.array(r["cmd_hf_rms"], float)))))
        base = d["summary"]
    if not runs:
        return None
    keys = {(m["seed"], m["episodes"], m["safety"], tuple(m["overrides"]), m["train_dist"],
             m.get("trajectory")) for m in metas}
    if len(keys) > 1:
        raise SystemExit(f"evaluations were run under different settings: {keys}")
    good = [r for r in runs if not np.isnan(r["delta"])]
    out = dict(runs=runs, n=len(runs), ref=ref_key, meta=metas[0], baselines={
        k: base[k] for k in ("baseline_fast", "baseline_robust") if k in base})
    out["crash"] = (iqm([r["crash"] for r in runs]), *boot_ci([r["crash"] for r in runs]))
    if good:
        out["rmse"] = (iqm([r["rmse"] for r in good]), *boot_ci([r["rmse"] for r in good]))
        out["delta"] = (iqm([r["delta"] for r in good]), *boot_ci([r["delta"] for r in good]))
        hfr = [r["hf"] / max(r["ref_hf"], 1e-12) for r in good]
        out["hf_ratio"] = (iqm(hfr), *boot_ci(hfr))
    out["improved"] = sum((r["delta"] < 0) and (r["crash"] <= r["ref_crash"]) for r in runs)
    return out


def _fmt(t, sign=False, nd=3):
    if t is None:
        return "-"
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default="robust", choices=["robust", "fast"])
    args = ap.parse_args()
    a = aggregate(args.files, args.ref)
    if a is None:
        sys.exit("no policy evaluations given")
    m = a["meta"]
    print(f"{a['n']} training runs, {m['episodes']} eval episodes each, eval seed {m['seed']}, "
          f"trajectory {m.get('trajectory') or 'config default'}, safety {'on' if m['safety'] else 'off'}")
    print("\nclassical baselines (no training randomness):")
    for k, s in a["baselines"].items():
        print(f"  {k:16s} crash {s['crash_rate']:.2f}   settled RMSE (survivors) "
              f"{s.get('rmse_settled_m_mean', float('nan')):.3f} m   cmd HF rms {s.get('cmd_hf_rms_mean', float('nan')):.4f}")
    print("\nper run:")
    for r in a["runs"]:
        print(f"  {r['label']:28s} crash {r['crash']:.2f}  settled RMSE {r['rmse']:.3f} m  "
              f"delta vs {args.ref} {r['delta']:+.4f} m  HF {r['hf']:.4f}")
    print(f"\nmethod level (IQM over runs, 95% bootstrap CI), reference {a['ref']}:")
    print(f"  crash rate            {_fmt(a['crash'])}   (reference {a['runs'][0]['ref_crash']:.3f})")
    print(f"  settled RMSE          {_fmt(a.get('rmse'))} m")
    print(f"  delta settled RMSE    {_fmt(a.get('delta'), sign=True, nd=4)} m")
    print(f"  HF chatter ratio      {_fmt(a.get('hf_ratio'), nd=2)}   (policy / reference)")
    print(f"  runs improved         {a['improved']}/{a['n']}")
    if a["n"] < 10:
        print("\nnote: fewer than 10 runs -- Agarwal et al. validate percentile CIs for IQM "
              "from 10 runs; treat these as indicative.")


if __name__ == "__main__":
    main()
