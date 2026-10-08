"""Evaluation metrics, computed per episode.

Every env in an evaluation runs exactly one episode. If it crashes (hits a
termination condition) the episode is marked ``crashed`` and nothing after the
crash is counted -- the env's automatic reset must not leak a fresh episode's
transient into the statistics, and a crash must never be averaged away inside
an RMSE. Crash rate and tracking accuracy are therefore reported separately:

* reliability            -- crash rate
* tracking accuracy      -- position RMSE (whole episode, and after a 2 s
                            settling window), max error; survivors only
* control smoothness     -- command rate, CAPS smoothness Sm, and the share of
                            the command spectrum above the motor bandwidth
* the env parameters of each episode (delay, motor tau, wind) are returned
  alongside, so results can be sliced by condition (generalization)

Episode return is deliberately *not* among them: it depends on a reward the
designer chose and cannot be compared against a classical controller.

Smoothness Sm follows Mysore et al., "Regularizing Action Policies for Smooth
Control with Reinforcement Learning" (arXiv:2012.06644), Sec. IV, Eq. 4:
Sm = 2/(n fs) * sum_i M_i f_i over the FFT bins. Removing the mean and
averaging over the 4 rotor channels are this project's choices. ``cmd_hf_rms``
(this project's addition) is the RMS of the command content above the motor
bandwidth 1/(2 pi tau): motion the motors cannot follow, i.e. chatter, as
opposed to large but slow corrections, which also raise Sm. It is an absolute
amplitude on purpose -- a *share* of the spectrum would call a tiny,
noise-dominated command "chattery".
"""

from __future__ import annotations

import numpy as np

SETTLE_S = 2.0


def caps_smoothness(cmd: np.ndarray, fs: float) -> float:
    """CAPS Sm for one episode. ``cmd``: (T, C) -> mean over channels."""
    if cmd.shape[0] < 4:
        return float("nan")
    x = cmd - cmd.mean(axis=0, keepdims=True)
    mag = np.abs(np.fft.rfft(x, axis=0)) / x.shape[0]
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    n = mag.shape[0]
    sm = 2.0 / (n * fs) * (mag * f[:, None]).sum(axis=0)
    return float(sm.mean())


def hf_rms(cmd: np.ndarray, fs: float, f_cut: float) -> float:
    """RMS (mean over channels) of the part of ``cmd`` above ``f_cut`` Hz."""
    if cmd.shape[0] < 4:
        return float("nan")
    X = np.fft.rfft(cmd - cmd.mean(axis=0, keepdims=True), axis=0)
    f = np.fft.rfftfreq(cmd.shape[0], d=1.0 / fs)
    X[f <= f_cut] = 0.0
    hp = np.fft.irfft(X, n=cmd.shape[0], axis=0)
    return float(np.sqrt((hp ** 2).mean(axis=0)).mean())


def episode_metrics(err, cmd, omega, crashed, dt, motor_tau_nominal, tripped=None) -> dict:
    """Per-episode metrics.

    err: (T, N) position-error norm; cmd: (T, N, 4); omega: (T, N, 3);
    crashed: (N,) bool -- crashed episodes get NaN for every accuracy and
    smoothness metric (their samples are not used at all).
    """
    T, N = err.shape
    fs = 1.0 / dt
    f_cut = 1.0 / (2.0 * np.pi * motor_tau_nominal)
    k0 = int(round(SETTLE_S / dt))
    nan = np.full(N, np.nan)
    out = {k: nan.copy() for k in (
        "rmse_m", "rmse_settled_m", "max_err_m", "final_err_m", "cmd_rate_mean",
        "cmd_rate_p95", "smoothness_sm", "cmd_hf_rms", "body_rate_mean")}
    for i in np.nonzero(~crashed)[0]:
        e = err[:, i]
        c = cmd[:, i]
        out["rmse_m"][i] = np.sqrt(np.mean(e ** 2))
        out["rmse_settled_m"][i] = np.sqrt(np.mean(e[k0:] ** 2))
        out["max_err_m"][i] = e.max()
        out["final_err_m"][i] = e[-max(1, T // 10):].mean()
        rate = np.abs(np.diff(c, axis=0)) / dt
        out["cmd_rate_mean"][i] = rate.mean()
        out["cmd_rate_p95"][i] = np.percentile(rate, 95)
        out["smoothness_sm"][i] = caps_smoothness(c, fs)
        out["cmd_hf_rms"][i] = hf_rms(c, fs, f_cut)
        out["body_rate_mean"][i] = np.linalg.norm(omega[:, i], axis=-1).mean()
    out["crashed"] = crashed.astype(bool)
    if tripped is not None:
        out["safety_tripped"] = np.asarray(tripped, bool)
    return out


def summarize(ep: dict) -> dict:
    """Crash rate plus survivor mean / median of each metric."""
    crashed = ep["crashed"]
    s = {"episodes": int(crashed.size), "crash_rate": float(crashed.mean())}
    if "safety_tripped" in ep:
        s["safety_trip_rate"] = float(np.mean(ep["safety_tripped"]))
    if "fallback" in ep:
        s["fallback_rate"] = float(np.mean(ep["fallback"]))
    for k, v in ep.items():
        if k in ("crashed", "safety_tripped", "fallback") or not isinstance(v, np.ndarray) or v.dtype == bool:
            continue
        if k.startswith("env_"):
            continue
        ok = ~np.isnan(v)
        if ok.any():
            s[f"{k}_mean"] = float(v[ok].mean())
            s[f"{k}_median"] = float(np.median(v[ok]))
    return s


def paired(ep_a: dict, ep_ref: dict, key: str = "rmse_settled_m") -> dict:
    """A vs reference on the SAME episodes (same seed -> same disturbances).

    Accuracy is compared only where both survived; crashes are counted
    separately so a controller cannot look accurate by crashing on hard
    episodes, nor look bad because the reference crashed.
    """
    a, r = ep_a[key], ep_ref[key]
    both = ~ep_a["crashed"] & ~ep_ref["crashed"]
    d = a[both] - r[both]
    return {
        "metric": key,
        "both_survived": int(both.sum()),
        "only_ref_crashed": int((ep_ref["crashed"] & ~ep_a["crashed"]).sum()),
        "only_this_crashed": int((ep_a["crashed"] & ~ep_ref["crashed"]).sum()),
        "delta_mean": float(d.mean()) if d.size else float("nan"),
        "delta_median": float(np.median(d)) if d.size else float("nan"),
        "win_rate": float(np.mean(d < 0)) if d.size else float("nan"),
    }


def table(rows: dict, keys=None) -> str:
    """rows: {label: summary dict} -> aligned text table."""
    labels = list(rows)
    if keys is None:
        keys = []
        for m in rows.values():
            for k in m:
                if k not in keys:
                    keys.append(k)
    w = max(len(k) for k in keys) + 2
    cw = max(16, max(len(l) for l in labels) + 2)
    head = "metric".ljust(w) + "".join(f"{l:>{cw}}" for l in labels)
    lines = [head, "-" * len(head)]
    for k in keys:
        line = k.ljust(w)
        for l in labels:
            v = rows[l].get(k)
            if isinstance(v, float):
                line += f"{v:>{cw}.4f}"
            elif isinstance(v, int):
                line += f"{v:>{cw}d}"
            else:
                line += f"{'-':>{cw}}"
        lines.append(line)
    return "\n".join(lines)
