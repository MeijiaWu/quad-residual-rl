# Residual RL for Quadrotor Trajectory Tracking

Reinforcement learning for multirotor attitude and position control, comparing
three approaches on the same task, the same reward and the same disturbances:

| mode | what the policy outputs |
|---|---|
| `baseline` | nothing — the classical cascaded geometric controller runs alone |
| `direct` | the rotor command itself |
| `residual` | a bounded correction **added to** the classical controller's command |

Residual control is the interesting case: the classical controller handles the
nominal dynamics it was designed for, and the policy only has to learn what the
model got wrong — unmodelled aerodynamics, motor nonlinearity, wind. That is a
much smaller function to learn than the full control law, and the aircraft is
still flyable when the policy outputs zero.

## Status

**Both paths run.** sim_lite (CPU, SB3) and Isaac Lab (GPU, rsl_rl) — see Findings and Isaac Lab results.

| component | state |
|---|---|
| quadrotor dynamics, motor lag, per-env actuation delay | done, tested |
| cascaded geometric controller (baseline) | done, tested — `fast` / `robust` presets, optional position integral |
| fair baseline: nominal params, same noisy measurement as the policy | done, tested |
| domain randomization (parameter / latency / wind / observation) + curriculum | done, tested |
| trajectories, reward, termination, safety layer | done |
| evaluation: one episode per env, crash rate separate, paired comparison, multi-seed IQM | done, tested |
| `sim_lite` CPU environment | done — all three modes run |
| system identification | done — recovers τ and delay (including zero delay) from noisy data |
| config files drive the code (`configs/*.yaml`, `--set` overrides) | done, tested |
| latency benchmark + TorchScript export | done |
| PPO training on `sim_lite` (SB3) | done — 10 seeds per method + ablations |
| OOD supervisor (lag estimate + fall-back), sim_lite and Isaac core | done, tested (parity step for step); evaluated in Isaac, all pre-registered criteria met |
| Isaac Lab env (Isaac Sim 5.1 / Isaac Lab 2.3) | done — sim2sim matches sim_lite; 10 seeds per method trained and evaluated on an RTX 4090 |

`python3 tests/test_core.py` — 40 tests.

## Layout

```
source/quad_residual/
  backend.py              numpy/torch shim + quaternion math
  config.py               configs/*.yaml -> dataclasses, --set overrides, snapshots
  dynamics/actuator.py    per-env delay + motor lag (numpy, plus torch twin for Isaac)
  dynamics/quadrotor.py   batched rigid body; rotor order/spin and yaw sign
  dynamics/randomization.py  parameter / latency / observation randomization
  controllers/geometric.py   cascaded geometric controller (the baseline and
                             the base of the residual)
  controllers/safety.py      rate limiting, anomaly monitor, latching fallback
  controllers/supervisor.py  OOD supervisor: online actuation-lag estimate, fall back
                             to the base controller outside the training distribution
  tasks/trajectories.py      hover, lemniscate, random waypoints (with ff accel)
  utils/metrics.py           per-episode crash / accuracy / smoothness metrics, paired comparison
  sim_lite/env.py            CPU environment — the source of truth for reward,
                             observation and termination
  task.py                    observation / reward / termination, shared by both envs
  envs/torch_core.py         all Isaac-env task logic in plain torch (CPU-testable)
  envs/quad_track_env*.py    Isaac Lab DirectRLEnv: thin PhysX glue around the core
scripts/   train_sim_lite, eval, aggregate, experiments, select_baseline_gains, sysid, bench_latency,
           check_supervisor, train_isaac, eval_isaac, isaac_suite, check_torch_core
configs/   env.yaml, domain_rand.yaml, ppo.yaml
tests/     test_core.py
```

### Why two environments

Isaac Lab needs a CUDA GPU and a multi-gigabyte Isaac Sim install. Reward
shaping, observation design and curriculum logic do not — and getting those
wrong is the usual reason a run fails to converge. `sim_lite` runs the same
task on NumPy so that work happens in seconds on a laptop. The Isaac Lab env
then imports the same reward and observation functions, so the two cannot
silently drift apart.

Inside Isaac Lab, PhysX integrates the rigid body — the NumPy integrator is for
the CPU path only. But the **actuator** model is still applied by hand in both,
because PhysX models neither motor lag nor transport delay, and on a small
multirotor those two terms dominate the sim-to-real gap.

## Quick start

```bash
python3 tests/test_core.py                 # 40 tests; numpy + pyyaml (+ torch for the torch/supervisor checks)
python3 scripts/select_baseline_gains.py   # re-derives the robust preset from the rule below
python3 scripts/eval.py                    # both baselines + residual(zero) sanity check
python3 scripts/sysid.py
python3 scripts/bench_latency.py

pip install stable-baselines3 gymnasium torch pyyaml
for s in 0 1 2 3 4 5 6 7 8 9; do                        # 10 runs per method
  python3 scripts/train_sim_lite.py --mode residual --seed $s --out results/residual_s$s
  python3 scripts/eval.py --policy results/residual_s$s.zip --out results/eval_residual_s$s.json
done
python3 scripts/aggregate.py results/eval_residual_s*.json
```

Training writes `<out>.zip` plus `<out>.cfg.yaml`, the fully resolved config.
`eval.py --policy` takes the policy's interface (mode, residual scale, the base
controller) from that snapshot, but the *evaluation conditions* from
`configs/domain_rand.yaml`, so every checkpoint — ablations trained under a
different distribution included — is tested under the same conditions.
Checkpoints from before the baseline change are refused: their residual was
trained on a different base controller.

## Design notes

**Cascade bandwidth.** The attitude torque is scaled by the inertia, so the
closed-loop angular dynamics reduce to `θ̈ = -kr·e_R - kw·e_ω` and the gains map
straight onto a second-order response: `kr = ωₙ²`, `kw = 2ζωₙ`. Attitude is set
from a single natural frequency, position at 1/8 of it. That ~8× separation is
what makes the cascade stable — at 2× the inner loop is too slow to serve the
outer one and the whole thing limit-cycles. (It did, during development;
`test_controller_tracks_step` is the regression test.)

**Two baselines, because a fixed-gain controller must pick a trade-off.**
`fast` (attitude ωₙ = 20 rad/s, the original tuning) tracks tightly but
limit-cycles or diverges over much of the randomized delay × motor-lag range.
`robust` (8 rad/s) is chosen by `scripts/select_baseline_gains.py` with a rule
fixed before looking at any RL result: the largest candidate bandwidth that is
stable at every corner of the randomization range (mass, inertia, thrust, motor
τ, delay at their extremes) plus a margin (+1 delay step, +20 % τ), with the
controller using nominal parameters. `robust` is the residual's base and the
safety fallback — it must never be the thing that crashes. The question for RL
is whether it beats *this trade-off*: near-`fast` accuracy at `robust`'s crash
rate.

**A fair baseline.** The classical controller gets exactly what a real flight
controller has: nominal airframe parameters (never the episode's randomized
mass, inertia or thrust coefficient) and the same noisy, biased state
measurement the policy observes — one draw per step, shared by both. It has a
position integral (PID) with anti-windup, as any real stack would; without one
a mass or thrust mismatch leaves a large altitude offset that a learned policy
would "win" against trivially. Episodes start from a steady hover, so the
altitude integrator starts converged (`GeometricController.reset(hover_ratio=…)`).
`baseline_info: privileged` restores the old, unfair setting for ablation only.

**Feed-forward for the baseline.** Every trajectory returns an analytic
acceleration and the baseline uses it. A baseline denied feed-forward looks
artificially bad next to a learned policy, which makes the comparison worthless.

**Metrics, not return.** Episode return is a function of a reward the designer
chose and cannot be compared against a controller that has no reward at all.

**Crashes are counted, never averaged away.** Every env in an evaluation runs
one episode; after a termination nothing more is recorded (the auto-reset would
otherwise leak a fresh episode's transient into the statistics). Reported:
crash rate; settled tracking RMSE (after 2 s) over survivors; CAPS smoothness
Sm (Mysore et al., arXiv:2012.06644, Eq. 4) and the RMS of command content
above the motor bandwidth (chatter). Policy vs base is compared **pairwise** on
the same episodes, on those both survived, with crashes listed separately.
Across training seeds, `aggregate.py` reduces each run to one score and reports
the interquartile mean with a bootstrap CI over runs (Agarwal et al., NeurIPS
2021, Sec. 4.1/4.3); use at least 10 runs.

**Latency is randomized, not fixed.** Actuation delay is the most commonly
underestimated term in sim-to-real on small multirotors; a policy trained at
zero delay generally fails on hardware regardless of how well everything else
matches. (In this simulator the training range turned out not to be where it
matters: see Findings — a policy trained without delay randomization is as good
inside the range, but crashes twice as often beyond it.) Each env draws its own delay per episode (nominal 1 + extra 0–3
physics steps, i.e. 4–16 ms); the plant's buffer is sized for the largest draw
and each env reads its own depth (`dynamics/actuator.py`). `scripts/sysid.py`
identifies the real value from bench logs using that same actuator function,
and the spread across runs is what should set the randomization range.

**Curriculum starts at nominal.** At curriculum scale 0 every randomized range
collapses onto the nominal `QuadParams` — no wind, no extra delay, no sensor
noise — and widens linearly to the configured `[lo, hi]` over the first
`curriculum_fraction` of training.

**Safety layer seeds from hover.** After a reset the rate limiter starts from
the hover command, not zero; otherwise the first commands of every episode
would be clamped to a slow ramp up from zero thrust.

**Rotor convention.** Order and spin directions follow PX4 "Quadrotor X"
(0 FR CCW, 1 RL CCW, 2 FL CW, 3 RR CW). A rotor's drag reaction opposes its
spin, so CW rotors give positive yaw torque. Plant and controller share the
mixing matrix, so a sign error there is invisible in simulation and only shows
up on a real airframe or USD asset — `test_yaw_reaction_opposes_spin` pins it.

## Experiment plan

Pre-registered claim: *under the training distribution, residual RL on the
`robust` base crashes no more often than `robust` and reduces its settled
tracking RMSE (IQM CI below 0 over ≥ 10 runs), without raising high-frequency
command content by more than 20 %.* Plot every controller as a point on crash
rate vs RMSE; the claim is that RL lies outside the `fast`–`robust` trade-off.

All experiments are defined in one place, `scripts/experiments.py`, and run
through it — training is skipped when a checkpoint exists, evaluation is
re-run, and `report` writes `results/report.md`:

```bash
python3 scripts/experiments.py list
python3 scripts/experiments.py run main --seeds 0-9 --jobs 2
python3 scripts/experiments.py run ablations --seeds 0-9 --jobs 2
python3 scripts/experiments.py run generalization          # eval-only, reuses main checkpoints
python3 scripts/experiments.py report
```

| group | experiment | the one change |
|---|---|---|
| main | `main_residual`, `main_direct` | — |
| ablations | `abl_residual_fastbase` | residual on the `fast` base — does the slow base limit the residual? |
| | `abl_residual_scale05` | residual authority 0.25 → 0.5 |
| | `abl_direct_nodelay` | no delay randomization in training |
| | `abl_direct_norand` | no domain randomization in training |
| | `abl_direct_nosmooth` | no action-smoothness penalty |
| | `abl_direct_nocurriculum` | full randomization from step one |
| generalization | `wp_residual`, `wp_direct` | evaluated on random waypoint steps (trained on the lemniscate only) |
| | `safety_residual`, `safety_direct` | evaluated with the deployment safety layer on |
| stress | `stress_*` | evaluated with extra delay 4–8 steps (20–36 ms total), beyond the 0–3 trained on |

Ablations change *training* only; every checkpoint is evaluated on the same
episodes under `configs/domain_rand.yaml`. Deltas in the report are against
`baseline_robust` for every row, so rows with different residual bases stay
comparable. Report the table with CIs, not one best number.

## Findings (10 training runs per row, IQM with 95% bootstrap CI)

Full table: `python3 scripts/experiments.py report` → `results/report.md`.

1. **The classical trade-off is real.** `fast`: 31 % crashes, 0.147 m settled
   RMSE on survivors. `robust`: 0 % crashes, 0.312 m.
2. **Direct RL leaves it, inside the training distribution.** 0 % crashes,
   0.111 m — about `fast`'s accuracy on the episodes `fast` survives, with half
   its high-frequency command content. Holds on unseen waypoint steps
   (0 % crashes vs 35 % for `fast`) and with the safety layer on.
3. **Residual RL is bounded by its base — at sim_lite's budget.** *(Revised by
   the Isaac runs below: with ~300× more samples it matches direct RL.)* On `robust`: 0 % crashes, −19 % RMSE
   (Δ −0.059 m), but 1.8× the base's high-frequency content, so the
   pre-registered smoothness criterion (≤ 1.2×) is **not met**. Doubling its
   authority buys nothing (Δ +0.012 m, CI spans 0) and doubles chatter. On
   `fast`: better than `fast` (−3.3 cm, crashes 31 % → 10 %), but it cannot
   remove the fragility.
4. **Ablations (direct).** No domain randomization: +0.101 m RMSE, worst for
   heavy airframes — mass/thrust randomization is the ingredient that matters.
   No smoothness penalty: +0.025 m and no significant change in chatter.
   No curriculum: no significant effect. No delay randomization: no effect
   *inside* the 0–3 step range.
5. **Out of distribution, the ranking flips.** With 4–8 extra delay steps:
   `robust` 0 % crashes, residual 0 % (still −0.040 m vs `robust`), direct 16 %,
   direct trained without delay randomization 30 %. The safety layer does not
   rescue direct RL (18 %): its monitor trips after the aircraft is already
   unrecoverable. Direct RL is the most accurate controller where it was
   trained; residual on a robust base is the one that degrades gracefully.
   *(Weakened by the Isaac runs: the better-trained residual also crashes out of
   distribution.)*

6. **OOD supervisor (sim_lite).** The supervisor
   (`controllers/supervisor.py`, below) flies `robust` for 0.5 s while it
   identifies the effective actuation lag, then hands over to the policy unless
   the lag exceeds the training maximum (76 ms). Stress test: direct RL crashes
   182 → 29 / 1000, residual 11 → 1 / 1000. In distribution: fallback on 3–4 %
   of episodes (lags within a few ms of the boundary); accuracy cost +0.014 m
   for direct, none for residual (−0.007 m). The remaining direct crashes have
   a lag *inside* the training range but a delay outside it: sim_lite's direct
   policies are sensitive to delay itself, which the lag gate does not see
   (`--gate support` covers it, at ~14 % false alarms).

**Waypoint test.** Each episode gets its own random walk: hover at the start
point for 3 s, then a step every 3 s, 0.75–1.5 m in a random 3-D direction,
kept inside a ±2 × ±2 × ±1 m box. No feed-forward. Steps are bounded below the
4 m termination limit so a crash is the controller's, not the test's.

## Isaac Lab results (10 runs per row, IQM with 95 % bootstrap CI)

rsl_rl PPO, 4096 envs × 24 steps × 1000 iterations ≈ 98 M samples per run
(sim_lite: 300 k). Same evaluation protocol as above: 100 episodes, seed 12345,
reference `robust`. Baselines in PhysX match sim_lite (`robust` 0.308 vs
0.312 m, `fast` crashes 38 % vs 31 %), and the sim_lite-trained direct policy
flown in PhysX reproduces its sim_lite numbers (0 %, 0.111 m), so differences
below come from training, not from the simulator.

| | crashes | settled RMSE | Δ vs `robust` | HF vs `robust` |
|---|---|---|---|---|
| `fast` | 38 % | 0.172 m | – | 11.9× |
| `robust` | 0 % | 0.308 m | 0 | 1.0× |
| direct (Isaac) | 2 / 1000 | 0.090 [0.081, 0.123] | −0.217 [−0.227, −0.184] | 2.67× [2.34, 3.03] |
| residual (Isaac) | 0 / 1000 | 0.084 [0.075, 0.114] | −0.223 [−0.232, −0.194] | 1.80× [1.70, 1.91] |

1. **Residual catches up with direct.** Residual − direct RMSE: −0.006 m,
   CI [−0.041, +0.025] — no difference. Residual has lower chatter (HF diff CI
   [−0.0084, −0.0035]) and no crashes. Pre-registered claim: no extra crashes
   ✅, RMSE CI below 0 ✅, HF ≤ 1.2× ❌ (1.80× — unchanged from sim_lite's 1.83×,
   so the chatter is structural, not a training-budget effect).
2. **Learning curve (seed 0 only).** Δ vs `robust` at iteration
   100 / 200 / 400 / 600 / 999 (≈ 10 / 20 / 39 / 59 / 98 M samples):
   residual +1.39 / −0.13 / −0.23 / −0.24 / −0.23 m;
   direct +0.28 / +0.03 / −0.17 / −0.23 / −0.22 m.
   Residual reaches its plateau about twice as fast. Chatter grows with
   training (HF residual 1.1× → 1.9×, direct 2.3× → 3.1×): accuracy is bought
   with high-frequency command content. sim_lite's 300 k samples are
   <1 % of the first point here, which explains finding 3's small residual gain;
   the SB3/rsl_rl hyper-parameter difference is not separately controlled.
3. **Out of distribution (4–8 extra delay steps), both RL policies crash.**
   `robust` 0 %, `fast` 94 %; direct 8.8 % [4.5, 11.8], residual 6.0 %
   [4.2, 8.8] (difference not significant, CI [−6.3, +2.3] pp). Crashes
   concentrate at the largest delay (total 9 steps = 36 ms; direct 20 %,
   residual 18 %) combined with slow motors (τ 45–57 ms). Below that,
   residual is almost crash-free (7 / 670) and direct is at ~2.5 % (17 / 670).
   The well-trained residual uses its authority and so inherits RL's OOD
   fragility, unlike the under-trained sim_lite residual (0 %). Only the
   fixed-gain `robust` controller is safe at every tested delay; using RL at the
   edge of the envelope needs a delay estimate and a fall-back to `robust`, or
   training on the wider range.

Caveat: the 100 stress episodes draw delay 9 for 33 of them (uniform would
be 20), which inflates the pooled crash rates; the per-delay numbers above are
the ones to read.

### Follow-ups (`make isaac-suite`, report in `results/isaac_report.md`)

**OOD supervisor.** Finding 3 shows *where* the policies fail: crashes jump
exactly where the effective actuation lag `delay + motor_tau` leaves the
training range (76 ms) — residual 2 / 1450 inside, 66 / 300 beyond. Lag is
also what can be identified online: delay and a first-order motor lag cost
almost the same phase at the attitude-loop bandwidth, so their sum is
well-determined from closed-loop data while each term alone is not.

`controllers/supervisor.py`: for every candidate `(delay, tau)` on a grid it
replays the issued rotor commands through that delay + lag (the plant's own
actuator model) and fits the measured change in roll/pitch rate by least
squares, with a free gain absorbing the unknown inertia and thrust scale. The
best candidate gives `L_hat`. The base controller flies for the first 0.5 s;
then the policy is enabled unless `L_hat` exceeds the training maximum, and an
env that trips stays on the base controller (latching). The threshold is read
from the policy's training config, not tuned. Calibration on seeds disjoint
from all evaluations (`make supervisor`): lag error 2.0 ms (p90 4.1 ms) after
0.5 s; every airframe more than 5 ms beyond the boundary detected; 2.7 % false
alarms inside it, all near the boundary. It needs excitation: on a pure hover
task the error is about twice as large.

Because the lag variable was chosen after looking at the seed-12345 stress
results, the stress test is re-run on a new seed (54321), with and without the
supervisor. Pre-registered criteria:

- **S1** stress: pooled crash rate falls by ≥ 80 %;
- **S2** in distribution: fallback ≤ 5 % and IQM accuracy cost ≤ 0.01 m;
- **S3** stress: the supervised policies still beat `robust` (Δ CI below 0).

**Learning curve over all 10 seeds.** Checkpoints 100–999 of every run on the
standard evaluation; reports Δ vs `robust` per checkpoint, residual − direct
with CI, and the iterations each method needs to reach Δ ≤ −0.15 m.

About 190 Isaac evaluations of one process each (the baselines are cached);
`JOBS=2` runs two at a time.

### Follow-up results (full tables: `results/isaac_report.md`)

**Residual RL is about twice as sample-efficient.** Iterations until Δ vs
`robust` ≤ −0.15 m: residual 283 [233, 300] (every run by 300), direct 533
[433, 700] (one run never); difference −250 [−433, −133], significant.

| iteration (samples) | 100 (10 M) | 200 (20 M) | 300 (29 M) | 400 (39 M) | 600 (59 M) | 999 (98 M) |
|---|---|---|---|---|---|---|
| residual Δ (m) | +0.58 | −0.12 | −0.21 | −0.23 | −0.24 | −0.22 |
| direct Δ (m) | +0.47 | +0.25 | +0.08 | −0.12 | −0.21 | −0.22 |
| residual − direct | +0.11 [−0.31, +0.63] | **−0.38** [−0.73, −0.19] | **−0.29** [−0.37, −0.20] | **−0.12** [−0.17, −0.06] | **−0.02** [−0.07, −0.00] | −0.01 [−0.04, +0.03] |
| HF × residual / direct | 1.14 / 2.29 | 1.42 / 2.33 | 1.74 / 2.20 | 1.87 / 2.44 | 1.85 / 2.57 | 1.80 / 2.67 |

Both start worse than `robust` (an untrained residual actively hurts its
base). The advantage is transient: from ~800 iterations the two are
indistinguishable. Chatter rises with accuracy; no checkpoint both beats
`robust` and stays within the 1.2× smoothness criterion (closest: residual at
200 iterations, −0.12 m at 1.42×).

**The supervisor meets all three pre-registered criteria, for both methods**
(stress on the held-out seed 54321):

| | stress crashes: none → supervisor | stress Δ vs robust (supervised) | in-distribution fallback | in-distribution cost |
|---|---|---|---|---|
| direct | 54 → 4 / 1000 (−93 %) | −0.143 [−0.149, −0.119] m | 2.3 % | +0.0065 [+0.0036, +0.0091] m |
| residual | 30 → 2 / 1000 (−93 %) | −0.145 [−0.152, −0.124] m | 2.3 % | +0.0091 [+0.0069, +0.0115] m |

Lag estimate at the gate decision: MAE 1.9 ms (p90 3.7 ms), as in the CPU
calibration. Every crash beyond the training lag disappears (0 / 300 above
76 ms); the few that remain (4 and 2) all have a lag *inside* the training
range but a delay outside it — the case the lag gate does not cover, as
predicted by the sim_lite result. Residual's S2 cost passes on its IQM
(0.0091 ≤ 0.01) but its CI reaches 0.0115: the margin is thin. The cost is
mostly the 2.3 % false alarms (lags within a few ms of the boundary flown by
`robust`) plus the 0.5 s identification window.

## Isaac Lab

**Layout.** The Isaac env is thin glue: it reads the rigid-body state from
PhysX and writes back a body wrench, reset poses and randomized mass/inertia.
Everything else — references, randomization, actuator delay + motor lag, drag
and wind, the baseline, the policy-to-command mapping, observation, reward,
termination — is `envs/torch_core.py`, which calls the *same* functions as
sim_lite (`task.py`, the backend-agnostic `GeometricController` and
trajectories, `actuate_torch`). With sim_lite's integrator standing in for
PhysX, the core reproduces sim_lite step for step (`test_torch_core_reproduces_sim_lite_step_for_step`),
and statistically under full randomization (`scripts/check_torch_core.py`).
The airframe is a 0.85 kg box sized so its uniform-density inertia equals the
nominal one; rotors exist only in the wrench model, exactly as in sim_lite.
Control at 50 Hz in `_pre_physics_step`; actuator, gusts and drag every
physics step (250 Hz) in `_apply_action`. The OOD supervisor runs in the core
(`test_torch_core_supervisor_matches_sim_lite`). Not ported: the safety layer.

**Install** (Isaac Sim 5.1 + Isaac Lab 2.3 need Python 3.11, Ubuntu 22.04,
≥ 16 GB VRAM, NVIDIA driver ≥ 580.65):

```bash
conda create -n isaaclab python=3.11 -y && conda activate isaaclab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && git checkout v2.3.2
./isaaclab.sh --install                                  # isaaclab + isaaclab_rl (rsl_rl)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py   # smoke test; accept the EULA
pip install pyyaml stable-baselines3                     # for sim-to-sim evaluation
```

**Run** (from this repo, inside the `isaaclab` env; free the GPU first —
Isaac Sim with thousands of envs wants most of a 24 GB card):

```bash
python3 scripts/check_torch_core.py --policy results/direct_s0.zip   # CPU parity, no Isaac needed
python3 scripts/train_isaac.py --mode direct --num_envs 64 --max_iterations 5 --headless   # smoke test
python3 scripts/train_isaac.py --mode direct --seed 0 --headless      # 4096 envs
python3 scripts/eval_isaac.py --rsl results/isaac/<run>/model_<it>.pt --mode direct --headless
python3 scripts/eval_isaac.py --sb3 results/direct_s0.zip --headless  # sim-to-sim: sim_lite policy in PhysX
python3 scripts/eval_isaac.py --rsl ... --mode residual --supervisor  # with the OOD supervisor
python3 scripts/isaac_suite.py run ood curve --jobs 2                  # = make isaac-suite JOBS=2
python3 scripts/isaac_suite.py report                                  # -> results/isaac_report.md
```

`eval_isaac.py` writes the same JSON as `eval.py` (aggregate it the same way).
The sim-to-sim run is the first thing to look at: the policy never saw PhysX,
so any gap there is a gap between the two simulators, not a training effect.

## To finish

1. Optional: a delay-aware gate that keeps the lag gate's low false-alarm
   rate (the remaining OOD crashes are lag-in-range / delay-out-of-range).
2. Optional: identify a real airframe with `sysid.py` and narrow the
   randomization ranges to match it.
