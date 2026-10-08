"""Core numerical checks.

Run with ``python tests/test_core.py`` (no pytest needed) or ``pytest tests -q``.
These cover the physics, the actuator/latency model, the baseline controller
(presets, integral, information parity), randomization, the safety layer,
sysid, evaluation metrics and config loading -- all on CPU with
NumPy alone (one optional check also needs torch). No Isaac Sim, no GPU.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import numpy as np  # noqa: E402

from quad_residual.backend import quat_mul, quat_normalize, quat_rotate  # noqa: E402
from quad_residual.controllers.geometric import (  # noqa: E402
    GeometricController, GeometricGains)
from quad_residual.dynamics.quadrotor import (  # noqa: E402
    QuadParams, QuadrotorModel, mixing_matrix)


def _ref(n, pos, yaw=0.0):
    return {
        "pos": np.tile(np.asarray(pos, np.float32), (n, 1)),
        "vel": np.zeros((n, 3), np.float32),
        "acc": np.zeros((n, 3), np.float32),
        "yaw": np.full((n,), np.float32(yaw)),
    }


def test_mixing_invertible():
    M = mixing_matrix(0.15, 0.016)
    assert abs(np.linalg.det(M)) > 1e-6
    assert np.allclose(M @ np.linalg.inv(M), np.eye(4), atol=1e-9)


def test_quat_rotate_identity_and_90deg():
    q = np.array([[1.0, 0, 0, 0]], np.float32)
    v = np.array([[1.0, 2.0, 3.0]], np.float32)
    assert np.allclose(quat_rotate(q, v), v, atol=1e-6)
    s = np.float32(np.sqrt(0.5))
    qz = np.array([[s, 0, 0, s]], np.float32)
    out = quat_rotate(qz, np.array([[1.0, 0, 0]], np.float32))
    assert np.allclose(out, [[0, 1, 0]], atol=1e-5), out


def test_quat_mul_norm_preserved():
    rng = np.random.default_rng(0)
    a = quat_normalize(rng.normal(size=(8, 4)).astype(np.float32))
    b = quat_normalize(rng.normal(size=(8, 4)).astype(np.float32))
    assert np.allclose(np.linalg.norm(quat_mul(a, b), axis=-1), 1.0, atol=1e-5)


def test_hover_command_holds_altitude():
    n = 4
    m = QuadrotorModel(QuadParams(), n)
    st = m.zero_state()
    cmd = np.tile(m.hover_command(), (1, 4)).astype(np.float32)
    for _ in range(500):
        st = m.step(st, cmd)
    assert np.abs(st.pos[:, 2]).max() < 0.02, st.pos[:, 2]
    assert np.abs(st.vel).max() < 0.05


def test_thrust_above_hover_climbs():
    m = QuadrotorModel(QuadParams(), 2)
    st = m.zero_state()
    cmd = np.tile(m.hover_command() * 1.2, (1, 4)).astype(np.float32)
    for _ in range(250):
        st = m.step(st, cmd)
    assert (st.pos[:, 2] > 0.3).all(), st.pos[:, 2]


def test_motor_lag_is_first_order():
    p = QuadParams(motor_tau=0.05, actuation_delay_steps=1)
    m = QuadrotorModel(p, 1)
    st = m.zero_state()
    st.motor[:] = 0.0
    st.delay_buf[:] = 0.0
    target = np.full((1, 4), 0.5, np.float32)
    for _ in range(int(p.motor_tau / p.dt) + 1):
        st = m.step(st, target)
    frac = st.motor.mean() / 0.5
    assert 0.5 < frac < 0.75, frac


def test_actuation_delay_blocks_first_steps():
    p = QuadParams(actuation_delay_steps=3)
    m = QuadrotorModel(p, 1)
    st = m.zero_state()
    hover = st.motor.copy()
    st = m.step(st, np.ones((1, 4), np.float32))
    assert np.allclose(st.motor, hover, atol=2e-3), "command applied too early"


def test_controller_hovers_in_place():
    n = 8
    p = QuadParams()
    m = QuadrotorModel(p, n)
    c = GeometricController(p, GeometricGains(), n)
    st = m.zero_state()
    ref = _ref(n, [0.0, 0.0, 0.0])
    for _ in range(750):
        st = m.step(st, c(st, ref, m.mass, m.inertia, m.max_thrust))
    assert np.linalg.norm(st.pos, axis=-1).max() < 0.05, st.pos


def test_controller_tracks_step():
    n = 4
    p = QuadParams()
    m = QuadrotorModel(p, n)
    c = GeometricController(p, GeometricGains(), n)
    st = m.zero_state()
    ref = _ref(n, [1.0, -0.5, 1.5])
    for _ in range(1500):
        st = m.step(st, c(st, ref, m.mass, m.inertia, m.max_thrust))
    err = np.linalg.norm(st.pos - ref["pos"], axis=-1)
    assert err.max() < 0.12, (st.pos, err)


def test_controller_rejects_initial_tilt():
    n = 4
    p = QuadParams()
    m = QuadrotorModel(p, n)
    c = GeometricController(p, GeometricGains(), n)
    st = m.zero_state()
    ang = np.float32(0.25)
    st.quat[:] = np.array([np.cos(ang / 2), np.sin(ang / 2), 0, 0], np.float32)
    ref = _ref(n, [0.0, 0.0, 0.0])
    for _ in range(1000):
        st = m.step(st, c(st, ref, m.mass, m.inertia, m.max_thrust))
    assert np.abs(st.quat[:, 1]).max() < 0.03, st.quat


# ---------------------------------------------------------------- yaw sign
def _yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def test_yaw_reaction_opposes_spin():
    mix = mixing_matrix(0.15, 0.016)
    ccw = mix @ np.array([1.0, 1.0, 0.0, 0.0])     # rotors 0, 1 spin CCW
    cw = mix @ np.array([0.0, 0.0, 1.0, 1.0])      # rotors 2, 3 spin CW
    assert ccw[3] < 0 < cw[3], (ccw, cw)


def test_controller_tracks_yaw():
    n = 2
    p = QuadParams()
    m = QuadrotorModel(p, n)
    c = GeometricController(p, GeometricGains(), n)
    st = m.zero_state()
    ref = _ref(n, [0.0, 0.0, 0.0], yaw=0.6)
    for _ in range(1000):
        st = m.step(st, c(st, ref, m.mass, m.inertia, m.max_thrust))
    assert np.abs(_yaw(st.quat) - 0.6).max() < 0.02, _yaw(st.quat)


# ---------------------------------------------------------------- latency
def test_per_env_delay():
    from quad_residual.dynamics.actuator import actuate
    buf = np.zeros((3, 3, 1), np.float32)
    motor = np.zeros((3, 1), np.float32)
    delays = np.array([0, 1, 3])
    first_move = [None] * 3
    for k in range(6):
        motor, buf = actuate(buf, motor, np.ones((3, 1), np.float32), delays, 1.0)
        for i in range(3):
            if first_move[i] is None and motor[i, 0] > 0:
                first_move[i] = k
    assert first_move == [0, 1, 3], first_move


def test_zero_delay_plant():
    p = QuadParams(actuation_delay_steps=0, motor_tau=1e-9)
    m = QuadrotorModel(p, 1)
    st = m.zero_state()
    st = m.step(st, np.full((1, 4), 0.9, np.float32))
    assert np.allclose(st.motor, 0.9), st.motor


def test_latency_randomization_reaches_plant():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    env = QuadTrackEnv(EnvCfg(num_envs=256, seed=1))
    env.set_curriculum(1.0)
    env.reset()
    d = env.model.delay_steps
    assert env.model.max_delay == 4 and env.state.delay_buf.shape[1] == 4
    assert set(np.unique(d)) == {1, 2, 3, 4}, np.unique(d)


def test_curriculum_zero_is_nominal():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    env = QuadTrackEnv(EnvCfg(num_envs=16, seed=2))
    env.set_curriculum(0.0)
    env.reset()
    p = env.p
    assert np.allclose(env.model.mass, p.mass)
    assert np.allclose(env.model.motor_tau, p.motor_tau)
    assert np.allclose(env.model.drag, p.drag_coeff)
    assert np.allclose(env.model.max_thrust, p.max_thrust_per_rotor)
    assert np.allclose(env.model.wind, 0.0)
    assert (env.model.delay_steps == p.actuation_delay_steps).all()


# ---------------------------------------------------------------- safety
def test_safety_reset_seeds_hover():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    env = QuadTrackEnv(EnvCfg(num_envs=4, seed=3, safety=True, randomize=False))
    for _ in range(5):
        env.step(np.zeros((4, 4), np.float32))
    env.reset(np.array([1]))                     # mid-run auto-reset of one env
    hover = env.model.hover_command()[1, 0]
    env.step(np.zeros((4, 4), np.float32))
    # Rate limit is 0.16 per step; seeded at 0 this could not exceed 0.16.
    assert env.last_cmd[1].min() > hover - 0.2, (env.last_cmd[1], hover)


# ---------------------------------------------------------------- sysid
def test_sysid_distinguishes_zero_and_one_delay():
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "sysid.py")
    spec = importlib.util.spec_from_file_location("sysid", path)
    sysid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sysid)
    rng = np.random.default_rng(0)
    cmd = (rng.random(600) < 0.03).cumsum() % 2 * 0.4 + 0.3
    for true_d in (0, 1):
        meas = sysid.simulate(0.03, true_d, cmd, 0.004) + rng.normal(0, 0.002, 600)
        out = sysid.identify(cmd, meas, 0.004, delay_grid=range(0, 4))
        assert out["delay_steps"] == true_d, (true_d, out)


# ---------------------------------------------------------------- configs
def test_configs_drive_the_env():
    from quad_residual import config as C
    cfg = C.load(overrides=["env.reward.w_smooth=0.0",
                            "domain_rand.latency.extra_delay_steps=[0,0]"])
    ec = C.env_cfg(cfg, num_envs=2)
    assert ec.w_smooth == 0.0 and ec.rand_cfg.extra_delay_steps == (0, 0)
    from quad_residual.sim_lite.env import QuadTrackEnv
    assert QuadTrackEnv(ec).model.max_delay == 1
    for bad in (["env.reward.w_smoth=0"], ["nope.x=1"]):
        try:
            C.load(overrides=bad)
        except KeyError:
            continue
        raise AssertionError(f"override {bad} was accepted")


# ---------------------------------------------------------------- baseline presets / fairness
def _load_script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fast_preset_is_original_tuning():
    assert GeometricGains.preset("fast", integral=False) == GeometricGains()


def test_robust_preset_satisfies_selection_rule():
    from quad_residual import config as C
    from quad_residual.controllers.geometric import BASELINE_PRESETS
    sel = _load_script("select_baseline_gains")
    p = QuadParams()
    cs = sel.corners(C.rand_cfg(C.load()), p)
    ok_robust, _ = sel.check(BASELINE_PRESETS["robust"], cs, p, integral=True)
    ok_fast, _ = sel.check(BASELINE_PRESETS["fast"], cs, p, integral=True)
    assert ok_robust.all(), f"robust preset unstable at {np.sum(~ok_robust)} corners"
    assert not ok_fast.all()          # the trade-off the comparison is about


def test_integral_removes_mass_offset():
    p = QuadParams()
    out = {}
    for integral in (False, True):
        m = QuadrotorModel(p, 1)
        m.mass[:] = 1.0                                   # controller assumes 0.85 kg
        g = GeometricGains.preset("robust", integral=integral)
        c = GeometricController(p, g, 1, dt=p.dt * 5)
        st = m.zero_state()
        st.pos[:] = [0.0, 0.0, 1.5]
        nom = (np.full((1, 1), p.mass, np.float32), np.asarray([p.inertia], np.float32),
               np.full((1, 1), p.max_thrust_per_rotor, np.float32))
        for k in range(int(25.0 / p.dt)):
            if k % 5 == 0:
                cmd = c(st, _ref(1, [0.0, 0.0, 1.5]), *nom)
            st = m.step(st, cmd)
        out[integral] = abs(st.pos[0, 2] - 1.5)
    assert out[False] > 0.2 and out[True] < 0.01, out


def test_hover_warm_start_holds_true_hover():
    p = QuadParams()
    m_true, t_true = 1.0, 0.9 * p.max_thrust_per_rotor
    c = GeometricController(p, GeometricGains.preset("robust"), 1, dt=p.dt * 5)
    c.reset(None, hover_ratio=[(m_true / p.mass) * (p.max_thrust_per_rotor / t_true)])
    st = QuadrotorModel(p, 1).zero_state()
    st.pos[:] = [0.0, 0.0, 1.5]
    cmd = c(st, _ref(1, [0.0, 0.0, 1.5]), np.full((1, 1), p.mass, np.float32),
            np.asarray([p.inertia], np.float32), np.full((1, 1), p.max_thrust_per_rotor, np.float32))
    thrust = cmd.sum() * t_true
    assert abs(thrust - m_true * p.gravity) < 0.05, thrust


def test_baseline_uses_nominal_params_not_true_ones():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    cmds = {}
    for info in ("nominal", "privileged"):
        env = QuadTrackEnv(EnvCfg(num_envs=2, seed=5, randomize=False, baseline_info=info))
        ref = env._reference()
        a = env._baseline_command(ref).copy()
        env.ctrl.reset()
        env.model.mass[:] *= 1.2
        env.model.max_thrust[:] *= 0.9
        b = env._baseline_command(ref)
        cmds[info] = np.abs(a - b).max()
    assert cmds["nominal"] < 1e-6 and cmds["privileged"] > 1e-3, cmds


def test_policy_and_baseline_share_one_measurement():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    env = QuadTrackEnv(EnvCfg(num_envs=4, seed=6))
    env.set_curriculum(1.0)
    obs = env.reset()
    env.state.pos[0] += 10.0                          # force a termination + auto-reset of env 0
    obs, _, term, _, _ = env.step(np.zeros((4, 4), np.float32))
    assert term[0]
    ref = env._reference()
    assert np.allclose(obs[:, 0:3], ref["pos"] - env.meas.pos, atol=1e-5)
    assert np.allclose(obs[:, 3:6], ref["vel"] - env.meas.vel, atol=1e-5)
    assert not np.allclose(env.meas.pos, env.state.pos)   # noise is really on


# ---------------------------------------------------------------- evaluation
def test_metrics_ignore_samples_after_a_crash():
    from quad_residual.utils.metrics import episode_metrics, paired
    T, dt = 500, 0.02
    err = np.full((T, 3), 0.1)
    err[:, 1] = 0.2
    err[300:, 2] = 50.0                               # what an auto-reset would leak
    crashed = np.array([False, False, True])
    cmd = np.full((T, 3, 4), 0.4)
    om = np.zeros((T, 3, 3))
    ep = episode_metrics(err, cmd, om, crashed, dt, 0.035)
    assert np.isnan(ep["rmse_m"][2]) and np.isclose(ep["rmse_m"][0], 0.1)
    ref = dict(ep)
    ref["rmse_settled_m"] = np.array([0.15, 0.15, np.nan])
    ref["crashed"] = np.array([False, True, False])
    pr = paired(ep, ref)
    assert pr["both_survived"] == 1 and pr["only_this_crashed"] == 1 and pr["only_ref_crashed"] == 1
    assert np.isclose(pr["delta_mean"], -0.05)


def test_high_frequency_rms():
    from quad_residual.utils.metrics import hf_rms
    t = np.arange(500) * 0.02
    slow = 0.1 * np.sin(2 * np.pi * 1.0 * t)[:, None].repeat(4, 1)
    fast = 0.1 * np.sin(2 * np.pi * 10.0 * t)[:, None].repeat(4, 1)
    fc = 1.0 / (2 * np.pi * 0.035)
    assert hf_rms(np.full((500, 4), 0.4), 50.0, fc) < 1e-9
    assert hf_rms(slow, 50.0, fc) < 1e-3
    assert abs(hf_rms(fast, 50.0, fc) - 0.1 / np.sqrt(2)) < 2e-3


def test_eval_conditions_come_from_configs():
    import tempfile
    import yaml
    from quad_residual import config as C
    cfg = C.load()
    cfg["domain_rand"]["latency"]["extra_delay_steps"] = [0, 0]   # e.g. a no-delay ablation
    with tempfile.TemporaryDirectory() as d:
        snap = os.path.join(d, "m.cfg.yaml")
        C.save(cfg, snap)
        assert C.load_for_eval(snap)["domain_rand"]["latency"]["extra_delay_steps"] == [0, 3]
        assert C.load_for_eval(snap, train_dist=True)["domain_rand"]["latency"]["extra_delay_steps"] == [0, 0]
        old = dict(cfg)
        old["env"] = {k: v for k, v in cfg["env"].items() if k != "baseline"}
        with open(snap, "w") as fh:
            yaml.safe_dump(old, fh)
        try:
            C.load_for_eval(snap)
        except SystemExit:
            pass
        else:
            raise AssertionError("pre-change snapshot was accepted")


def test_waypoints_are_per_episode_bounded_and_paired():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    envs = [QuadTrackEnv(EnvCfg(num_envs=8, seed=7, trajectory="waypoints", policy_mode=m))
            for m in ("baseline", "direct")]
    wp = envs[0].traj._wp
    assert np.allclose(wp, envs[1].traj._wp)          # same episodes for every controller
    assert np.allclose(wp[:, 0], [0.0, 0.0, 1.5])     # hover at the start point first
    steps = np.linalg.norm(np.diff(wp, axis=1), axis=-1)
    assert steps.max() <= 1.5 + 1e-5                  # never beyond the termination limit
    assert not np.allclose(wp[0], wp[1])              # each episode gets its own walk
    old = wp[3].copy()
    envs[0].reset(np.array([3]))
    assert not np.allclose(envs[0].traj._wp[3], old) and np.allclose(envs[0].traj._wp[2], wp[2])


def test_experiment_registry_is_consistent():
    exp = _load_script("experiments")
    for name, e in exp.EXPERIMENTS.items():
        assert e["group"] in exp.GROUPS, name
        if "from_" in e:
            assert e["from_"] in exp.EXPERIMENTS and "from_" not in exp.EXPERIMENTS[e["from_"]]
        else:
            assert e["mode"] in ("residual", "direct") and e["ckpt"]
    from quad_residual import config as C
    for e in exp.EXPERIMENTS.values():            # every training override must be a real key
        C.load(overrides=e.get("train", []))


def test_eval_records_pre_reset_state_at_truncation():
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    env = QuadTrackEnv(EnvCfg(num_envs=2, seed=9, randomize=False, episode_seconds=0.1))
    for _ in range(env.max_steps - 1):
        env.step(np.zeros((2, 4), np.float32))
    before = env.state.pos.copy()
    _, _, term, trunc, info = env.step(np.zeros((2, 4), np.float32))
    assert trunc.all() and not term.any()
    # env.state is the next episode's spawn; final_state is where this one ended
    assert np.abs(info["final_state"].pos - before).max() < 0.05
    assert np.abs(env.state.pos - info["final_state"].pos).max() > 1e-3


# ---------------------------------------------------------------- torch twins (Isaac path)
def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise Skip("torch not installed")


def test_controller_numpy_and_torch_agree():
    torch = _torch()
    from types import SimpleNamespace as NS
    p = QuadParams()
    n, rng = 32, np.random.default_rng(0)
    q = rng.normal(0, 0.3, (n, 4)).astype(np.float32)
    q[:, 0] += 1
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    st = dict(pos=rng.normal(0, 1, (n, 3)), vel=rng.normal(0, 1, (n, 3)), quat=q,
              omega=rng.normal(0, 1, (n, 3)))
    st = {k: np.asarray(v, np.float32) for k, v in st.items()}
    ref = {k: rng.normal(0, 1, (n, 3)).astype(np.float32) for k in ("pos", "vel", "acc")}
    ref["yaw"] = rng.uniform(-1, 1, n).astype(np.float32)
    args = [np.full((n, 1), 0.9, np.float32), np.tile(np.float32(p.inertia), (n, 1)),
            np.full((n, 1), 5.0, np.float32)]
    T = lambda a: torch.tensor(a)
    for preset in ("fast", "robust"):
        cn = GeometricController(p, GeometricGains.preset(preset), n, dt=0.02)
        ct = GeometricController(p, GeometricGains.preset(preset), n, dt=0.02)
        hr = rng.uniform(0.8, 1.3, n).astype(np.float32)
        cn.reset(None, hover_ratio=hr)
        ct.reset(None, hover_ratio=T(hr))
        for _ in range(5):
            a = cn(NS(**st), ref, *args)
            b = ct(NS(**{k: T(v) for k, v in st.items()}), {k: T(v) for k, v in ref.items()},
                   *[T(x) for x in args])
            assert np.abs(a - b.numpy()).max() < 1e-5


def test_trajectories_numpy_and_torch_agree():
    torch = _torch()
    from quad_residual.tasks import trajectories as TR
    t = np.linspace(0, 10, 41).astype(np.float32)
    for name in ("hover", "lemniscate", "waypoints"):
        a = TR.make(name).sample(t, np.random.default_rng(3))
        b = TR.make(name).sample(torch.tensor(t), np.random.default_rng(3))
        for k in a:
            assert np.abs(a[k] - b[k].numpy()).max() < 1e-5, (name, k)


def test_torch_randomizer_matches_ranges_and_curriculum():
    torch = _torch()
    from quad_residual.dynamics.randomization import RandomizationCfg
    from quad_residual.dynamics.randomization_torch import TorchDomainRandomizer
    p, cfg = QuadParams(), RandomizationCfg()
    r = TorchDomainRandomizer(cfg, p, 4000, "cpu", seed=1)
    r.apply(torch.arange(4000))
    assert cfg.mass_range[0] <= r.mass.min() and r.mass.max() <= cfg.mass_range[1]
    assert abs(float(r.mass.mean()) - np.mean(cfg.mass_range)) < 0.01
    assert set(r.delay_steps.unique().tolist()) == {1, 2, 3, 4}
    w = r.wind_mean.norm(dim=-1)
    assert w.max() <= cfg.wind_range[1] + 1e-5 and abs(float(w.mean()) - 1.5) < 0.1
    r.set_scale(0.0)
    r.apply(torch.arange(4000))
    assert torch.allclose(r.mass, torch.full_like(r.mass, p.mass))
    assert (r.delay_steps == p.actuation_delay_steps).all() and float(r.wind_mean.abs().max()) == 0.0


def test_torch_core_reproduces_sim_lite_step_for_step():
    """The Isaac env's task logic (torch core + the same integrator PhysX
    replaces) follows sim_lite exactly when given the same parameters."""
    torch = _torch()
    from types import SimpleNamespace as NS
    from quad_residual.envs.torch_core import TorchQuadCore, rigid_body_step
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    from quad_residual.dynamics.randomization import RandomizationCfg
    quiet = RandomizationCfg(obs_noise_pos=0.0, obs_noise_vel=0.0, obs_noise_omega=0.0,
                             gust_std=0.0, obs_bias_pos=0.0)
    for mode in ("baseline", "residual", "direct"):
        n = 6
        ec = EnvCfg(num_envs=n, seed=3, policy_mode=mode, rand_cfg=quiet)
        env = QuadTrackEnv(ec)
        env.set_curriculum(1.0)
        obs_n = env.reset()
        core = TorchQuadCore(ec, n, "cpu", RandomizationCfg(**vars(quiet)), seed=3)
        core.set_curriculum(1.0)
        core.reset(torch.arange(n))
        m, r = env.model, core.rand
        for a, b in [(r.mass, m.mass), (r.inertia, m.inertia), (r.max_thrust, m.max_thrust),
                     (r.motor_tau, m.motor_tau), (r.drag, m.drag)]:
            a[:] = torch.tensor(b)
        r.delay_steps[:] = torch.tensor(m.delay_steps)
        r.wind_mean[:] = torch.tensor(env.rand.wind)
        core.motor[:] = torch.tensor(env.state.motor)
        core.delay_buf[:] = torch.tensor(env.state.delay_buf)
        core.ctrl.reset(None, hover_ratio=torch.tensor(
            (m.mass[:, 0] / 0.85) * (5.0 / m.max_thrust[:, 0])))
        st = NS(**{k: torch.tensor(getattr(env.state, k)) for k in ("pos", "vel", "quat", "omega")})
        obs_t = core.observe(st)
        assert np.abs(obs_n - obs_t.numpy()).max() < 1e-5
        rng = np.random.default_rng(0)
        alive = np.ones(n, bool)        # sim_lite auto-resets crashed envs; the core does not
        for _ in range(40):
            if mode == "direct":        # near hover, so the comparison is not all crashes
                a = (2.0 * m.hover_command() - 1.0) + 0.05 * rng.normal(size=(n, 4))
            else:
                a = rng.uniform(-1, 1, (n, 4))
            a = np.clip(a, -1, 1).astype(np.float32)
            obs_n, rew_n, term_n, _, info = env.step(a)
            core.control(torch.tensor(a), st)
            for _ in range(ec.decimation):
                f, tq = core.wrench(st)
                st = rigid_body_step(st, f, tq, r.mass, r.inertia, core.p.dt, core.p.gravity)
            rew_t, _ = core.reward(st)
            term_t = core.terminated(st).numpy()
            obs_t = core.observe(st)
            k = alive & ~term_n
            assert np.array_equal(term_n[alive], term_t[alive]), mode
            assert np.abs(info["final_state"].pos[alive] - st.pos.numpy()[alive]).max() < 1e-4, mode
            assert np.abs(rew_n[alive] - rew_t.numpy()[alive]).max() < 1e-4, mode
            if k.any():
                assert np.abs(obs_n[k] - obs_t.numpy()[k]).max() < 1e-3, mode
            alive = k
        assert alive.any()


# --------------------------------------------------------------- OOD supervisor
def _support(lag_max, delay_max=4, tau_max=0.06):
    return {"lag_max": lag_max, "delay_max": delay_max, "tau_max_train": tau_max}


def test_supervisor_threshold_comes_from_training_config():
    from quad_residual import config as C
    from quad_residual.controllers.supervisor import training_support
    s = training_support(QuadParams(), C.rand_cfg(C.load()))
    assert s["delay_max"] == 4 and abs(s["tau_max_train"] - 0.06) < 1e-9
    assert abs(s["lag_max"] - 0.076) < 1e-9
    s0 = training_support(QuadParams(), C.rand_cfg(
        C.load(overrides=["domain_rand.latency.extra_delay_steps=[0,0]"])))
    assert s0["delay_max"] == 1 and abs(s0["lag_max"] - 0.064) < 1e-9


def test_supervisor_recovers_effective_lag_in_closed_loop():
    """Classical controller flying, delay 1-10 steps: the lag estimate after the
    0.5 s identification window is within a few ms."""
    _torch()
    from quad_residual import config as C
    from quad_residual.controllers.supervisor import LagSupervisor, SupervisorCfg
    from quad_residual.sim_lite.env import QuadTrackEnv
    conf = C.load(overrides=["domain_rand.latency.extra_delay_steps=[0,9]"])
    env = QuadTrackEnv(C.env_cfg(conf, num_envs=64, policy_mode="baseline", seed=901))
    env.set_curriculum(1.0)
    env.reset()
    sup = LagSupervisor(SupervisorCfg(enabled=True), env.p, 64, env.cfg.decimation, _support(0.076))
    sup.reset(None, env.meas.omega)
    for _ in range(sup.probe_steps):
        env.step(np.zeros((64, 4), np.float32))
        sup.update(env.last_cmd, env.meas.omega)
    lag = env.model.delay_steps * env.p.dt + env.model.motor_tau[:, 0]
    err = np.abs(sup.lag_hat.numpy() - lag)
    assert err.mean() < 0.004 and np.percentile(err, 90) < 0.008, err.mean()


def test_supervisor_fallback_reproduces_the_base_controller():
    """If every env trips, the residual policy's actions never reach the motors:
    the rollout must equal the base controller's exactly; before the probe ends
    the same holds for every env."""
    _torch()
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    n, rng = 8, np.random.default_rng(1)
    base = QuadTrackEnv(EnvCfg(num_envs=n, seed=5, policy_mode="baseline"))
    sup = QuadTrackEnv(EnvCfg(num_envs=n, seed=5, policy_mode="residual", supervisor=True,
                              supervisor_support=_support(lag_max=0.0)))
    for env in (base, sup):
        env.set_curriculum(1.0)
        env.reset()
    for k in range(60):
        a = rng.uniform(-1, 1, (n, 4)).astype(np.float32)
        base.step(a)
        sup.step(a)
        assert np.allclose(base.last_cmd, sup.last_cmd, atol=1e-6), k
    assert sup.sup.tripped.all()
    assert np.allclose(sup.prev_action, 0.0)          # what the policy is told it applied


def test_torch_core_supervisor_matches_sim_lite():
    """Same gating decisions and lag estimates in the Isaac core as in sim_lite."""
    torch = _torch()
    from types import SimpleNamespace as NS
    from quad_residual.envs.torch_core import TorchQuadCore, rigid_body_step
    from quad_residual.sim_lite.env import EnvCfg, QuadTrackEnv
    from quad_residual.dynamics.randomization import RandomizationCfg
    quiet = RandomizationCfg(obs_noise_pos=0.0, obs_noise_vel=0.0, obs_noise_omega=0.0,
                             gust_std=0.0, obs_bias_pos=0.0, extra_delay_steps=(0, 8))
    n = 8
    for mode in ("residual", "direct"):
        ec = EnvCfg(num_envs=n, seed=4, policy_mode=mode, rand_cfg=quiet, supervisor=True,
                    supervisor_probe_s=0.2, supervisor_support=_support(0.06))
        env = QuadTrackEnv(ec)
        env.set_curriculum(1.0)
        env.reset()
        core = TorchQuadCore(ec, n, "cpu", RandomizationCfg(**vars(quiet)), seed=4)
        core.set_curriculum(1.0)
        core.reset(torch.arange(n))
        m, r = env.model, core.rand
        for a, b in [(r.mass, m.mass), (r.inertia, m.inertia), (r.max_thrust, m.max_thrust),
                     (r.motor_tau, m.motor_tau), (r.drag, m.drag)]:
            a[:] = torch.tensor(b)
        r.delay_steps[:] = torch.tensor(m.delay_steps)
        r.wind_mean[:] = torch.tensor(env.rand.wind)
        core.motor[:] = torch.tensor(env.state.motor)
        core.delay_buf[:] = torch.tensor(env.state.delay_buf)
        core.ctrl.reset(None, hover_ratio=torch.tensor(
            (m.mass[:, 0] / 0.85) * (5.0 / m.max_thrust[:, 0])))
        st = NS(**{k: torch.tensor(getattr(env.state, k)) for k in ("pos", "vel", "quat", "omega")})
        core.observe(st)
        rng = np.random.default_rng(0)
        alive = np.ones(n, bool)
        for k in range(40):
            if mode == "direct":
                a = (2.0 * m.hover_command() - 1.0) + 0.05 * rng.normal(size=(n, 4))
            else:
                a = 0.3 * rng.uniform(-1, 1, (n, 4))
            a = np.clip(a, -1, 1).astype(np.float32)
            allow_n = env.sup.allow().numpy().copy()
            _, _, term_n, _, _ = env.step(a)
            allow_t = core.sup.allow().numpy().copy()
            core.control(torch.tensor(a), st)
            for _ in range(ec.decimation):
                f, tq = core.wrench(st)
                st = rigid_body_step(st, f, tq, r.mass, r.inertia, core.p.dt, core.p.gravity)
            core.observe(st)
            assert np.array_equal(allow_n[alive], allow_t[alive]), (mode, k)
            assert np.abs(env.last_cmd[alive] - core.cmd.numpy()[alive]).max() < 1e-4, (mode, k)
            alive &= ~term_n
        lag_n, lag_t = env.sup.lag_hat.numpy(), core.sup.lag_hat.numpy()
        assert np.allclose(lag_n[alive], lag_t[alive], atol=1e-6), mode
        assert alive.any() and env.sup.tripped[alive].any() and (~env.sup.tripped[alive]).any(), mode


class Skip(Exception):
    pass


def test_torch_actuator_matches_numpy():
    try:
        import torch
    except ImportError:
        raise Skip("torch not installed")
    from quad_residual.dynamics.actuator import actuate, actuate_torch
    rng = np.random.default_rng(4)
    buf = rng.random((5, 3, 4)).astype(np.float32)
    motor = rng.random((5, 4)).astype(np.float32)
    delays = np.array([0, 1, 2, 3, 1])
    alpha = rng.random((5, 1)).astype(np.float32)
    tb, tm = torch.tensor(buf), torch.tensor(motor)
    for _ in range(6):
        cmd = rng.random((5, 4)).astype(np.float32)
        motor, buf = actuate(buf, motor, cmd, delays, alpha)
        tm, tb = actuate_torch(tb, tm, torch.tensor(cmd), torch.tensor(delays),
                               torch.tensor(alpha))
    assert np.allclose(motor, tm.numpy(), atol=1e-6)
    assert np.allclose(buf, tb.numpy(), atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS   {fn.__name__}")
        except Skip as exc:
            skipped += 1
            print(f"  SKIP   {fn.__name__}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL   {fn.__name__}: {str(exc)[:150]}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR  {fn.__name__}: {type(exc).__name__}: {str(exc)[:150]}")
    print(f"\n{len(fns) - failures - skipped}/{len(fns)} passed"
          + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failures else 0)
