"""CPU environment -- same task, same reward, same observation as the Isaac Lab
env, but running on the NumPy model in ``dynamics/quadrotor.py``.

Why this exists: Isaac Lab needs a GPU and a multi-gigabyte Isaac Sim install.
Reward shaping, observation design and curriculum logic do not. Getting those
wrong is the usual reason a run fails to converge, and finding out on a laptop
in seconds beats finding out on a cluster in hours. The Isaac Lab env then
reuses this file's reward and observation functions verbatim, so what is tuned
here is what trains there.

Vectorised: ``step`` advances all ``num_envs`` at once and auto-resets
terminated environments, which is the convention both SB3's ``VecEnv`` and
Isaac Lab's ``DirectRLEnv`` expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np

from ..controllers.geometric import GeometricController, GeometricGains
from ..controllers.safety import SafetyCfg, SafetyLayer
from ..controllers.supervisor import LagSupervisor, SupervisorCfg
from ..dynamics.quadrotor import QuadParams, QuadrotorModel
from ..dynamics.randomization import DomainRandomizer, RandomizationCfg
from ..tasks import trajectories
from .. import task


@dataclass
class EnvCfg:
    num_envs: int = 64
    policy_mode: str = "residual"      # "direct" | "residual" | "baseline"
    decimation: int = 5                # physics steps per policy step -> 50 Hz
    episode_seconds: float = 10.0
    trajectory: str = "lemniscate"
    residual_scale: float = 0.25       # max fraction of full command the policy adds
    # classical baseline (also the residual's base and the safety fallback)
    baseline_gains: str = "robust"     # key of controllers.geometric.BASELINE_PRESETS
    baseline_integral: bool = True     # PID position loop (False = original PD)
    baseline_info: str = "nominal"     # "nominal" | "privileged" -- see _baseline_command
    # reward weights
    w_pos: float = 1.0
    w_vel: float = 0.05
    w_rate: float = 0.02
    w_action: float = 0.01
    w_smooth: float = 0.03
    w_alive: float = 0.5
    # termination
    max_pos_err: float = 4.0
    max_tilt: float = 1.3
    seed: int = 0
    randomize: bool = True
    safety: bool = False               # off during training, on for deployment eval
    # OOD supervisor (controllers/supervisor.py): classical controller for the
    # first probe seconds, then the policy unless the estimated actuation lag
    # leaves the *training* distribution. Evaluation only.
    supervisor: bool = False
    supervisor_probe_s: float = 0.5
    supervisor_gate: str = "lag"              # lag | support
    supervisor_support: dict | None = None    # controllers.supervisor.training_support(...)
    rand_cfg: RandomizationCfg | None = None   # None -> RandomizationCfg() defaults


class QuadTrackEnv:
    """Vectorised quadrotor trajectory-tracking environment."""

    def __init__(self, cfg: EnvCfg | None = None, params: QuadParams | None = None):
        self.cfg = cfg or EnvCfg()
        self.p = params or QuadParams()
        n = self.cfg.num_envs

        rcfg = replace(self.cfg.rand_cfg or RandomizationCfg(),
                       enabled=self.cfg.randomize)
        self.rand = DomainRandomizer(rcfg, n, self.cfg.seed)
        self.model = QuadrotorModel(
            self.p, n, max_delay_steps=self.p.actuation_delay_steps + self.rand.max_extra_delay)
        self.dt = self.p.dt * self.cfg.decimation
        if self.cfg.baseline_info not in ("nominal", "privileged"):
            raise ValueError(f"baseline_info must be nominal|privileged, got {self.cfg.baseline_info!r}")
        self.ctrl = GeometricController(
            self.p, GeometricGains.preset(self.cfg.baseline_gains, self.cfg.baseline_integral),
            n, dt=self.dt)
        # What a real flight controller knows: the nominal airframe, not this
        # episode's randomized mass / inertia / thrust coefficient.
        self._nom_mass = np.full((n, 1), self.p.mass, np.float32)
        self._nom_inertia = np.tile(np.asarray(self.p.inertia, np.float32), (n, 1))
        self._nom_thrust = np.full((n, 1), self.p.max_thrust_per_rotor, np.float32)
        self.traj = trajectories.make(self.cfg.trajectory)
        self.safety = SafetyLayer(SafetyCfg(enabled=self.cfg.safety), n, self.dt)
        self.sup = None
        if self.cfg.supervisor:
            if self.cfg.supervisor_support is None:
                raise ValueError("supervisor_support must be the policy's *training* support")
            self.sup = LagSupervisor(
                SupervisorCfg(enabled=True, probe_s=self.cfg.supervisor_probe_s,
                              gate=self.cfg.supervisor_gate),
                self.p, n, self.cfg.decimation, self.cfg.supervisor_support)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.max_steps = int(self.cfg.episode_seconds / self.dt)

        self.obs_dim = task.OBS_DIM
        self.act_dim = task.ACT_DIM

        self.state = self.model.zero_state()
        self.t = np.zeros(n, np.float32)
        self.step_count = np.zeros(n, np.int64)
        self.prev_action = np.zeros((n, 4), np.float32)
        self.last_cmd = np.zeros((n, 4), np.float32)
        self.reset()

    # ------------------------------------------------------------------ obs
    def _reference(self):
        return self.traj.sample(self.t, self.rng)

    def _measure(self):
        """One noisy measurement of the state, drawn once per control step and
        shared by the policy's observation and the baseline controller, so both
        act on the same information. Attitude is passed through exactly, as in
        the observation."""
        st = self.state
        pos, vel, omega = self.rand.corrupt_obs(st.pos, st.vel, st.omega)
        return SimpleNamespace(pos=np.array(pos, np.float32), vel=np.array(vel, np.float32),
                               quat=st.quat.copy(), omega=np.array(omega, np.float32))

    def _observe(self, ref, meas=None):
        meas = meas if meas is not None else self._measure()
        self.meas = meas
        return task.observation(ref, meas.pos, meas.vel, self.state.quat, meas.omega,
                                self.prev_action).astype(np.float32)

    # ---------------------------------------------------------------- reset
    def reset(self, env_ids=None):
        n = self.cfg.num_envs
        ids = np.arange(n) if env_ids is None else np.asarray(env_ids)
        if ids.size == 0:
            return self._observe(self._reference())

        self.rand.apply(self.model, self.p, ids)

        hover = self.model.hover_command()
        self.state.pos[ids] = self.rng.uniform(-0.5, 0.5, (ids.size, 3)) \
            .astype(np.float32) + np.float32([0.0, 0.0, 1.5])
        self.state.vel[ids] = self.rng.normal(0, 0.2, (ids.size, 3)).astype(np.float32)
        q = self.rng.normal(0, 0.05, (ids.size, 4)).astype(np.float32)
        q[:, 0] = 1.0
        self.state.quat[ids] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        self.state.omega[ids] = self.rng.normal(0, 0.1, (ids.size, 3)).astype(np.float32)
        self.state.motor[ids] = np.tile(hover[ids], (1, 4))
        self.state.delay_buf[ids] = np.tile(hover[ids][:, None, :],
                                            (1, self.state.delay_buf.shape[1], 4))
        self.t[ids] = 0.0
        self.step_count[ids] = 0
        if hasattr(self.traj, "reset"):             # per-env random references (waypoints)
            self.traj.reset(ids, n, self.rng)
        self.prev_action[ids] = 0.0
        self.safety.reset(ids, init_cmd=self.state.motor[ids])   # = hover, see SafetyLayer.reset
        # Episodes start from a steady hover (motors at the true hover command),
        # so the baseline starts with its hover-thrust integrator converged.
        if self.cfg.baseline_info == "nominal":
            ratio = (self.model.mass[ids, 0] / self.p.mass) * \
                    (self.p.max_thrust_per_rotor / self.model.max_thrust[ids, 0])
        else:
            ratio = np.ones(ids.size, np.float32)
        self.ctrl.reset(ids, hover_ratio=ratio)
        obs = self._observe(self._reference())
        if self.sup is not None:
            self.sup.reset(ids, self.meas.omega)
        return obs

    # -------------------------------------------------------------- baseline
    def _baseline_command(self, ref):
        """Classical command. ``nominal`` (default): nominal parameters and the
        same noisy measurement the policy observed. ``privileged``: the true
        randomized parameters and true state -- information no real controller
        has; kept only to measure how much that advantage was worth."""
        if self.cfg.baseline_info == "privileged":
            return self.ctrl(self.state, ref, self.model.mass, self.model.inertia,
                             self.model.max_thrust)
        return self.ctrl(self.meas, ref, self._nom_mass, self._nom_inertia, self._nom_thrust)

    # ----------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        ref = self._reference()

        base = self._baseline_command(ref)

        if cfg.policy_mode == "baseline":
            cmd = base
        elif cfg.policy_mode == "direct":
            cmd = 0.5 * (action + 1.0)                       # [-1,1] -> [0,1]
        elif cfg.policy_mode == "residual":
            cmd = np.clip(base + cfg.residual_scale * action, 0.0, 1.0)
        else:
            raise ValueError(f"unknown policy_mode {cfg.policy_mode!r}")

        applied_action = action
        if self.sup is not None and cfg.policy_mode != "baseline":
            allow = self.sup.allow().cpu().numpy()
            cmd = np.where(allow[:, None], cmd, base).astype(np.float32)
            # what the policy will see as its previous action: the one that was applied
            held = (np.zeros_like(action) if cfg.policy_mode == "residual"
                    else np.clip(2.0 * base - 1.0, -1.0, 1.0))
            applied_action = np.where(allow[:, None], action, held).astype(np.float32)

        cmd = self.safety(cmd, base, self.state, ref["pos"])
        self.last_cmd = cmd

        for _ in range(cfg.decimation):
            self.rand.gust(self.model)
            self.state = self.model.step(self.state, cmd)
        self.t += self.dt
        self.step_count += 1

        reward, terms = self._reward(ref, action, cmd)
        term, trunc = self._done(ref)
        obs = self._observe(self._reference())
        meas = self.meas
        if self.sup is not None:
            self.sup.update(cmd, meas.omega)

        self.prev_action = applied_action.copy()
        done_ids = np.nonzero(term | trunc)[0]
        final_obs = obs.copy()              # pre-reset obs, needed to bootstrap truncated episodes
        # Pre-reset state: anything that logs the state after step() must use
        # this, because terminated *and truncated* envs are reset below.
        final_state = SimpleNamespace(pos=self.state.pos.copy(), vel=self.state.vel.copy(),
                                      quat=self.state.quat.copy(), omega=self.state.omega.copy())
        if done_ids.size:
            obs_reset = self.reset(done_ids)
            obs[done_ids] = obs_reset[done_ids]
            # reset() re-measured every env; keep the surviving envs' measurement
            # consistent with the observation the policy is about to act on.
            for k in ("pos", "vel", "quat", "omega"):
                getattr(meas, k)[done_ids] = getattr(self.meas, k)[done_ids]
            self.meas = meas
        return obs, reward, term, trunc, {"terms": terms, "final_obs": final_obs,
                                          "final_state": final_state}

    # --------------------------------------------------------------- reward
    def _reward(self, ref, action, cmd):
        st = self.state
        r, terms = task.reward(self.cfg, ref, st.pos, st.vel, st.omega, action, self.prev_action)
        return r.astype(np.float32), terms

    # ----------------------------------------------------------------- done
    def _done(self, ref):
        term = task.termination(self.cfg, ref, self.state.pos, self.state.quat)
        trunc = self.step_count >= self.max_steps
        return term, trunc

    # ------------------------------------------------------------ curriculum
    def set_curriculum(self, progress: float):
        """progress in [0, 1] -- ramps domain randomization from off to full."""
        self.rand.set_scale(progress)
