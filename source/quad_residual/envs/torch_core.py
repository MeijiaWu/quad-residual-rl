"""Everything the Isaac Lab env does except talking to PhysX -- in plain torch.

The Isaac env (``quad_track_env.py``) is deliberately thin glue: it reads the
rigid-body state from PhysX, hands it to this core, and writes back the body
wrench and the reset poses. All task logic -- references, randomization,
actuator (delay + motor lag), drag and wind, the classical baseline, the
policy-to-command mapping, observation, reward, termination -- lives here and
reuses the very functions sim_lite uses:

  quad_residual.task                     observation / reward / termination
  controllers.geometric                  baseline (backend-agnostic)
  dynamics.actuator.actuate_torch        delay + motor lag (tested against NumPy)
  tasks.trajectories                     references (backend-agnostic)
  dynamics.randomization_torch           same distributions as the NumPy one

Because it does not import Isaac Lab, it runs (and is tested) on a CPU with
torch alone: ``scripts/check_torch_core.py`` integrates the rigid body with the
same semi-implicit Euler as sim_lite and checks that both envs give the same
baseline statistics. On the GPU, PhysX takes the integrator's place.

State convention (as sim_lite): position / linear velocity in the world frame
relative to the env origin, quaternion (w, x, y, z) body -> world, angular
velocity in the body frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from .. import task
from ..backend import quat_rotate, quat_rotate_inverse
from ..controllers.geometric import GeometricController, GeometricGains
from ..controllers.supervisor import LagSupervisor, SupervisorCfg
from ..dynamics.actuator import actuate_torch
from ..dynamics.quadrotor import QuadParams, mixing_matrix
from ..dynamics.randomization import RandomizationCfg
from ..dynamics.randomization_torch import TorchDomainRandomizer
from ..tasks import trajectories


class TorchQuadCore:
    def __init__(self, cfg, num_envs: int, device, rand_cfg: RandomizationCfg | None = None,
                 params: QuadParams | None = None, seed: int = 0):
        """``cfg`` needs the task fields of ``sim_lite.env.EnvCfg`` (policy_mode,
        residual_scale, baseline_*, trajectory, reward weights, termination,
        randomize) plus ``decimation``."""
        self.cfg, self.n, self.device = cfg, num_envs, device
        self.p = params or QuadParams()
        p, n = self.p, num_envs
        self.dt_ctrl = p.dt * cfg.decimation

        rc = rand_cfg or RandomizationCfg()
        rc.enabled = bool(cfg.randomize) and rc.enabled
        self.rand = TorchDomainRandomizer(rc, p, n, device, seed)
        self.ctrl = GeometricController(
            p, GeometricGains.preset(cfg.baseline_gains, cfg.baseline_integral), n, dt=self.dt_ctrl)
        self.traj = trajectories.make(cfg.trajectory)
        self.traj_rng = np.random.default_rng(seed)
        self.mix = torch.as_tensor(mixing_matrix(p.arm_length, p.torque_coeff),
                                   dtype=torch.float32, device=device)

        f = dict(device=device, dtype=torch.float32)
        self.nom_mass = torch.full((n, 1), p.mass, **f)
        self.nom_inertia = torch.tensor(p.inertia, **f).repeat(n, 1)
        self.nom_thrust = torch.full((n, 1), p.max_thrust_per_rotor, **f)

        d_max = p.actuation_delay_steps + self.rand.max_extra_delay
        self.motor = torch.zeros((n, 4), **f)
        self.delay_buf = torch.zeros((n, max(d_max, 0), 4), **f)
        self.t = torch.zeros(n, **f)
        self.actions = torch.zeros((n, 4), **f)
        self.prev_action = torch.zeros((n, 4), **f)
        self.cmd = torch.zeros((n, 4), **f)
        self.ref = None
        self.meas = None
        # Optional OOD supervisor (evaluation only), same code as sim_lite.
        self.sup = None
        if getattr(cfg, "supervisor", False):
            if cfg.supervisor_support is None:
                raise ValueError("supervisor_support must be the policy's *training* support")
            self.sup = LagSupervisor(
                SupervisorCfg(enabled=True, probe_s=cfg.supervisor_probe_s, gate=cfg.supervisor_gate),
                p, n, cfg.decimation, dict(cfg.supervisor_support), device=device)

    # ------------------------------------------------------------- curriculum
    def set_curriculum(self, progress: float):
        self.rand.set_scale(progress)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids, gen: torch.Generator | None = None):
        """Resample parameters and return the spawn state for ``env_ids``:
        (pos, quat, lin_vel_world, ang_vel_body), same distribution as sim_lite."""
        ids = env_ids
        k = len(ids)
        self.rand.apply(ids)
        g = gen or self.rand.gen
        dev = self.device
        pos = (torch.rand((k, 3), generator=g, device=dev) - 0.5) + torch.tensor([0.0, 0.0, 1.5], device=dev)
        vel = 0.2 * torch.randn((k, 3), generator=g, device=dev)
        q = 0.05 * torch.randn((k, 4), generator=g, device=dev)
        q[:, 0] = 1.0
        quat = q / q.norm(dim=-1, keepdim=True)
        omega = 0.1 * torch.randn((k, 3), generator=g, device=dev)

        hover = (self.rand.mass[ids] * self.p.gravity) / (4.0 * self.rand.max_thrust[ids])
        self.motor[ids] = hover.expand(-1, 4)
        if self.delay_buf.shape[1]:
            self.delay_buf[ids] = hover[:, None, :].expand(-1, self.delay_buf.shape[1], 4)
        self.t[ids] = 0.0
        self.prev_action[ids] = 0.0
        if hasattr(self.traj, "reset"):
            self.traj.reset(ids, self.n, self.traj_rng)
        # Steady-hover start: warm-start the baseline's hover-thrust integrator.
        if self.cfg.baseline_info == "nominal":
            ratio = (self.rand.mass[ids, 0] / self.p.mass) * \
                    (self.p.max_thrust_per_rotor / self.rand.max_thrust[ids, 0])
        else:
            ratio = torch.ones(k, device=dev)
        self.ctrl.reset(ids, hover_ratio=ratio)
        if self.sup is not None:
            self.sup.reset(ids)          # first measurement arrives with observe()
        return pos, quat, vel, omega

    # ------------------------------------------------------------ observation
    def observe(self, state):
        """Draw one measurement (shared with the baseline at the next control
        step) and build the observation. Call once per control step, after resets."""
        mp, mv, mw = self.rand.corrupt_obs(state.pos, state.vel, state.omega)
        self.meas = SimpleNamespace(pos=mp, vel=mv, quat=state.quat, omega=mw)
        if self.sup is not None:         # the command just flown + the rate it produced
            self.sup.update(self.cmd, mw)
        ref = self.traj.sample(self.t, self.traj_rng)
        obs = task.observation(ref, mp, mv, state.quat, mw, self.prev_action)
        # Same ordering as sim_lite: the observation carries the action from the
        # previous step, then prev_action is advanced.
        self.prev_action = self.actions.clone()
        return obs

    # ---------------------------------------------------------------- control
    def control(self, actions, state):
        """Once per control step: reference, baseline, policy -> rotor command."""
        c = self.cfg
        self.actions = actions.clamp(-1.0, 1.0)
        self.ref = self.traj.sample(self.t, self.traj_rng)
        if c.baseline_info == "privileged":
            base = self.ctrl(state, self.ref, self.rand.mass, self.rand.inertia, self.rand.max_thrust)
        else:
            base = self.ctrl(self.meas, self.ref, self.nom_mass, self.nom_inertia, self.nom_thrust)
        if c.policy_mode == "baseline":
            cmd = base
        elif c.policy_mode == "direct":
            cmd = 0.5 * (self.actions + 1.0)
        elif c.policy_mode == "residual":
            cmd = (base + c.residual_scale * self.actions).clamp(0.0, 1.0)
        else:
            raise ValueError(f"unknown policy_mode {c.policy_mode!r}")
        if self.sup is not None and c.policy_mode != "baseline":
            allow = self.sup.allow()[:, None]
            cmd = torch.where(allow, cmd, base)
            # the policy's next observation carries the action that was applied
            held = (torch.zeros_like(self.actions) if c.policy_mode == "residual"
                    else (2.0 * base - 1.0).clamp(-1.0, 1.0))
            self.actions = torch.where(allow, self.actions, held)
        self.base_cmd = base
        self.cmd = cmd
        self.t = self.t + self.dt_ctrl
        return cmd

    # ---------------------------------------------------------------- physics
    def wrench(self, state):
        """Once per physics step: actuator + drag/wind -> body-frame force and
        torque (gravity is left to the integrator / PhysX)."""
        self.rand.gust()
        alpha = (self.p.dt / self.rand.motor_tau.clamp_min(1e-6)).clamp(0.0, 1.0)
        self.motor, self.delay_buf = actuate_torch(
            self.delay_buf, self.motor, self.cmd, self.rand.delay_steps, alpha)
        w = (self.motor * self.rand.max_thrust) @ self.mix.T
        drag_world = -self.rand.drag * (state.vel - self.rand.wind)
        force = quat_rotate_inverse(state.quat, drag_world)
        force = force + torch.cat([torch.zeros_like(w[:, :2]), w[:, 0:1]], -1)
        return force, w[:, 1:4]

    # -------------------------------------------------------- reward / done
    def reward(self, state):
        r, terms = task.reward(self.cfg, self.ref, state.pos, state.vel, state.omega,
                               self.actions, self.prev_action)
        return r, terms

    def terminated(self, state):
        return task.termination(self.cfg, self.ref, state.pos, state.quat)


def rigid_body_step(state, force_b, torque_b, mass, inertia, dt, gravity):
    """Semi-implicit Euler, identical to dynamics/quadrotor.py -- the CPU
    stand-in for PhysX used to validate the core without Isaac Sim."""
    f_world = quat_rotate(state.quat, force_b)
    acc = f_world / mass
    acc = acc - torch.tensor([0.0, 0.0, gravity], device=acc.device)
    vel = state.vel + dt * acc
    pos = state.pos + dt * vel
    I = inertia
    om = state.omega
    om_dot = (torque_b - torch.cross(om, I * om, dim=-1)) / I
    om = om + dt * om_dot
    wq = torch.cat([torch.zeros_like(om[:, :1]), om], -1)
    from ..backend import quat_mul, quat_normalize
    quat = quat_normalize(state.quat + dt * 0.5 * quat_mul(state.quat, wq))
    return SimpleNamespace(pos=pos, vel=vel, quat=quat, omega=om)
