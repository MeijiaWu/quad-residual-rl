"""Isaac Lab DirectRLEnv for quadrotor trajectory tracking (Isaac Lab 2.3).

Thin glue around :class:`envs.torch_core.TorchQuadCore`, which holds all task
logic and is validated against sim_lite on a CPU (scripts/check_torch_core.py).
This file only does what needs PhysX:

* read the rigid-body state (``root_pos_w`` minus the env origin,
  ``root_quat_w``, ``root_lin_vel_w``, ``root_ang_vel_b``);
* apply the body-frame wrench the core computes, every physics step
  (PhysX integrates it together with gravity);
* write reset poses / velocities and the randomized mass and inertia.

Timing matches sim_lite exactly: the command (baseline + policy) is computed
once per control step in ``_pre_physics_step`` and held for the ``decimation``
physics steps, while the actuator model, wind gusts and drag are advanced every
physics step in ``_apply_action``. Isaac Lab calls ``_get_dones`` and
``_get_rewards`` after physics and ``_get_observations`` after resets, which is
the order sim_lite uses.

Not ported: the deployment safety layer (training runs without it in sim_lite
too).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject
    from isaaclab.envs import DirectRLEnv
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Isaac Lab is required for this module. Use "
        "quad_residual.sim_lite.env.QuadTrackEnv for CPU development."
    ) from exc

from .. import config as C
from ..backend import quat_rotate
from .quad_track_env_cfg import QuadTrackEnvCfg
from .torch_core import TorchQuadCore


class QuadTrackEnv(DirectRLEnv):
    cfg: QuadTrackEnvCfg

    def __init__(self, cfg: QuadTrackEnvCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        rand_cfg = C.rand_cfg(C.load(overrides=list(cfg.config_overrides)))
        self.core = TorchQuadCore(cfg, self.num_envs, self.device, rand_cfg,
                                  seed=int(cfg.seed or 0))
        self.core.set_curriculum(0.0 if cfg.curriculum_ramp_steps > 0 else 1.0)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.rigid_objects["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ state
    def _state(self) -> SimpleNamespace:
        d = self.robot.data
        return SimpleNamespace(pos=d.root_pos_w - self.scene.env_origins,
                               vel=d.root_lin_vel_w,
                               quat=d.root_quat_w,
                               omega=d.root_ang_vel_b)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor):
        # Curriculum: env.step() calls so far / ramp length.
        if self.cfg.curriculum_ramp_steps > 0:
            self.core.set_curriculum(self.common_step_counter / self.cfg.curriculum_ramp_steps)
        self.core.control(actions, self._state())

    def _apply_action(self):
        force_b, torque_b = self.core.wrench(self._state())
        self.robot.set_external_force_and_torque(force_b.unsqueeze(1), torque_b.unsqueeze(1))

    # ------------------------------------------------------- obs / reward / done
    def _get_observations(self) -> dict:
        return {"policy": self.core.observe(self._state())}

    def _get_rewards(self) -> torch.Tensor:
        r, terms = self.core.reward(self._state())
        self.extras.setdefault("log", {}).update(
            {f"reward/{k}": v.mean() for k, v in terms.items()})
        return r

    def _get_dones(self):
        st = self._state()
        terminated = self.core.terminated(st)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Pre-reset snapshot for evaluation: Isaac Lab resets terminated and
        # timed-out envs before step() returns.
        self.final_err = (st.pos - self.core.ref["pos"]).norm(dim=-1)
        self.final_omega = st.omega.clone()
        self.final_cmd = self.core.cmd.clone()
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        pos, quat, vel, omega_b = self.core.reset(env_ids)
        self._write_mass_inertia(env_ids)

        pose = torch.cat([pos + self.scene.env_origins[env_ids], quat], dim=-1)
        twist = torch.cat([vel, quat_rotate(quat, omega_b)], dim=-1)   # PhysX wants world-frame rates
        self.robot.write_root_pose_to_sim(pose, env_ids)
        self.robot.write_root_velocity_to_sim(twist, env_ids)

    def _write_mass_inertia(self, env_ids):
        """Push the core's randomized mass and inertia into PhysX (CPU tensors,
        as the PhysX tensor views require -- same pattern as Isaac Lab's
        randomize_rigid_body_mass event)."""
        view = self.robot.root_physx_view
        idx = env_ids.to("cpu", dtype=torch.long)
        ids = idx
        masses = view.get_masses().clone()
        masses.view(self.num_envs, -1)[idx, 0] = self.core.rand.mass[env_ids, 0].cpu()
        view.set_masses(masses, ids)
        inertias = view.get_inertias().clone()
        flat = inertias.view(self.num_envs, -1)          # (N, 9) row-major 3x3
        I = self.core.rand.inertia[env_ids].cpu()
        flat[idx, 0], flat[idx, 4], flat[idx, 8] = I[:, 0], I[:, 1], I[:, 2]
        view.set_inertias(inertias, ids)
