"""Isaac Lab configuration for the quadrotor tracking task.

Requires Isaac Sim 5.1 + Isaac Lab 2.3 (``isaaclab.envs.DirectRLEnvCfg``) and a
CUDA GPU; see README "Isaac Lab". The CPU env ``sim_lite/env.py`` is the
reference: same task, same observation / reward / termination (shared through
``quad_residual.task``), same controller and actuator code.

The airframe is a rigid box, not a rotorcraft mesh, on purpose. PhysX only
integrates the rigid body; rotor thrust, motor lag, transport delay, drag and
wind are all applied by our own (shared) code as a body wrench. What matters to
PhysX is therefore mass and inertia, and a uniform box of 0.252 x 0.252 x
0.084 m and 0.85 kg has exactly the nominal inertia (5e-3, 5e-3, 9e-3) kg m^2.
A Crazyflie USD would bring a 27 g mass and propeller bodies we would only
have to override.
"""

from __future__ import annotations

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Isaac Lab is required for this module. Install Isaac Sim + Isaac Lab, "
        "or use quad_residual.sim_lite.env.QuadTrackEnv for CPU development."
    ) from exc

from ..dynamics.quadrotor import QuadParams

_P = QuadParams()
# Box whose uniform-density inertia equals the nominal diagonal inertia:
# I = m/12 * (b^2 + c^2, a^2 + c^2, a^2 + b^2)  ->  a = b = 0.252 m, c = 0.084 m.
BOX_SIZE = (0.252, 0.252, 0.084)


@configclass
class QuadTrackEnvCfg(DirectRLEnvCfg):
    # --- simulation -------------------------------------------------------
    decimation = 5                    # 250 Hz physics -> 50 Hz control, as sim_lite
    episode_length_s = 10.0
    sim = sim_utils.SimulationCfg(dt=_P.dt, render_interval=decimation)

    # --- spaces (must match quad_residual.task) ----------------------------
    observation_space = 20
    action_space = 4
    state_space = 0

    # --- scene ------------------------------------------------------------
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=6.0, replicate_physics=True)

    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # sim_lite has no damping besides the explicit drag term: PhysX's
                # defaults would add some, so switch them off.
                linear_damping=0.0,
                angular_damping=0.0,
                max_angular_velocity=6000.0,      # deg/s; termination acts long before
                max_linear_velocity=100.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_P.mass),
            # Envs overlap when a vehicle wanders; sim_lite has no collisions either.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
    )

    # --- task (names identical to sim_lite.env.EnvCfg) ---------------------
    policy_mode = "residual"          # "direct" | "residual" | "baseline"
    residual_scale = 0.25
    baseline_gains = "robust"         # controllers.geometric.BASELINE_PRESETS
    baseline_integral = True
    baseline_info = "nominal"         # nominal params + the policy's measurement
    trajectory = "lemniscate"         # hover | lemniscate | waypoints

    # reward
    w_pos = 1.0
    w_vel = 0.05
    w_rate = 0.02
    w_action = 0.01
    w_smooth = 0.03
    w_alive = 0.5

    # termination
    max_pos_err = 4.0
    max_tilt = 1.3

    # --- sim-to-real ------------------------------------------------------
    randomize = True
    # Randomization ramps from nominal to full over this many env.step() calls
    # (each call advances every env by one control step). train_isaac.py sets it
    # to 60 % of training, matching sim_lite's curriculum_fraction.
    curriculum_ramp_steps = 20_000
    seed = 0
    # Randomization ranges come from configs/domain_rand.yaml, as in sim_lite;
    # these --set style overrides are applied on top ("domain_rand.x.y=...").
    config_overrides: list = []

    # --- OOD supervisor (evaluation only; controllers/supervisor.py) --------
    supervisor = False
    supervisor_probe_s = 0.5
    supervisor_gate = "lag"           # lag | support
    supervisor_support: dict | None = None   # training_support() of the policy's training config
