"""Isaac Lab DirectRLEnv config for the Unitree Go2, Phase 3 (terrain curriculum +
domain randomization).

Physical/behavioural parameters are read from the existing Isaac-Gym config
(configs/go2_config.Go2RoughCfg) so both simulators stay in sync from a single
source of truth.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, UrdfFileCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from configs.go2_config import Go2RoughCfg
from global_config import ROOT_DIR

_cfg = Go2RoughCfg()
_dr = _cfg.domain_rand

# Sub-terrain proportions follow Go2RoughCfg.terrain.terrain_proportions
# ([smooth slope, rough slope, stairs up, stairs down, discrete, flat] =
# [0.3, 0.3, 0.0, 0.0, 0.1, 0.3]), mapped onto Isaac Lab's built-in terrain generator
# functions (isaaclab.terrains).
_p = _cfg.terrain.terrain_proportions
GO2_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(_cfg.terrain.terrain_length, _cfg.terrain.terrain_width),
    border_width=_cfg.terrain.border_size,
    num_rows=_cfg.terrain.num_rows,
    num_cols=_cfg.terrain.num_cols,
    horizontal_scale=_cfg.terrain.horizontal_scale,
    vertical_scale=_cfg.terrain.vertical_scale,
    slope_threshold=_cfg.terrain.slope_treshold,
    curriculum=_cfg.terrain.curriculum,
    use_cache=False,
    sub_terrains={
        # now part of the training curriculum (proportion=_p[5], was a fixed 0.0
        # reachable only via play_lab.py's --terrain_type flat).
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=_p[5]),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=_p[0], slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "rough_slope": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=_p[1], noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=_p[2], step_height_range=(0.05, 0.23), step_width=0.3, platform_width=3.0, border_width=1.0,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=_p[3], step_height_range=(0.05, 0.23), step_width=0.3, platform_width=3.0, border_width=1.0,
        ),
        "discrete": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=_p[4], grid_width=0.45, grid_height_range=(0.02, 0.15), platform_width=2.0
        ),
    },
)


# --- domain randomization events, applied once at env creation (mode="startup") ---
# Equivalent of envs/legged_robot.py's _process_rigid_shape_props / _process_rigid_body_props.
# push_robots, randomize_motor and randomize_kpkd have no direct mdp event that matches the
# original's semantics (synchronized push interval, torque/PD-gain scaling) closely enough,
# so those are applied by hand in Go2DirectEnv instead (see go2_direct_env.py).
_event_terms = {}
if _dr.randomize_friction:
    _event_terms["physics_material"] = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": tuple(_dr.friction_range),
            "dynamic_friction_range": tuple(_dr.friction_range),
            "restitution_range": tuple(_dr.restitution_range) if _dr.randomize_restitution else (0.0, 0.0),
            "num_buckets": 64,
        },
    )
if _dr.randomize_base_mass:
    _event_terms["add_base_mass"] = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": tuple(_dr.added_mass_range),
            "operation": "add",
        },
    )
if _dr.randomize_base_com:
    _event_terms["add_base_com"] = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": tuple(_dr.added_com_range),
                "y": tuple(_dr.added_com_range),
                "z": tuple(_dr.added_com_range),
            },
        },
    )


@configclass
class Go2DirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = _cfg.control.decimation
    episode_length_s = _cfg.env.episode_length_s
    action_space = _cfg.env.num_actions
    observation_space = _cfg.env.num_observations
    state_space = 0
    is_finite_horizon = False

    # domain randomization (Go2RoughCfg.domain_rand): friction/base-mass/base-com,
    # applied once at env creation. EventManager accepts a plain {name: EventTermCfg}
    # dict directly (see isaaclab.managers.manager_base.ManagerBase._prepare_terms).
    events: dict | None = _event_terms if _event_terms else None

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=_cfg.sim.dt,
        render_interval=_cfg.control.decimation,
        gravity=tuple(_cfg.sim.gravity),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=_cfg.terrain.static_friction,
            dynamic_friction=_cfg.terrain.dynamic_friction,
            restitution=_cfg.terrain.restitution,
        ),
        # 4096 robots on generated trimesh terrain overflow the default contact patch
        # buffer (5 * 2**15 = 163840; PhysX asked for ~167k). Overflow silently drops
        # contacts, which corrupts the foot-contact-driven stride and collision rewards,
        # and the error spam alone slows the run down.
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=5 * 2**17),
    )

    # Phase 2: generated rough terrain with curriculum (levels = rows, types = cols),
    # matching Go2RoughCfg.terrain (mesh_type='trimesh').
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=_cfg.terrain.max_init_terrain_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=_cfg.terrain.static_friction,
            dynamic_friction=_cfg.terrain.dynamic_friction,
            restitution=_cfg.terrain.restitution,
        ),
        debug_vis=False,
    )

    # height sampling under the robot base, replaces envs/legged_robot.py's
    # _get_base_heights() (63-point grid: x in [-0.15,0.15], y in [-0.2,0.2], 0.05 spacing),
    # used by the base_height reward.
    base_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.3, 0.4)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_cfg.env.num_envs, env_spacing=_cfg.env.env_spacing, replicate_physics=True
    )

    # robot - imported from the same URDF used by the Isaac Gym env, joint drive
    # is disabled ("none") because torques are computed manually in
    # Go2DirectEnv._apply_action to match the original PD control loop exactly.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UrdfFileCfg(
            asset_path=_cfg.asset.file.format(ROOT_DIR=ROOT_DIR),
            fix_base=_cfg.asset.fix_base_link,
            merge_fixed_joints=_cfg.asset.collapse_fixed_joints,
            replace_cylinders_with_capsules=_cfg.asset.replace_cylinder_with_capsule,
            self_collision=(_cfg.asset.self_collisions == 0),
            activate_contact_sensors=True,
            joint_drive=UrdfFileCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=_cfg.asset.disable_gravity,
                linear_damping=_cfg.asset.linear_damping,
                angular_damping=_cfg.asset.angular_damping,
                max_linear_velocity=_cfg.asset.max_linear_velocity,
                max_angular_velocity=_cfg.asset.max_angular_velocity,
                retain_accelerations=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=(_cfg.asset.self_collisions == 0),
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(_cfg.init_state.pos),
            joint_pos=dict(_cfg.init_state.default_joint_angles),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=_cfg.rewards.soft_dof_pos_limit,
        actuators={
            # stiffness/damping = 0 -> no implicit PD, pure effort (torque) passthrough.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.0, track_air_time=False
    )
