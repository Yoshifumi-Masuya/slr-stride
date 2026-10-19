"""Isaac Lab DirectRLEnv port of envs/legged_robot.py for the Unitree Go2 - Phase 2.

Phase 2 scope:
    generated rough terrain with a game-inspired difficulty curriculum (rows =
    levels, cols = types) and a height scanner under the base for the
    base_height reward. Still no domain randomization / depth camera.
    Observation layout and joint I/O order are kept numerically compatible with
    the existing Isaac-Gym env so that policy.jit / encoder.jit keep working
    (note: the 187-point terrain scan from the original env is only used there
    for debug visualization, never fed to the policy - so it is intentionally
    not ported here; only the 63-point base_height scan, which the reward
    actually consumes, is).
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from configs.go2_config import Go2RoughCfg

from .go2_direct_env_cfg import Go2DirectEnvCfg

# SDK/policy-facing joint order (matches envs/legged_robot.py's reindex() target,
# i.e. the order policy.jit / encoder.jit were trained on).
POLICY_JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
POLICY_HIP_INDICES = [0, 3, 6, 9]

REWARD_SCALE_NAMES = [
    "powers", "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "ang_vel_xy",
    "dof_acc", "base_height", "action_rate", "action_smoothness", "foot_clearance",
    "orientation", "stride_length", "collision", "stand_still",
]


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    angles = angles % (2 * torch.pi)
    angles -= 2 * torch.pi * (angles > torch.pi)
    return angles


class Go2DirectEnv(DirectRLEnv):
    cfg: Go2DirectEnvCfg

    def __init__(self, cfg: Go2DirectEnvCfg, render_mode: str | None = None, **kwargs):
        # Original Isaac-Gym config, single source of truth for scalar params.
        self._gym_cfg = Go2RoughCfg()

        super().__init__(cfg, render_mode, **kwargs)

        self.n_proprio = self._gym_cfg.env.n_proprio
        self.history_len = self._gym_cfg.env.history_len
        self.num_actions = self._gym_cfg.env.num_actions

        # --- joint order mapping: Isaac Lab sim order <-> SDK/policy order ---
        sim_names = self._robot.data.joint_names
        self._policy_to_sim_idx = torch.tensor(
            [sim_names.index(n) for n in POLICY_JOINT_ORDER], device=self.device, dtype=torch.long
        )
        self._sim_to_policy_idx = torch.tensor(
            [POLICY_JOINT_ORDER.index(n) for n in sim_names], device=self.device, dtype=torch.long
        )

        # --- PD gains / default pose, all in policy order ---
        ctrl = self._gym_cfg.control
        p_gains = torch.zeros(self.num_actions, device=self.device)
        d_gains = torch.zeros(self.num_actions, device=self.device)
        default_dof_pos = torch.zeros(self.num_actions, device=self.device)
        for i, name in enumerate(POLICY_JOINT_ORDER):
            default_dof_pos[i] = self._gym_cfg.init_state.default_joint_angles[name]
            for dof_key in ctrl.stiffness.keys():
                if dof_key in name:
                    p_gains[i] = ctrl.stiffness[dof_key]
                    d_gains[i] = ctrl.damping[dof_key]
        self.default_dof_pos_policy = default_dof_pos.unsqueeze(0)
        self.p_gains_policy = p_gains
        self.d_gains_policy = d_gains

        joint_effort_limits_sim = self._robot.data.joint_effort_limits[0]
        self.torque_limits_policy = joint_effort_limits_sim[self._policy_to_sim_idx]
        # Torque is hard-clipped at the URDF rating, but an episode ends earlier, once the
        # pre-clip demand passes this softer threshold - so the policy learns to keep a
        # margin to the motor rating rather than riding it. Same split as envs/legged_robot.py.
        self.soft_torque_limits_policy = self.torque_limits_policy * self._gym_cfg.rewards.soft_torque_limit

        # --- domain randomization: motor strength, PD gain scaling, periodic push ---
        # (friction/base-mass/base-com are handled by the "startup" events in
        # Go2DirectEnvCfg.events; those have no per-step state, unlike these).
        dr = self._gym_cfg.domain_rand
        if dr.randomize_motor:
            lo, hi = dr.motor_strength_range
            self.motor_strength = (hi - lo) * torch.rand(self.num_envs, self.num_actions, device=self.device) + lo
        else:
            self.motor_strength = torch.ones(self.num_envs, self.num_actions, device=self.device)
        self.randomize_kpkd = dr.randomize_kpkd
        if dr.randomize_kpkd:
            kp_lo, kp_hi = dr.kp_range
            kd_lo, kd_hi = dr.kd_range
            self.kp_factor = (kp_hi - kp_lo) * torch.rand(self.num_envs, self.num_actions, device=self.device) + kp_lo
            self.kd_factor = (kd_hi - kd_lo) * torch.rand(self.num_envs, self.num_actions, device=self.device) + kd_lo
        self.push_robots = dr.push_robots
        self.max_push_vel_xy = dr.max_push_vel_xy
        self.push_interval_steps = int((dr.push_interval_s / self.step_dt) + 0.999999)

        # --- feet / body bookkeeping ---
        self._feet_ids, feet_names = self._robot.find_bodies(f".*{self._gym_cfg.asset.foot_name}")
        # same feet, indexed into the contact sensor's own body order (matched by name so it
        # lines up with self._feet_ids regardless of each API's internal body ordering).
        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(feet_names, preserve_order=True)
        # knee (calf) links, for the collision penalty - these should never touch the ground.
        self._knee_contact_ids, _ = self._contact_sensor.find_bodies(".*calf")
        # bodies whose ground contact ends the episode (asset.terminate_after_contacts_on),
        # same list the Isaac Gym env builds termination_contact_indices from.
        self._termination_contact_ids = []
        for name in self._gym_cfg.asset.terminate_after_contacts_on:
            ids, _ = self._contact_sensor.find_bodies(f".*{name}.*")
            self._termination_contact_ids += ids

        # --- buffers ---
        # commands: [lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stride_length]
        self.commands = torch.zeros(self.num_envs, 5, device=self.device)
        self.command_ranges = {
            "lin_vel_x": list(self._gym_cfg.commands.ranges.lin_vel_x),
            "lin_vel_y": list(self._gym_cfg.commands.ranges.lin_vel_y),
            "ang_vel_yaw": list(self._gym_cfg.commands.ranges.ang_vel_yaw),
            "heading": list(self._gym_cfg.commands.ranges.heading),
            "stride_length": list(self._gym_cfg.commands.ranges.stride_length),
        }
        self.commands_scale = torch.tensor(
            [self._gym_cfg.normalization.obs_scales.lin_vel] * 2
            + [self._gym_cfg.normalization.obs_scales.ang_vel],
            device=self.device,
        )
        self.forward_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.forward_vec[:, 0] = 1.0

        # --- stride length tracking: same-foot touchdown-to-touchdown horizontal distance ---
        num_feet = len(self._feet_ids)
        self._last_foot_contact = torch.zeros(self.num_envs, num_feet, dtype=torch.bool, device=self.device)
        self._last_touchdown_pos_w = torch.zeros(self.num_envs, num_feet, 2, device=self.device)
        self._stride_length = torch.zeros(self.num_envs, num_feet, device=self.device)

        self._actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._last_dof_vel = torch.zeros_like(self._actions)
        self._last_torques = torch.zeros_like(self._actions)
        self.action_history_buf = torch.zeros(self.num_envs, self.history_len, self.num_actions, device=self.device)
        self.obs_history_buf = torch.zeros(self.num_envs, self.history_len, self.n_proprio, device=self.device)

        # true gravity direction (down), unit vector, world frame
        self.gravity_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.gravity_vec[:, 2] = -1.0
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = self.gravity_vec.clone()

        # --- Madgwick IMU estimator (policy gravity observation g_hat) ---
        self.estimated_quat = self._robot.data.root_quat_w.clone()
        self.estimated_gravity = quat_apply_inverse(self.estimated_quat, self.gravity_vec)
        self.prev_base_lin_vel_world = self._robot.data.root_lin_vel_w.clone()
        self.imu_gyro = torch.zeros(self.num_envs, 3, device=self.device)
        self.imu_accel = torch.zeros(self.num_envs, 3, device=self.device)
        self.madgwick_beta = 0.1
        self.imu_gyro_noise_std = 0.01
        self.imu_accel_noise_std = 0.05
        self.madgwick_accel_min = 7.0
        self.madgwick_accel_max = 12.5

        # --- episode reward logging ---
        rewards_cfg = self._gym_cfg.rewards
        self.reward_scales = {
            name: getattr(rewards_cfg.scales, name) * self.step_dt
            for name in REWARD_SCALE_NAMES
            if getattr(rewards_cfg.scales, name) != 0.0
        }
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_scales.keys()
        }
        # termination penalty is a one-time event reward, not scaled by step_dt like the
        # others above, so it's applied separately in _get_rewards() rather than through
        # self.reward_scales; tracked here too so it still shows up in episode logging.
        self._episode_sums["termination"] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums["fall"] = torch.zeros(self.num_envs, device=self.device)
        self._torque_limit_exceeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # The stride_length reward only switches on once the robot walks the commanded
        # velocity well (see _update_stride_curriculum). Before that it is dead weight:
        # a policy that cannot hold a velocity has no gait to modulate, and the stride
        # term just adds noise to the gradient. Latches on, never off.
        self._stride_reward_enabled = not self._gym_cfg.rewards.stride_curriculum

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.base_height_scanner)
        self.scene.sensors["base_height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # action pipeline
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._torque_limit_exceeded[:] = False
        self.action_history_buf = torch.cat(
            [self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1
        )
        clip_actions = self._gym_cfg.normalization.clip_actions
        self._actions = torch.clip(actions, -clip_actions, clip_actions)

        if self._gym_cfg.control.use_filter:
            filtered = self._last_actions * 0.2 + self._actions * 0.8
        else:
            filtered = self._actions

        actions_scaled = filtered * self._gym_cfg.control.action_scale
        actions_scaled = actions_scaled.clone()
        actions_scaled[:, POLICY_HIP_INDICES] *= self._gym_cfg.control.hip_scale_reduction
        self._joint_pos_target_policy = actions_scaled + self.default_dof_pos_policy

    def _apply_action(self):
        dof_pos_policy = self._robot.data.joint_pos[:, self._policy_to_sim_idx]
        dof_vel_policy = self._robot.data.joint_vel[:, self._policy_to_sim_idx]

        if self.randomize_kpkd:
            torques_policy = self.kp_factor * self.p_gains_policy * (
                self._joint_pos_target_policy - dof_pos_policy
            ) - self.kd_factor * self.d_gains_policy * dof_vel_policy
        else:
            torques_policy = (
                self.p_gains_policy * (self._joint_pos_target_policy - dof_pos_policy)
                - self.d_gains_policy * dof_vel_policy
            )
        torques_policy = torques_policy * self.motor_strength
        # PD-demanded torque before clipping - any joint over the soft threshold on any
        # physics substep marks the whole control step for termination (see _get_dones()).
        self._torque_limit_exceeded |= (torques_policy.abs() > self.soft_torque_limits_policy).any(dim=1)
        torques_policy = torch.clip(torques_policy, -self.torque_limits_policy, self.torque_limits_policy)
        torques_sim = torques_policy[:, self._sim_to_policy_idx]
        self._robot.set_joint_effort_target(torques_sim)

    # ------------------------------------------------------------------
    # per-env-step derived quantities (equivalent of the original
    # post_physics_step() header, run once before dones/rewards/obs)
    # ------------------------------------------------------------------
    def _update_post_physics_buffers(self):
        base_quat = self._robot.data.root_quat_w
        current_lin_vel_world = self._robot.data.root_lin_vel_w

        self.base_lin_vel = self._robot.data.root_lin_vel_b
        self.base_ang_vel = self._robot.data.root_ang_vel_b
        self.projected_gravity = quat_apply_inverse(base_quat, self.gravity_vec)

        # simulated IMU + Madgwick update
        lin_acc_world = (current_lin_vel_world - self.prev_base_lin_vel_world) / self.step_dt
        self.prev_base_lin_vel_world = current_lin_vel_world.clone()
        gravity_acc_world = 9.81 * self.gravity_vec
        specific_force_world = lin_acc_world - gravity_acc_world
        self.imu_accel = quat_apply_inverse(base_quat, specific_force_world)
        self.imu_gyro = self.base_ang_vel.clone()

        if self.imu_gyro_noise_std > 0.0:
            self.imu_gyro = self.imu_gyro + self.imu_gyro_noise_std * torch.randn_like(self.imu_gyro)
        if self.imu_accel_noise_std > 0.0:
            self.imu_accel = self.imu_accel + self.imu_accel_noise_std * torch.randn_like(self.imu_accel)

        self.estimated_quat = self._madgwick_imu_update(self.estimated_quat, self.imu_gyro, self.imu_accel, self.step_dt)
        self.estimated_gravity = quat_apply_inverse(self.estimated_quat, self.gravity_vec)

        # heading command follow-up
        if self._gym_cfg.commands.heading_command:
            forward = quat_apply(base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        resample_ids = (
            self.episode_length_buf % int(self._gym_cfg.commands.resampling_time / self.step_dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_ids)

        if self.push_robots and (self.common_step_counter % self.push_interval_steps == 0):
            self._push_robots()

        self._update_stride_tracking()

    def _update_stride_tracking(self):
        """Per-foot stride length: horizontal distance between successive touchdowns of the
        same foot. Held at its last measured value between touchdowns (like the original's
        feet_air_time bookkeeping), for use by the stride_length reward."""
        contact_force_z = self._contact_sensor.data.net_forces_w[:, self._feet_contact_ids, 2]
        in_contact = contact_force_z > 1.0
        touchdown = in_contact & ~self._last_foot_contact
        self._last_foot_contact = in_contact

        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids, :2]
        new_stride = torch.norm(feet_pos_w - self._last_touchdown_pos_w, dim=-1)
        self._stride_length = torch.where(touchdown, new_stride, self._stride_length)
        self._last_touchdown_pos_w = torch.where(touchdown.unsqueeze(-1), feet_pos_w, self._last_touchdown_pos_w)

    def _push_robots(self):
        """Random push: emulate an impulse by setting a random base xy velocity for all envs."""
        vel_w = self._robot.data.root_vel_w.clone()
        vel_w[:, :2] = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * self.max_push_vel_xy
        self._robot.write_root_velocity_to_sim(vel_w)

    def _madgwick_imu_update(self, q_wxyz, gyro, accel, dt):
        eps = 1e-8
        q1, q2, q3, q4 = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
        gx, gy, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]

        accel_norm = torch.linalg.norm(accel, dim=1, keepdim=True)
        valid_accel = (accel_norm[:, 0] >= self.madgwick_accel_min) & (accel_norm[:, 0] <= self.madgwick_accel_max)
        a = accel / torch.clamp(accel_norm, min=eps)
        ax, ay, az = a[:, 0], a[:, 1], a[:, 2]

        f1 = 2.0 * (q2 * q4 - q1 * q3) - ax
        f2 = 2.0 * (q1 * q2 + q3 * q4) - ay
        f3 = 2.0 * (0.5 - q2 * q2 - q3 * q3) - az

        s1 = -2.0 * q3 * f1 + 2.0 * q2 * f2
        s2 = 2.0 * q4 * f1 + 2.0 * q1 * f2 - 4.0 * q2 * f3
        s3 = -2.0 * q1 * f1 + 2.0 * q4 * f2 - 4.0 * q3 * f3
        s4 = 2.0 * q2 * f1 + 2.0 * q3 * f2

        step = torch.stack((s1, s2, s3, s4), dim=1)
        step_norm = torch.linalg.norm(step, dim=1, keepdim=True)
        step = step / torch.clamp(step_norm, min=eps)
        use_correction = valid_accel & (step_norm[:, 0] > eps)
        step = torch.where(use_correction[:, None], step, torch.zeros_like(step))

        q_dot_gyro = 0.5 * torch.stack(
            (
                -q2 * gx - q3 * gy - q4 * gz,
                q1 * gx + q3 * gz - q4 * gy,
                q1 * gy - q2 * gz + q4 * gx,
                q1 * gz + q2 * gy - q3 * gx,
            ),
            dim=1,
        )
        q_dot = q_dot_gyro - self.madgwick_beta * step
        q_new = torch.stack((q1, q2, q3, q4), dim=1) + q_dot * dt
        q_new = q_new / torch.clamp(torch.linalg.norm(q_new, dim=1, keepdim=True), min=eps)
        return q_new

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        obs_scales = self._gym_cfg.normalization.obs_scales
        dof_pos_policy = self._robot.data.joint_pos[:, self._policy_to_sim_idx]
        dof_vel_policy = self._robot.data.joint_vel[:, self._policy_to_sim_idx]

        obs_proprio = torch.cat(
            (
                self.base_lin_vel * obs_scales.lin_vel,
                self.base_ang_vel * obs_scales.ang_vel,
                self.estimated_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.commands[:, 4:5],
                (dof_pos_policy - self.default_dof_pos_policy) * obs_scales.dof_pos,
                dof_vel_policy * obs_scales.dof_vel,
                self.action_history_buf[:, -1],
            ),
            dim=-1,
        )

        noise_scales = self._gym_cfg.noise.noise_scales
        noise_level = self._gym_cfg.noise.noise_level
        noise_vec = torch.cat(
            (
                torch.zeros(3),
                torch.ones(3) * noise_scales.ang_vel * noise_level,
                torch.ones(3) * noise_scales.gravity * noise_level,
                torch.zeros(3),
                torch.zeros(1),
                torch.ones(12) * noise_scales.dof_pos * noise_level * obs_scales.dof_pos,
                torch.ones(12) * noise_scales.dof_vel * noise_level * obs_scales.dof_vel,
                torch.zeros(self.num_actions),
            ),
            dim=0,
        ).to(self.device)
        if self._gym_cfg.noise.add_noise:
            obs_proprio = obs_proprio + (2 * torch.rand_like(obs_proprio) - 1) * noise_vec

        obs = torch.cat([obs_proprio, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_proprio] * self.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_proprio.unsqueeze(1)], dim=1),
        )
        return {"policy": obs}

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        dof_vel = self._robot.data.joint_vel
        applied_torque = self._robot.data.applied_torque
        # terrain-relative base height (mean of root_z - ground_z over a 63-point
        # grid under the base), equivalent of the original _get_base_heights().
        terrain_height_under_base = self._height_scanner.data.ray_hits_w[..., 2].mean(dim=1)
        base_height = self._robot.data.root_pos_w[:, 2] - terrain_height_under_base

        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids, :]
        feet_vel_w = self._robot.data.body_lin_vel_w[:, self._feet_ids, :]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        cur_footpos_translated = feet_pos_w - root_pos_w.unsqueeze(1)
        cur_footvel_translated = feet_vel_w - self._robot.data.root_lin_vel_w.unsqueeze(1)
        footpos_body = torch.zeros_like(cur_footpos_translated)
        footvel_body = torch.zeros_like(cur_footvel_translated)
        for i in range(len(self._feet_ids)):
            footpos_body[:, i, :] = quat_apply_inverse(root_quat_w, cur_footpos_translated[:, i, :])
            footvel_body[:, i, :] = quat_apply_inverse(root_quat_w, cur_footvel_translated[:, i, :])

        raw = {}
        raw["lin_vel_z"] = torch.square(self.base_lin_vel[:, 2])
        raw["ang_vel_xy"] = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        raw["orientation"] = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        raw["base_height"] = torch.square(base_height - self._gym_cfg.rewards.base_height_target)
        raw["powers"] = torch.sum(torch.abs(applied_torque) * torch.abs(dof_vel), dim=1)
        raw["dof_acc"] = torch.sum(torch.square((self._last_dof_vel - dof_vel) / self.step_dt), dim=1)
        raw["action_rate"] = torch.sum(torch.square(self._last_actions - self._actions), dim=1)
        raw["action_smoothness"] = torch.sum(
            torch.square(
                self.action_history_buf[:, -1, :]
                - 2 * self.action_history_buf[:, -2, :]
                + self.action_history_buf[:, -3, :]
            ),
            dim=1,
        )
        height_error = torch.square(
            footpos_body[:, :, 2] - self._gym_cfg.rewards.clearance_height_target
        ).view(self.num_envs, -1)
        foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_body[:, :, :2]), dim=2)).view(self.num_envs, -1)
        raw["foot_clearance"] = torch.sum(height_error * foot_lateral_vel, dim=1)

        # Marching on the spot satisfies both tracking_lin_vel (base stays put) and
        # stride_length (a foot lands where it took off, so stride is ~0), leaving nothing
        # to separate it from actually standing still. Charge for foot motion in the world
        # frame while the robot is told to hold position.
        standing_cmd = (torch.norm(self.commands[:, :2], dim=1) == 0.0) & (self.commands[:, 4] == 0.0)
        raw["stand_still"] = torch.norm(feet_vel_w, dim=-1).sum(dim=1) * standing_cmd.float()

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        raw["tracking_lin_vel"] = torch.exp(-lin_vel_error / self._gym_cfg.rewards.tracking_sigma)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        raw["tracking_ang_vel"] = torch.exp(-ang_vel_error / self._gym_cfg.rewards.tracking_sigma)
        stride_error = torch.square(self.commands[:, 4] - self._stride_length.mean(dim=1))
        raw["stride_length"] = torch.exp(-stride_error / self._gym_cfg.rewards.tracking_sigma_stride)
        if not self._stride_reward_enabled:
            raw["stride_length"] = torch.zeros_like(raw["stride_length"])

        # knees on the ground: count how many calf links are in contact, like the
        # original _reward_collision() over penalised_contact_indices.
        knee_forces = self._contact_sensor.data.net_forces_w[:, self._knee_contact_ids, :]
        raw["collision"] = torch.sum(torch.norm(knee_forces, dim=-1) > 0.1, dim=1).float()

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, scale in self.reward_scales.items():
            term = raw[name] * scale
            reward += term
            self._episode_sums[name] += term
        if self._gym_cfg.rewards.only_positive_rewards:
            reward = torch.clip(reward, min=0.0)

        # Termination penalties, added after the positive-only clip above so they aren't
        # zeroed out by it (mirrors the original legged_robot.py termination reward). The
        # two causes are priced separately: falling is what an untrained policy does all
        # the time, riding the torque limit is a choice it makes.
        scales = self._gym_cfg.rewards.scales
        torque_penalty = (self._torque_limit_exceeded & ~self._fell).float() * scales.termination
        fall_penalty = self._fell.float() * scales.fall
        reward += torque_penalty + fall_penalty
        self._episode_sums["termination"] += torque_penalty
        self._episode_sums["fall"] += fall_penalty

        self._last_actions = self._actions.clone()
        self._last_dof_vel = dof_vel.clone()
        self._last_torques = applied_torque.clone()
        return reward

    # ------------------------------------------------------------------
    # termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_post_physics_buffers()
        if self._termination_contact_ids:
            forces = self._contact_sensor.data.net_forces_w[:, self._termination_contact_ids, :]
            self._fell = (torch.norm(forces, dim=-1) > 1.0).any(dim=1)
        died = self._torque_limit_exceeded | self._fell
        time_out = self.episode_length_buf > self.max_episode_length
        return died, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        r = self.command_ranges
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r["lin_vel_x"])
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r["lin_vel_y"])
        if self._gym_cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch.empty(len(env_ids), device=self.device).uniform_(*r["heading"])
        else:
            self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r["ang_vel_yaw"])
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        # Deliberate stand-still commands on top of whatever landed in the dead zone above,
        # so the share stays put as the command curriculum widens the sampling range.
        hold = torch.rand(len(env_ids), device=self.device) < self._gym_cfg.commands.stand_still_prob
        self.commands[env_ids[hold], :2] = 0.0
        # Stride only makes sense alongside a velocity: standing still means a stride of
        # zero, so a non-zero stride command at zero velocity asks for two incompatible
        # things and pays more for walking than for holding still. Zero it out to match.
        stride = torch.empty(len(env_ids), device=self.device).uniform_(*r["stride_length"])
        standing = torch.norm(self.commands[env_ids, :2], dim=1) == 0.0
        self.commands[env_ids, 4] = torch.where(standing, torch.zeros_like(stride), stride)

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Game-inspired terrain curriculum: robots that walked far enough (relative
        to their env origin) move to harder terrain rows; robots that fell short of
        half their commanded distance move to easier ones. Equivalent of the
        original _update_terrain_curriculum(), using TerrainImporter.update_env_origins()
        which implements the same level/type bookkeeping.
        """
        distance = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self.cfg.terrain.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        ) & ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _update_command_curriculum(self, env_ids: torch.Tensor):
        """Widen the commandable velocity range once tracking reward is good enough.
        Equivalent of the original _update_command_curriculum()."""
        if "tracking_lin_vel" not in self._episode_sums:
            return
        if (
            torch.mean(self._episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            c = self._gym_cfg.commands
            r = self.command_ranges
            r["lin_vel_x"][0] = float(max(r["lin_vel_x"][0] - 0.1, -c.max_backward_curriculum))
            r["lin_vel_x"][1] = float(min(r["lin_vel_x"][1] + 0.1, c.max_forward_curriculum))
            r["lin_vel_y"][0] = float(max(r["lin_vel_y"][0] - 0.1, -c.max_lat_curriculum))
            r["lin_vel_y"][1] = float(min(r["lin_vel_y"][1] + 0.1, c.max_lat_curriculum))

    def _update_stride_curriculum(self, env_ids: torch.Tensor):
        """Switch the stride_length reward on once velocity tracking is good enough.

        Measured the same way _update_command_curriculum() measures it - episode sum over
        max_episode_length - so the bar covers both tracking quality and surviving the
        episode. A policy that still terminates early cannot clear it, which is the point:
        there is no gait to modulate until the robot stays up.
        """
        if self._stride_reward_enabled or "tracking_lin_vel" not in self._episode_sums:
            return
        mean_rate = torch.mean(self._episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
        if mean_rate > self._gym_cfg.rewards.stride_curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self._stride_reward_enabled = True
            print("[curriculum] velocity tracking passed the bar - stride_length reward enabled")

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if len(env_ids) == 0:
            return

        self._update_terrain_curriculum(env_ids)
        if self._gym_cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self._update_command_curriculum(env_ids)
        self._update_stride_curriculum(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        # dof reset: random 0.5-1.5x default pose, in sim order
        dof_pos_policy = self.default_dof_pos_policy.expand(n, -1) * (
            0.5 + torch.rand(n, self.num_actions, device=self.device)
        )
        dof_pos_sim = dof_pos_policy[:, self._sim_to_policy_idx]
        dof_vel_sim = torch.zeros(n, self.num_actions, device=self.device)
        self._robot.write_joint_state_to_sim(dof_pos_sim, dof_vel_sim, None, env_ids)

        # root state reset
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, :2] += (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
        # upright spawn, facing a random direction (roll/pitch=0, random yaw) - not the
        # original's fully random orientation (any tilt, incl. upside-down), which was
        # meant for fall-recovery training and fights the torque-limit termination here.
        yaw = torch.rand(n, device=self.device) * 2 * torch.pi
        default_root_state[:, 3] = torch.cos(yaw * 0.5)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(yaw * 0.5)
        # spawn height: 0.35-0.42m above the terrain origin (was 0.42-0.62m, i.e. up to a
        # 0.2m drop) - a lower, gentler drop so landing doesn't itself demand near-limit
        # torque and trip the torque-limit termination right at episode start.
        default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + (
            0.35 + torch.rand(n, device=self.device) * (0.42 - 0.35)
        )
        default_root_state[:, 7:13] = (torch.rand(n, 6, device=self.device) * 2.0 - 1.0) * 0.5
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._last_actions[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._last_torques[env_ids] = 0.0
        self.obs_history_buf[env_ids] = 0.0
        self.action_history_buf[env_ids] = 0.0

        # stride tracking: approximate the reset touchdown position with the new base xy
        # (exact foot xy isn't available yet - it only updates after the next sim step),
        # so the first post-reset stride reading is a one-off, bounded approximation
        # rather than a stale cross-episode value.
        self._last_foot_contact[env_ids] = False
        self._stride_length[env_ids] = 0.0
        self._last_touchdown_pos_w[env_ids] = default_root_state[:, :2].unsqueeze(1)

        self._resample_commands(env_ids)

        # Madgwick estimator: reset to true orientation at the new state
        self.estimated_quat[env_ids] = default_root_state[:, 3:7]
        self.estimated_gravity[env_ids] = quat_apply_inverse(self.estimated_quat[env_ids], self.gravity_vec[env_ids])
        self.prev_base_lin_vel_world[env_ids] = default_root_state[:, 7:10]
        self.imu_gyro[env_ids] = 0.0
        self.imu_accel[env_ids] = 0.0

        extras = {}
        for key in self._episode_sums.keys():
            extras["rew_" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["terrain_level"] = torch.mean(self._terrain.terrain_levels.float())
        if self._gym_cfg.commands.curriculum:
            extras["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        self.extras["episode"] = extras
        self.extras["time_outs"] = self.reset_time_outs
