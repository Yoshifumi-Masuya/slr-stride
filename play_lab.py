"""Verification: load an existing trained checkpoint and run it in the new
Isaac Lab Go2 env (rough terrain curriculum, Phase 2), to sanity-check that the
observation convention (xyzw/reindex-compatible) still lets the policy walk.

Usage:
    conda activate isaaclab
    cd ~/my_workspace/IsaacLab_23
    ./isaaclab.sh -p /home/huro-lab/my_workspace/SLR-master-g/play_lab.py --headless --num_envs 4 --steps 400

Note on memory: the full terrain curriculum is a --terrain_rows x --terrain_cols
grid of 8x8m tiles (default 10x20 = 200 tiles), generated regardless of
--num_envs. That is fine headless, but in GUI mode (with --headless omitted)
the RTX renderer's VRAM/RAM usage on top of it can OOM on machines with ~16GB
RAM - pass e.g. --terrain_rows 2 --terrain_cols 5 for a light-weight GUI check.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play the Go2 (Isaac Lab, Phase 2: terrain curriculum)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="logs/rough_go2/Sep18_16-24-42_walk_stride_v2/model_7700.pt",
    help="Checkpoint path relative to the repo root.",
)
parser.add_argument(
    "--terrain_rows", type=int, default=None, help="Override terrain curriculum rows (levels). Default: Go2RoughCfg value (10)."
)
parser.add_argument(
    "--terrain_cols", type=int, default=None, help="Override terrain curriculum cols (types). Default: Go2RoughCfg value (20)."
)
parser.add_argument(
    "--terrain_border",
    type=float,
    default=None,
    help="Override the flat border width (m) around the terrain grid. Default: Go2RoughCfg value (25m), which "
    "dwarfs a small --terrain_rows/--terrain_cols grid; try e.g. 2-4 for a tight GUI view.",
)
parser.add_argument(
    "--print_env", type=int, default=0, help="Env index whose per-joint angle/angular velocity/torque to print."
)
parser.add_argument(
    "--stride_length", type=float, default=0.25, help="Target stride length command [m] (range 0.1-0.4)."
)
parser.add_argument(
    "--no_camera_follow",
    action="store_true",
    help="Fix the camera beside where the robot spawns instead of tracking it, so you can "
    "see the path it takes across the frame (a tracking camera moves with the robot and "
    "hides drift). Uses --camera_offset as the offset from the spawn point.",
)
parser.add_argument(
    "--camera_pan",
    type=float,
    default=1.5,
    help="With --no_camera_follow, how far down +x to shift the fixed camera and its aim "
    "point [m]. Puts the spawn point at the left of frame so the walk happens in view.",
)
parser.add_argument(
    "--guide_line",
    type=float,
    default=8.0,
    help="Length [m] of a straight reference line drawn along +x from the spawn point, to "
    "compare the robot's path against. 0 disables it.",
)
parser.add_argument(
    "--random_spawn",
    action="store_true",
    help="Spawn with the random heading and xy jitter that training uses. Off by default "
    "here: playback starts the robot on its env origin facing +x, so it begins on the "
    "guide line and along it rather than spending the first stretch turning onto it.",
)
parser.add_argument(
    "--camera_offset",
    type=float,
    nargs=3,
    #default=[0.0, -2.2, 0.6],
    default=[0, -2.2, 0.8],
    metavar=("X", "Y", "Z"),
    help="Tracking camera position relative to the robot [m]. The default is a side view: "
    "the robot walks along +x here, and the offset is in world axes (it does not rotate "
    "with the robot), so -y puts the camera abeam of it. Try '2 -2 0.8' for 3/4 view.",
)
parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="Write the tracked env's per-step state to this CSV (commands, base pose and "
    "velocity, stride, and every joint's angle/velocity/torque). Prints velocity and "
    "stride tracking RMSE at the end.",
)
parser.add_argument(
    "--export_dir",
    type=str,
    default=None,
    help="Directory to write policy.jit/encoder.jit into. Default: same directory as --checkpoint.",
)
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    choices=["flat", "smooth_slope", "rough_slope", "stairs_up", "stairs_down", "discrete"],
    help="Restrict the whole terrain grid to a single sub-terrain type (e.g. 'smooth_slope' for a "
    "pyramid-shaped hill), so every env spawns on that terrain regardless of which column it lands in.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below needs the simulation app running before it can be imported ---
import csv
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_config import ROOT_DIR
from modules import ActorCriticSLR
from configs.go2_config import Go2RoughCfgPPO

from envs_lab import Go2DirectEnv, Go2DirectEnvCfg, IsaacLabVecEnvWrapper
from envs_lab.go2_direct_env import POLICY_JOINT_ORDER
from envs_lab.utils_lab import class_to_dict


def print_joint_states(wrapped_env, env_id: int):
    """Print each joint's angle, angular velocity, and applied torque for one env."""
    robot = wrapped_env.env._robot
    idx = wrapped_env.env._policy_to_sim_idx
    joint_pos = robot.data.joint_pos[env_id, idx].tolist()
    joint_vel = robot.data.joint_vel[env_id, idx].tolist()
    torque = robot.data.applied_torque[env_id, idx].tolist()

    print(f"  {'joint':<16}{'angle [rad]':>14}{'ang vel [rad/s]':>18}{'torque [N*m]':>14}")
    for name, pos, vel, tau in zip(POLICY_JOINT_ORDER, joint_pos, joint_vel, torque):
        print(f"  {name:<16}{pos:>14.3f}{vel:>18.3f}{tau:>14.3f}")


def print_peak_torques(wrapped_env, peak_torque):
    """Print the per-joint peak |torque| over the whole run, against each joint's limit.

    The termination threshold is soft_torque_limit x the hard limit, so a peak that
    stays under 'limit' can still have ended episodes - see the soft_torque_limit
    comment in configs/go2_config.py.
    """
    limits = wrapped_env.env.torque_limits_policy.tolist()
    soft = wrapped_env.env._gym_cfg.rewards.soft_torque_limit
    peaks = peak_torque.tolist()

    print(f"  {'joint':<16}{'peak |torque|':>16}{'limit [N*m]':>14}{'used':>9}{f'of {soft:.0%} limit':>16}")
    for name, peak, limit in zip(POLICY_JOINT_ORDER, peaks, limits):
        print(
            f"  {name:<16}{peak:>16.3f}{limit:>14.3f}"
            f"{peak / limit * 100.0:>8.1f}%{peak / (limit * soft) * 100.0:>15.1f}%"
        )


def main():
    env_cfg = Go2DirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if not args_cli.no_camera_follow:
        # Track the robot instead of staying put, so it does not walk out of frame.
        # The viewport controller re-aims the camera on every render when origin_type
        # is "asset_root".
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = args_cli.print_env
        env_cfg.viewer.eye = tuple(args_cli.camera_offset)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    if args_cli.terrain_rows is not None:
        env_cfg.terrain.terrain_generator.num_rows = args_cli.terrain_rows
    if args_cli.terrain_cols is not None:
        env_cfg.terrain.terrain_generator.num_cols = args_cli.terrain_cols
    if args_cli.terrain_border is not None:
        env_cfg.terrain.terrain_generator.border_width = args_cli.terrain_border
    if args_cli.terrain_type is not None:
        chosen = env_cfg.terrain.terrain_generator.sub_terrains[args_cli.terrain_type]
        chosen.proportion = 1.0
        env_cfg.terrain.terrain_generator.sub_terrains = {args_cli.terrain_type: chosen}
    env_cfg.terrain.max_init_terrain_level = min(
        env_cfg.terrain.max_init_terrain_level, env_cfg.terrain.terrain_generator.num_rows - 1
    )

    env = Go2DirectEnv(cfg=env_cfg, render_mode=None)
    wrapped_env = IsaacLabVecEnvWrapper(env)

    spawn = env._terrain.env_origins[args_cli.print_env].cpu().numpy()

    if args_cli.guide_line > 0.0:
        # Straight reference line along +x from the spawn point. Judging drift by eye is
        # hard without something fixed to compare against. Visual only - no rigid body or
        # collider, so it cannot be walked into.
        length = args_cli.guide_line
        line = sim_utils.CuboidCfg(
            size=(length, 0.02, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        )
        line.func("/World/GuideLine", line, translation=(spawn[0] + length / 2, spawn[1], spawn[2] + 0.002))
        # One tick per metre, so distance covered can be read straight off the view.
        tick = sim_utils.CuboidCfg(
            size=(0.02, 0.30, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.2)),
        )
        for m in range(1, int(length) + 1):
            tick.func(f"/World/GuideTick{m}", tick, translation=(spawn[0] + m, spawn[1], spawn[2] + 0.003))

    if args_cli.no_camera_follow and env.viewport_camera_controller is not None:
        # Park the camera beside where the robot starts rather than at the world origin
        # (the ViewerCfg default), which is nowhere near the terrain tile it spawns on.
        # Fixed rather than tracking so the robot's path across the frame is visible -
        # a tracking camera hides drift by moving with it. Pan shifts both eye and aim
        # point down +x, which puts the spawn at the left of frame and leaves the rest of
        # it for the robot to walk into.
        pan = np.array([args_cli.camera_pan, 0.0, 0.0])
        eye = spawn + np.asarray(args_cli.camera_offset) + pan
        env.viewport_camera_controller.update_view_location(
            eye=eye.tolist(), lookat=(spawn + pan + np.array([0.0, 0.0, 0.3])).tolist()
        )

    train_cfg = Go2RoughCfgPPO()
    policy_cfg_dict = class_to_dict(train_cfg.policy)

    policy = ActorCriticSLR(
        num_props=wrapped_env.cfg.env.n_proprio,
        num_hist=wrapped_env.cfg.env.history_len,
        num_actions=wrapped_env.num_actions,
        **policy_cfg_dict,
    )
    ckpt_path = os.path.join(ROOT_DIR, args_cli.checkpoint)
    model_dict = torch.load(ckpt_path, map_location=wrapped_env.device)
    policy.load_state_dict(model_dict["model_state_dict"])
    policy.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    export_dir = args_cli.export_dir or os.path.dirname(ckpt_path)
    os.makedirs(export_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(export_dir)
    try:
        policy = policy.to("cpu")
        policy.save_torch_jit_policy()
    finally:
        os.chdir(cwd)
    print(f"Exported policy.jit / encoder.jit to {export_dir}")

    policy = policy.to(wrapped_env.device)

    def place_on_line():
        """Start every robot on its env origin, pointing down +x.

        Training randomises the spawn yaw and jitters xy by up to a metre, both of which
        the policy needs to cope with - but they make a fixed camera useless for judging
        whether the walk is straight, since the robot neither starts on the reference
        line nor along it. Re-applied after resets, which would otherwise hand back a
        random pose mid-run. Height is left alone so the drop-in still happens.
        """
        robot = wrapped_env.env._robot
        pose = robot.data.root_state_w[:, :7].clone()
        pose[:, :2] = wrapped_env.env._terrain.env_origins[:, :2]
        pose[:, 3] = 1.0  # quaternion (w, x, y, z) = identity, i.e. yaw 0
        pose[:, 4:7] = 0.0
        robot.write_root_pose_to_sim(pose)
        # Reset spawns the robot with a random shove (+-0.5 m/s, rad/s); keeping it after
        # moving the robot turns into a lurch off the line before it has taken a step.
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))

    if not args_cli.random_spawn:
        place_on_line()

    # A marker dropped at every touchdown of the tracked env's feet. The gap between
    # consecutive marks of the same foot is the stride the robot actually took, which is
    # what the stride_length command is asking for - so it can be compared against the
    # metre ticks by eye, without needing text in the viewport.
    footprints = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/World/Footprints",
            markers={
                "step": sim_utils.CylinderCfg(
                    radius=0.03,
                    height=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.95)),
                )
            },
        )
    )
    footprint_pos = []
    prev_contact = torch.zeros(len(env._feet_ids), dtype=torch.bool, device=wrapped_env.device)

    obs = wrapped_env.get_observations()

    base_heights = []
    reset_count = 0
    reward_sum = torch.zeros(wrapped_env.num_envs, device=wrapped_env.device)
    # per-joint peak |torque| over the run, max'd across envs, in policy joint order
    peak_torque = torch.zeros(wrapped_env.env.num_actions, device=wrapped_env.device)
    start_xy = env._robot.data.root_pos_w[args_cli.print_env, :2].clone()

    csv_file = csv_writer = None
    csv_rows = []
    if args_cli.csv:
        csv_file = open(args_cli.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["step", "t"]
            + ["cmd_vx", "cmd_vy", "cmd_wz", "cmd_stride"]
            + ["base_x", "base_y", "base_z", "vx", "vy", "vz", "wz"]
            + ["stride_measured", "stride_valid", "dist_from_start", "lateral_offset"]
            + [f"q_{n}" for n in POLICY_JOINT_ORDER]
            + [f"dq_{n}" for n in POLICY_JOINT_ORDER]
            + [f"tau_{n}" for n in POLICY_JOINT_ORDER]
        )

    for step in range(args_cli.steps):
        wrapped_env.env.commands[:, 0] = 0.6
        wrapped_env.env.commands[:, 1] = 0.0
        wrapped_env.env.commands[:, 3] = 0.0
        wrapped_env.env.commands[:, 4] = args_cli.stride_length

        with torch.no_grad():
            actions = policy.act_inference(obs)
        obs, _, rewards, dones, _ = wrapped_env.step(actions)

        reward_sum += rewards
        reset_count += int(dones.sum().item())
        if dones.any() and not args_cli.random_spawn:
            place_on_line()
        base_heights.append(wrapped_env.env._robot.data.root_pos_w[:, 2].mean().item())

        ev = args_cli.print_env
        contact = env._contact_sensor.data.net_forces_w[ev, env._feet_contact_ids, 2] > 1.0
        for foot in (contact & ~prev_contact).nonzero(as_tuple=False).flatten().tolist():
            pos = env._robot.data.body_pos_w[ev, env._feet_ids[foot]].clone()
            pos[2] = spawn[2] + 0.004
            footprint_pos.append(pos)
        prev_contact = contact
        if footprint_pos:
            footprints.visualize(translations=torch.stack(footprint_pos))

        torque_policy = wrapped_env.env._robot.data.applied_torque[:, wrapped_env.env._policy_to_sim_idx]
        peak_torque = torch.maximum(peak_torque, torque_policy.abs().amax(dim=0))

        if csv_writer is not None:
            idx = env._policy_to_sim_idx
            pos = env._robot.data.root_pos_w[ev]
            stride = env._stride_length[ev].mean().item()
            row = (
                [step, step * env.step_dt]
                + [env.commands[ev, i].item() for i in (0, 1, 2, 4)]
                + pos.tolist()
                + env.base_lin_vel[ev].tolist()
                + [env.base_ang_vel[ev, 2].item()]
                # stride stays 0 until the first touchdown; the flag marks rows where it
                # is a real measurement, so the RMSE below can skip the rest.
                + [stride, int(stride > 0.0)]
                + [torch.norm(pos[:2] - start_xy).item(), (pos[1] - start_xy[1]).item()]
                + env._robot.data.joint_pos[ev, idx].tolist()
                + env._robot.data.joint_vel[ev, idx].tolist()
                + env._robot.data.applied_torque[ev, idx].tolist()
            )
            csv_writer.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
            csv_rows.append(row)

        if step % 50 == 0:
            travelled = torch.norm(env._robot.data.root_pos_w[ev, :2] - start_xy).item()
            measured_stride = env._stride_length[ev].mean().item()
            print(
                f"step {step:4d} | mean base height {base_heights[-1]:.3f} m"
                f" | mean reward/env {reward_sum.mean().item() / max(step, 1):.4f}"
                f" | resets so far {reset_count}"
            )
            print(
                f"  travelled {travelled:6.2f} m | stride cmd {args_cli.stride_length:.3f} m"
                f" -> measured {measured_stride:.3f} m ({len(footprint_pos)} footfalls)"
            )
            print_joint_states(wrapped_env, args_cli.print_env)

    print("-" * 60)
    print(f"steps: {args_cli.steps}, num_envs: {args_cli.num_envs}")
    print(f"mean base height over run: {sum(base_heights) / len(base_heights):.3f} m")
    print(f"final base height: {base_heights[-1]:.3f} m")
    print(f"total resets (falls/timeouts): {reset_count}")
    print(f"mean reward/env/step: {reward_sum.mean().item() / args_cli.steps:.4f}")
    print(f"peak |torque| per joint (max over all {args_cli.num_envs} envs and all steps):")
    print_peak_torques(wrapped_env, peak_torque)

    if csv_writer is not None:
        csv_file.close()
        a = np.asarray(csv_rows, dtype=float)
        cmd_vx, cmd_vy, cmd_stride = a[:, 2], a[:, 3], a[:, 5]
        vx, vy = a[:, 9], a[:, 10]
        stride, valid = a[:, 13], a[:, 14].astype(bool)
        rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))
        print(f"\nwrote {len(a)} rows for env {args_cli.print_env} to {args_cli.csv}")
        print(f"  RMSE vx      {rmse(cmd_vx - vx):.4f} m/s")
        print(f"  RMSE vy      {rmse(cmd_vy - vy):.4f} m/s")
        print(f"  RMSE v (xy)  {rmse(np.hypot(cmd_vx - vx, cmd_vy - vy)):.4f} m/s")
        if valid.any():
            print(
                f"  RMSE stride  {rmse(cmd_stride[valid] - stride[valid]):.4f} m"
                f"  ({valid.sum()}/{len(a)} steps after the first touchdown)"
            )
        else:
            print("  RMSE stride  n/a (no touchdown recorded)")


if __name__ == "__main__":
    main()
    simulation_app.close()
