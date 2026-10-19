"""Kinematic replay of a real Go2 LowState log in Isaac Sim.

A LowState log holds measurements only - it has no LowCmd (the policy's target joint
angles), and no base position or linear velocity. So it cannot drive the simulator in
the usual sense. What it *can* do is play back: the logged joint angles are written
straight into the articulation and the logged IMU quaternion becomes the base
orientation, with physics neutralised so nothing the simulator computes can fight the
data. What you see is exactly what the robot's joints did.

The one thing the log does not contain is where the base was. It is reconstructed here
by contact odometry: while a foot is on the ground its world position is fixed, so the
base moved by minus the displacement of that foot (forward kinematics from the logged
joint angles gives the foot position in the base frame). Height comes from putting the
lowest loaded foot on the ground. This drifts over a long log and is meaningless while
the robot is off the ground, but over a walk it is accurate enough to watch.

Usage:
    conda activate isaaclab
    cd ~/my_workspace/IsaacLab_23
    ./isaaclab.sh -p /home/huro-lab/my_workspace/SLR-master-pbl/replay_lowstate.py \
        --csv_in /home/huro-lab/my_workspace/SLR-master-pbl/lowstate.csv --start 12 --end 30
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a Go2 LowState log in Isaac Sim (kinematic)")
parser.add_argument("--csv_in", type=str, default="lowstate.csv", help="LowState CSV to replay.")
parser.add_argument("--start", type=float, default=0.0, help="Start time in the log [s].")
parser.add_argument("--end", type=float, default=None, help="End time in the log [s]. Default: end of log.")
parser.add_argument(
    "--sample_hz",
    type=float,
    default=50.0,
    help="Replay rate [Hz]. The log is 500 Hz; the default matches the policy rate, so one "
    "replay frame is one control step.",
)
parser.add_argument(
    "--contact_threshold",
    type=float,
    default=25.0,
    help="foot_force above which a foot counts as loaded [N]. The odometry and the footprint "
    "markers both key off this. Unloaded feet read ~13-15 on this robot, a standing one ~30-45.",
)
parser.add_argument(
    "--no_align",
    action="store_true",
    help="Keep the log's own heading instead of rotating the whole trajectory so the robot "
    "starts facing +x. Aligned by default so the walk runs along the guide line.",
)
parser.add_argument(
    "--hold_base",
    action="store_true",
    help="Pin the base at the start point and only move the joints, skipping the odometry. "
    "Use this when the robot is held off the ground (no contacts -> no odometry) or when you "
    "only want to look at the leg motion.",
)
parser.add_argument("--no_camera_follow", action="store_true", help="Fix the camera beside the start point.")
parser.add_argument(
    "--camera_offset",
    type=float,
    nargs=3,
    default=[0.0, -2.2, 0.8],
    metavar=("X", "Y", "Z"),
    help="Camera position relative to the robot [m], in world axes. Default is a side view.",
)
parser.add_argument("--camera_pan", type=float, default=1.5, help="With --no_camera_follow, shift the camera down +x [m].")
parser.add_argument("--guide_line", type=float, default=8.0, help="Length of the +x reference line [m]. 0 disables it.")
parser.add_argument("--footprints", action="store_true", help="Drop a marker at every touchdown.")
parser.add_argument("--loop", action="store_true", help="Replay the segment over and over until interrupted.")
parser.add_argument("--csv_out", type=str, default=None, help="Write the reconstructed per-frame state to this CSV.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Omniverse swaps sys.stdout for its own logger once the app is up, which silently
# swallows plain print(). Hold on to the real stream first and report through that.
_STDOUT = sys.stdout

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def report(*parts):
    print(*parts, file=_STDOUT, flush=True)

# --- everything below needs the simulation app running before it can be imported ---
import csv
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs_lab import Go2DirectEnv, Go2DirectEnvCfg
from envs_lab.go2_direct_env import POLICY_JOINT_ORDER

# Unitree's LowState orders the motors FR, FL, RR, RL x (hip, thigh, calf), which is
# already POLICY_JOINT_ORDER - no reindexing between log and policy convention.
LEG_NAMES = ["FR", "FL", "RR", "RL"]
LEG_SIGN_X = np.array([+1.0, +1.0, -1.0, -1.0])  # FR/FL are the front pair
LEG_SIGN_Y = np.array([-1.0, +1.0, -1.0, +1.0])  # FL/RL are the left pair
# From resources/go2/urdf/go2.urdf
HIP_X, HIP_Y, THIGH_Y = 0.1934, 0.0465, 0.0955
L_THIGH = L_CALF = 0.213
FOOT_RADIUS = 0.02


def load_lowstate(path):
    """Read the columns a replay needs out of a LowState CSV."""
    with open(path) as f:
        reader = csv.reader(f)
        cols = next(reader)
        arr = np.array([row for row in reader], dtype=np.float64)
    ci = {c: i for i, c in enumerate(cols)}
    missing = [c for c in ("time_s", "imu_state.quaternion[0]", "motor_state[0].q") if c not in ci]
    if missing:
        raise SystemExit(f"{path} does not look like a LowState log - missing columns: {missing}")
    return {
        "t": arr[:, ci["time_s"]],
        # (N, 4, 3) - leg, then hip/thigh/calf
        "q": np.stack([arr[:, ci[f"motor_state[{m}].q"]] for m in range(12)], 1).reshape(-1, 4, 3),
        "dq": np.stack([arr[:, ci[f"motor_state[{m}].dq"]] for m in range(12)], 1).reshape(-1, 4, 3),
        "tau": np.stack([arr[:, ci[f"motor_state[{m}].tau_est"]] for m in range(12)], 1),
        "quat": np.stack([arr[:, ci[f"imu_state.quaternion[{i}]"]] for i in range(4)], 1),  # (w, x, y, z)
        "gyro": np.stack([arr[:, ci[f"imu_state.gyroscope[{i}]"]] for i in range(3)], 1),
        "foot_force": np.stack([arr[:, ci[f"foot_force[{i}]"]] for i in range(4)], 1),
    }


def forward_kinematics(q):
    """Foot positions in the base frame, (N, 4, 3), from the joint angles (N, 4, 3).

    Each leg is hip (roll about +x), thigh and calf (both pitch about +y), with the
    link offsets taken from the URDF above.
    """
    q_hip, q_thigh, q_calf = q[..., 0], q[..., 1], q[..., 2]
    c, s = np.cos(q_hip), np.sin(q_hip)

    px = np.broadcast_to(LEG_SIGN_X * HIP_X, q_hip.shape).copy()
    py = np.broadcast_to(LEG_SIGN_Y * HIP_Y, q_hip.shape).copy()
    pz = np.zeros_like(q_hip)
    # the thigh's sideways offset, rotated by the hip roll
    offset_y = LEG_SIGN_Y * THIGH_Y
    py = py + c * offset_y
    pz = pz + s * offset_y
    # the thigh and calf links, each pointing down and pitched, then rolled by the hip
    for pitch, length in ((q_thigh, L_THIGH), (q_thigh + q_calf, L_CALF)):
        local_x = -length * np.sin(pitch)
        local_z = -length * np.cos(pitch)
        px = px + local_x
        py = py - s * local_z
        pz = pz + c * local_z
    return np.stack([px, py, pz], -1)


def quat_to_matrix(quat):
    """(N, 4) quaternions in (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    m = np.empty((len(quat), 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z); m[:, 0, 1] = 2 * (x * y - z * w); m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w); m[:, 1, 1] = 1 - 2 * (x * x + z * z); m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w); m[:, 2, 1] = 2 * (y * z + x * w); m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def yaw_of(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotate_yaw(quat, angle):
    """Pre-multiply each quaternion by a rotation of `angle` about world +z."""
    h = angle / 2.0
    qw, qz = np.cos(h), np.sin(h)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.stack([qw * w - qz * z, qw * x - qz * y, qw * y + qz * x, qw * z + qz * w], 1)


def reconstruct_base(log, contact, hold_base):
    """Base position over time, from contact odometry. Returns (N, 3) and the foot world offsets.

    While a foot is loaded on two consecutive samples, its world position has not moved,
    so the base moved by minus that foot's displacement. Averaging over every foot that
    stayed down damps the noise from one foot slipping or the force reading flickering.
    """
    feet_base = forward_kinematics(log["q"])
    rot = quat_to_matrix(log["quat"])
    # foot position relative to the base, expressed in world axes
    feet_rel = np.einsum("nij,nkj->nki", rot, feet_base)

    pos = np.zeros((len(log["t"]), 3))
    stance = contact[0]
    pos[0, 2] = (-feet_rel[0, stance, 2].min() + FOOT_RADIUS) if stance.any() else 0.33
    for k in range(1, len(pos)):
        pos[k, :2] = pos[k - 1, :2]
        if not hold_base:
            held = contact[k] & contact[k - 1]
            if held.any():
                pos[k, :2] -= (feet_rel[k, held] - feet_rel[k - 1, held]).mean(0)[:2]
        stance = contact[k]
        pos[k, 2] = (-feet_rel[k, stance, 2].min() + FOOT_RADIUS) if stance.any() else pos[k - 1, 2]
    return pos, feet_rel


def report_strides(t, pos, feet_rel, contact):
    """Print the stride each foot actually took, measured the way the env measures it.

    A stride is the world-frame distance the foot travelled between one touchdown and
    the next - the same rising-edge rule Go2DirectEnv uses, so these numbers are directly
    comparable to the stride_length command.
    """
    feet_world = pos[:, None, :] + feet_rel
    report("\n  measured strides (touchdown to touchdown)")
    report(f"  {'foot':<6}{'steps':>8}{'mean [m]':>12}{'min':>9}{'max':>9}")
    all_strides = []
    for leg in range(4):
        touchdown = np.flatnonzero(contact[1:, leg] & ~contact[:-1, leg]) + 1
        xy = feet_world[touchdown, leg, :2]
        strides = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([])
        # A foot that is merely re-reading the same touchdown gives a near-zero "stride";
        # drop those so the average reflects real steps.
        strides = strides[strides > 0.02]
        all_strides.append(strides)
        if len(strides):
            report(f"  {LEG_NAMES[leg]:<6}{len(strides):>8}{strides.mean():>12.3f}{strides.min():>9.3f}{strides.max():>9.3f}")
        else:
            report(f"  {LEG_NAMES[leg]:<6}{0:>8}{'-':>12}{'-':>9}{'-':>9}")
    pooled = np.concatenate(all_strides) if any(len(s) for s in all_strides) else np.array([])
    if len(pooled):
        dist = np.linalg.norm(pos[-1, :2] - pos[0, :2])
        speed = dist / max(t[-1] - t[0], 1e-6)
        report(f"\n  all feet: {len(pooled)} steps, stride {pooled.mean():.3f} +- {pooled.std():.3f} m")
        report(f"  travelled {dist:.2f} m in {t[-1] - t[0]:.1f} s -> {speed:.2f} m/s average")


def main():
    log_path = args_cli.csv_in
    if not os.path.isabs(log_path):
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_path)
    log = load_lowstate(log_path)
    report(f"Loaded {log_path}: {len(log['t'])} samples, {log['t'][-1] - log['t'][0]:.1f} s")

    contact = log["foot_force"] > args_cli.contact_threshold
    pos, feet_rel = reconstruct_base(log, contact, args_cli.hold_base)
    quat = log["quat"]

    if not args_cli.no_align:
        # Rotate the whole reconstruction so the robot sets off down +x, along the guide line.
        angle = -yaw_of(quat[:1])[0]
        c, s = np.cos(angle), np.sin(angle)
        pos = np.stack([c * pos[:, 0] - s * pos[:, 1], s * pos[:, 0] + c * pos[:, 1], pos[:, 2]], 1)
        feet_rel = np.stack(
            [c * feet_rel[..., 0] - s * feet_rel[..., 1], s * feet_rel[..., 0] + c * feet_rel[..., 1], feet_rel[..., 2]], -1
        )
        quat = rotate_yaw(quat, angle)

    # Decimate to the replay rate and cut to the requested window.
    end = args_cli.end if args_cli.end is not None else log["t"][-1]
    stride_n = max(1, int(round((1.0 / args_cli.sample_hz) / np.mean(np.diff(log["t"])))))
    keep = np.flatnonzero((log["t"] >= args_cli.start) & (log["t"] <= end))[::stride_n]
    if len(keep) < 2:
        raise SystemExit(f"--start {args_cli.start} --end {end} selects {len(keep)} samples; widen the window.")
    report(
        f"Replaying t={log['t'][keep[0]]:.2f}..{log['t'][keep[-1]]:.2f}s as {len(keep)} frames "
        f"at {args_cli.sample_hz:g} Hz (every {stride_n} samples)"
    )
    report_strides(log["t"][keep], pos[keep], feet_rel[keep], contact[keep])

    env_cfg = Go2DirectEnvCfg()
    env_cfg.scene.num_envs = 1
    # Physics must not get a say: the whole point is that the pose comes from the log.
    # Zero gravity means the re-written state is not fought by a fall between frames.
    env_cfg.sim.gravity = (0.0, 0.0, 0.0)
    # A flat floor - the log has no terrain information, so anything else would be a guess.
    terrain_gen = env_cfg.terrain.terrain_generator
    flat = terrain_gen.sub_terrains["flat"]
    flat.proportion = 1.0
    terrain_gen.sub_terrains = {"flat": flat}
    terrain_gen.num_rows, terrain_gen.num_cols = 2, 2
    terrain_gen.border_width = 4.0
    env_cfg.terrain.max_init_terrain_level = 0

    if not args_cli.no_camera_follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = tuple(args_cli.camera_offset)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    env = Go2DirectEnv(cfg=env_cfg, render_mode=None)
    robot = env._robot
    device = env.device
    origin = env._terrain.env_origins[0].cpu().numpy()

    if args_cli.guide_line > 0.0:
        length = args_cli.guide_line
        line = sim_utils.CuboidCfg(
            size=(length, 0.02, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        )
        line.func("/World/GuideLine", line, translation=(origin[0] + length / 2, origin[1], origin[2] + 0.002))
        tick = sim_utils.CuboidCfg(
            size=(0.02, 0.30, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.2)),
        )
        for m in range(1, int(length) + 1):
            tick.func(f"/World/GuideTick{m}", tick, translation=(origin[0] + m, origin[1], origin[2] + 0.003))

    if args_cli.no_camera_follow and env.viewport_camera_controller is not None:
        pan = np.array([args_cli.camera_pan, 0.0, 0.0])
        eye = origin + np.asarray(args_cli.camera_offset) + pan
        env.viewport_camera_controller.update_view_location(
            eye=eye.tolist(), lookat=(origin + pan + np.array([0.0, 0.0, 0.3])).tolist()
        )

    markers = None
    if args_cli.footprints:
        markers = VisualizationMarkers(
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

    # The log starts wherever the robot happened to be; put that at the env origin so it
    # begins on the guide line.
    # xy is shifted onto the origin; z is left as the odometry's height above the floor,
    # offset by wherever the floor is.
    world_offset = origin - pos[keep[0]]
    world_offset[2] = origin[2]

    sim_idx = env._policy_to_sim_idx
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    root_pose = torch.zeros((1, 7), device=device)
    root_vel = torch.zeros((1, 6), device=device)

    steps_per_frame = max(1, int(round((1.0 / args_cli.sample_hz) / env.sim.get_physics_dt())))

    out_file = out_writer = None
    if args_cli.csv_out:
        out_file = open(args_cli.csv_out, "w", newline="")
        out_writer = csv.writer(out_file)
        out_writer.writerow(
            ["frame", "t"]
            + ["base_x", "base_y", "base_z", "qw", "qx", "qy", "qz"]
            + [f"contact_{n}" for n in LEG_NAMES]
            + [f"q_{n}" for n in POLICY_JOINT_ORDER]
            + [f"dq_{n}" for n in POLICY_JOINT_ORDER]
            + [f"tau_{n}" for n in POLICY_JOINT_ORDER]
        )

    def play_once():
        prev_contact = np.zeros(4, dtype=bool)
        for frame, k in enumerate(keep):
            joint_pos[0, sim_idx] = torch.as_tensor(log["q"][k].reshape(12), dtype=torch.float32, device=device)
            joint_vel[0, sim_idx] = torch.as_tensor(log["dq"][k].reshape(12), dtype=torch.float32, device=device)
            root_pose[0, :3] = torch.as_tensor(pos[k] + world_offset, dtype=torch.float32, device=device)
            root_pose[0, 3:] = torch.as_tensor(quat[k], dtype=torch.float32, device=device)

            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_vel)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # Hold the drives at the logged angles too, so the actuators have nothing to
            # correct during the physics steps that follow.
            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()

            if markers is not None:
                touchdown = contact[k] & ~prev_contact
                for leg in np.flatnonzero(touchdown):
                    foot = pos[k] + feet_rel[k, leg] + world_offset
                    foot[2] = origin[2] + 0.002
                    footprint_pos.append(foot)
                if footprint_pos:
                    markers.visualize(
                        translations=torch.as_tensor(np.array(footprint_pos), dtype=torch.float32, device=device)
                    )
            prev_contact = contact[k].copy()

            if out_writer is not None:
                out_writer.writerow(
                    [frame, f"{log['t'][k]:.4f}"]
                    + [f"{v:.5f}" for v in (pos[k] + world_offset)]
                    + [f"{v:.6f}" for v in quat[k]]
                    + [int(v) for v in contact[k]]
                    + [f"{v:.5f}" for v in log["q"][k].reshape(12)]
                    + [f"{v:.5f}" for v in log["dq"][k].reshape(12)]
                    + [f"{v:.4f}" for v in log["tau"][k]]
                )

            for _ in range(steps_per_frame):
                env.sim.step(render=True)
            if not simulation_app.is_running():
                return False
        return True

    try:
        while play_once() and args_cli.loop:
            footprint_pos.clear()
    finally:
        if out_file is not None:
            out_file.close()
            report(f"Wrote {args_cli.csv_out}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
