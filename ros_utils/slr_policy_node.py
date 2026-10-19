#!/usr/bin/env python3
"""
SLR policy ROS2 node for Unitree Go2.

Inputs
------
/lowstate         : unitree_go/msg/LowState
/cmd_vel          : geometry_msgs/msg/Twist
/slr_stride_length: std_msgs/msg/Float32   (optional; falls back to the
                    stride_length parameter when no message arrives)

Outputs
-------
/slr_joint_target : std_msgs/msg/Float32MultiArray
    12 desired joint positions [rad], Unitree motor order:
    FR_hip, FR_thigh, FR_calf,
    FL_hip, FL_thigh, FL_calf,
    RR_hip, RR_thigh, RR_calf,
    RL_hip, RL_thigh, RL_calf

/slr_raw_action : std_msgs/msg/Float32MultiArray
    Raw 12-D policy action before the SLR low-pass/action scaling.

The observation and action processing reproduce the SLR Go2 implementation:
  proprio = [body_ang_vel*0.25,
             projected_gravity,
             [vx*2, vy*2, wz*0.25],
             stride_length_command,
             q-q_default,
             dq*0.05,
             previous_raw_action]
  history length = 10
  encoder: 460 -> 20
  actor:   66 -> 12
  low-pass: 0.2 * previous_raw_action + 0.8 * current_raw_action
  action scale = 0.25
  hip scale reduction = 0.5

Note on the proprio size: the training env's observation is 49-D and starts
with base_lin_vel*2.0, but ActorCriticSLR slices that off (`num_props-3,
no lin_vel`) - the encoder estimates it from history instead. So the policy
interface here is 46-D and base_lin_vel is never needed on hardware.

The stride_length command is passed unscaled, in meters, matching
Go2DirectEnv (`self.commands[:, 4:5]` goes into the observation raw).
Training samples it uniformly from [0.1, 0.4] m.
"""

from collections import deque
from pathlib import Path
import math
import time

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Float32MultiArray
from unitree_go.msg import LowState


# Policy interface sizes (see the module docstring).
N_PROPRIO = 46
N_HISTORY = 10
N_LATENT = 20


# Unitree Go2 low-level motor order used by unitree_ros2:
# FR, FL, RR, RL, with hip/thigh/calf for each leg.
Q_DEFAULT = np.array([
    -0.1, 0.8, -1.5,   # FR
     0.1, 0.8, -1.5,   # FL
    -0.1, 1.0, -1.5,   # RR
     0.1, 1.0, -1.5,   # RL
], dtype=np.float32)

# Hard limits from the SLR Go2 URDF, in Unitree order.
Q_MIN = np.array([
    -1.0472, -1.5708, -2.7227,   # FR
    -1.0472, -1.5708, -2.7227,   # FL
    -1.0472, -0.5236, -2.7227,   # RR
    -1.0472, -0.5236, -2.7227,   # RL
], dtype=np.float32)

Q_MAX = np.array([
     1.0472, 3.4907, -0.83776,   # FR
     1.0472, 3.4907, -0.83776,   # FL
     1.0472, 4.5379, -0.83776,   # RR
     1.0472, 4.5379, -0.83776,   # RL
], dtype=np.float32)


def projected_gravity_from_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Return gravity direction [0,0,-1] expressed in the body frame.

    Unitree LowState IMU quaternion is handled here as [w, x, y, z].
    For identity attitude the result is [0, 0, -1].
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError("Invalid IMU quaternion")
    w, x, y, z = q / n

    # Rotation matrix body -> world.
    r00 = 1.0 - 2.0 * (y*y + z*z)
    r01 = 2.0 * (x*y - z*w)
    r02 = 2.0 * (x*z + y*w)

    r10 = 2.0 * (x*y + z*w)
    r11 = 1.0 - 2.0 * (x*x + z*z)
    r12 = 2.0 * (y*z - x*w)

    r20 = 2.0 * (x*z - y*w)
    r21 = 2.0 * (y*z + x*w)
    r22 = 1.0 - 2.0 * (x*x + y*y)

    # R^T @ [0, 0, -1] = - third row of R.
    return np.array([-r20, -r21, -r22], dtype=np.float32)


class SLRPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("slr_policy_node")

        # Paths / topics
        self.declare_parameter("encoder_path", "")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("stride_length_topic", "/slr_stride_length")
        self.declare_parameter("joint_target_topic", "/slr_joint_target")
        self.declare_parameter("raw_action_topic", "/slr_raw_action")

        # Timing
        self.declare_parameter("policy_hz", 50.0)
        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("lowstate_timeout_sec", 0.1)

        # Command range used during SLR Go2 training
        self.declare_parameter("max_vx", 0.5)
        self.declare_parameter("max_vy", 0.5)
        self.declare_parameter("max_wz", 1.0)

        # Target stride length [m]: same-foot touchdown-to-touchdown distance.
        # Training samples this from [min_stride_length, max_stride_length];
        # values outside that range were never seen by the policy.
        self.declare_parameter("stride_length", 0.25)
        self.declare_parameter("min_stride_length", 0.1)
        self.declare_parameter("max_stride_length", 0.4)

        # Added deployment safety guard.
        # SLR itself uses clip_actions=100.0. Start smaller on hardware.
        self.declare_parameter("action_clip", 4.0)

        # Keep a small margin from the URDF hard joint limits.
        self.declare_parameter("joint_limit_margin", 0.03)

        # Torch
        self.declare_parameter("device", "cpu")
        self.declare_parameter("torch_num_threads", 1)

        encoder_path = str(self.get_parameter("encoder_path").value)
        policy_path = str(self.get_parameter("policy_path").value)
        self.device = str(self.get_parameter("device").value)

        if not encoder_path or not policy_path:
            raise RuntimeError(
                "Set both encoder_path and policy_path ROS parameters."
            )
        if not Path(encoder_path).is_file():
            raise FileNotFoundError(f"encoder.jit not found: {encoder_path}")
        if not Path(policy_path).is_file():
            raise FileNotFoundError(f"policy.jit not found: {policy_path}")

        torch.set_num_threads(int(self.get_parameter("torch_num_threads").value))
        self.encoder = torch.jit.load(encoder_path, map_location=self.device)
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.encoder.eval()
        self.policy.eval()

        # Verify the exported SLR interface before receiving robot data.
        # A mismatch here usually means the .jit files were exported from a
        # checkpoint with a different observation layout.
        with torch.inference_mode():
            enc_in = N_PROPRIO * N_HISTORY
            z = self.encoder(torch.zeros((1, enc_in), dtype=torch.float32, device=self.device))
            if tuple(z.shape) != (1, N_LATENT):
                raise RuntimeError(f"Unexpected encoder output shape: {tuple(z.shape)}")
            a = self.policy(
                torch.zeros((1, N_LATENT + N_PROPRIO), dtype=torch.float32, device=self.device)
            )
            if tuple(a.shape) != (1, 12):
                raise RuntimeError(f"Unexpected policy output shape: {tuple(a.shape)}")

        self.history = deque(maxlen=N_HISTORY)
        self.prev_raw_action = np.zeros(12, dtype=np.float32)

        self.latest_q = None
        self.latest_dq = None
        self.latest_gyro = None
        self.latest_quat = None
        self.last_lowstate_time = None

        self.cmd = np.zeros(3, dtype=np.float32)
        self.last_cmd_time = None

        # Unlike /cmd_vel, the stride command has no timeout: it is a gait
        # style target, so it keeps its last value (or the parameter default)
        # rather than falling back to zero, which the policy never saw.
        self.stride_length = self.clamp_stride(
            float(self.get_parameter("stride_length").value)
        )

        # Low-state can be high-rate DDS data, so keep only the newest sample.
        lowstate_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        lowstate_topic = str(self.get_parameter("lowstate_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        stride_length_topic = str(self.get_parameter("stride_length_topic").value)
        joint_target_topic = str(self.get_parameter("joint_target_topic").value)
        raw_action_topic = str(self.get_parameter("raw_action_topic").value)

        self.lowstate_sub = self.create_subscription(
            LowState, lowstate_topic, self.lowstate_callback, lowstate_qos
        )
        self.cmd_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_callback, 10
        )
        self.stride_sub = self.create_subscription(
            Float32, stride_length_topic, self.stride_callback, 10
        )
        self.target_pub = self.create_publisher(
            Float32MultiArray, joint_target_topic, 10
        )
        self.action_pub = self.create_publisher(
            Float32MultiArray, raw_action_topic, 10
        )

        policy_hz = float(self.get_parameter("policy_hz").value)
        if policy_hz <= 0.0:
            raise ValueError("policy_hz must be positive")
        self.timer = self.create_timer(1.0 / policy_hz, self.policy_step)

        self._warned_no_state = False

        self.get_logger().info(
            f"SLR policy loaded. proprio={N_PROPRIO}-D, policy rate={policy_hz:.1f} Hz, "
            f"device={self.device}, motor order=FR/FL/RR/RL, "
            f"stride_length={self.stride_length:.3f} m."
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def clamp_stride(self, value: float) -> float:
        lo = float(self.get_parameter("min_stride_length").value)
        hi = float(self.get_parameter("max_stride_length").value)
        if not math.isfinite(value):
            return lo
        return float(np.clip(value, lo, hi))

    def stride_callback(self, msg: Float32) -> None:
        self.stride_length = self.clamp_stride(float(msg.data))

    def lowstate_callback(self, msg: LowState) -> None:
        # First 12 motor entries are the 12 leg joints on Go2.
        q = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
        dq = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        quat = np.array(msg.imu_state.quaternion, dtype=np.float32)

        if (
            q.shape != (12,)
            or dq.shape != (12,)
            or gyro.shape != (3,)
            or quat.shape != (4,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(dq))
            or not np.all(np.isfinite(gyro))
            or not np.all(np.isfinite(quat))
        ):
            self.get_logger().error("Rejected invalid /lowstate sample.")
            return

        self.latest_q = q
        self.latest_dq = dq
        self.latest_gyro = gyro
        self.latest_quat = quat
        self.last_lowstate_time = self.now_sec()

    def cmd_callback(self, msg: Twist) -> None:
        max_vx = float(self.get_parameter("max_vx").value)
        max_vy = float(self.get_parameter("max_vy").value)
        max_wz = float(self.get_parameter("max_wz").value)

        self.cmd[0] = np.clip(float(msg.linear.x), -max_vx, max_vx)
        self.cmd[1] = np.clip(float(msg.linear.y), -max_vy, max_vy)
        self.cmd[2] = np.clip(float(msg.angular.z), -max_wz, max_wz)
        self.last_cmd_time = self.now_sec()

    def build_proprio(self) -> np.ndarray:
        gravity = projected_gravity_from_wxyz(self.latest_quat)

        # Match SLR observation scales.
        ang_vel_obs = self.latest_gyro * 0.25
        cmd_obs = self.cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)
        q_obs = self.latest_q - Q_DEFAULT
        dq_obs = self.latest_dq * 0.05

        # Unscaled, in meters - Go2DirectEnv feeds commands[:, 4:5] in raw.
        stride_obs = np.array([self.stride_length], dtype=np.float32)

        proprio = np.concatenate(
            [
                ang_vel_obs,           # 3
                gravity,               # 3
                cmd_obs,               # 3
                stride_obs,            # 1
                q_obs,                 # 12
                dq_obs,                # 12
                self.prev_raw_action,  # 12
            ]
        ).astype(np.float32)

        if proprio.shape != (N_PROPRIO,):
            raise RuntimeError(f"proprio shape must be {N_PROPRIO}, got {proprio.shape}")
        return proprio

    def policy_step(self) -> None:
        now = self.now_sec()
        lowstate_timeout = float(self.get_parameter("lowstate_timeout_sec").value)

        if self.last_lowstate_time is None or (now - self.last_lowstate_time) > lowstate_timeout:
            if not self._warned_no_state:
                self.get_logger().warn(
                    "No fresh /lowstate. Policy output is not being published."
                )
                self._warned_no_state = True
            return
        self._warned_no_state = False

        # If /cmd_vel disappears, command zero velocity instead of holding
        # the last locomotion command.
        cmd_timeout = float(self.get_parameter("cmd_timeout_sec").value)
        if self.last_cmd_time is None or (now - self.last_cmd_time) > cmd_timeout:
            self.cmd[:] = 0.0

        try:
            proprio = self.build_proprio()
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return

        # SLR history semantics:
        # after startup, encoder input is the latest 10 proprio frames,
        # including the current frame.
        if len(self.history) == 0:
            for _ in range(N_HISTORY):
                self.history.append(proprio.copy())
        else:
            self.history.append(proprio.copy())

        hist = np.stack(self.history, axis=0).reshape(1, N_HISTORY * N_PROPRIO)
        prop = proprio.reshape(1, N_PROPRIO)

        hist_t = torch.from_numpy(hist).to(self.device)
        prop_t = torch.from_numpy(prop).to(self.device)

        with torch.inference_mode():
            z = self.encoder(hist_t)
            actor_input = torch.cat((z, prop_t), dim=1)
            raw_action = self.policy(actor_input).detach().cpu().numpy()[0].astype(np.float32)

        if not np.all(np.isfinite(raw_action)):
            self.get_logger().error("Policy produced NaN/Inf; target suppressed.")
            return

        # Added hardware safety guard. Set action_clip=100.0 later if you
        # deliberately want the original SLR clipping range.
        action_clip = float(self.get_parameter("action_clip").value)
        raw_action = np.clip(raw_action, -action_clip, action_clip)

        # Match SLR low-pass action filter.
        filtered_action = 0.2 * self.prev_raw_action + 0.8 * raw_action

        # Match SLR action scaling.
        delta_q = 0.25 * filtered_action
        delta_q[[0, 3, 6, 9]] *= 0.5  # hip scale reduction

        q_des = Q_DEFAULT + delta_q

        # Last deployment guard: clamp to URDF hard limits with margin.
        margin = max(0.0, float(self.get_parameter("joint_limit_margin").value))
        q_lo = Q_MIN + margin
        q_hi = Q_MAX - margin
        q_des = np.clip(q_des, q_lo, q_hi).astype(np.float32)

        target_msg = Float32MultiArray()
        target_msg.data = [float(x) for x in q_des]
        self.target_pub.publish(target_msg)

        action_msg = Float32MultiArray()
        action_msg.data = [float(x) for x in raw_action]
        self.action_pub.publish(action_msg)

        # Important: observation uses the previous RAW policy action,
        # not q_des and not the filtered action.
        self.prev_raw_action = raw_action.copy()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SLRPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
