#!/usr/bin/env python3
"""
Unitree Go2 LowCmd bridge in pure Python/ROS2.

Input
-----
/slr_joint_target : std_msgs/msg/Float32MultiArray
    12 desired joint positions in Unitree order:
    FR, FL, RR, RL (hip/thigh/calf)

Output
------
/lowcmd : unitree_go/msg/LowCmd

Safety behavior
---------------
- Default parameter armed:=false -> publishes NO LowCmd.
- When armed and a fresh 12-D target exists -> position control.
- When armed but target times out -> damping command.
- q_des is clamped to SLR Go2 URDF hard joint limits.
- CRC is computed in Python using the Go2 LowCmd packed layout.
"""

import struct
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowCmd


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Go2 motor order used by Unitree low-level control:
# 0..2 FR, 3..5 FL, 6..8 RR, 9..11 RL
Q_MIN = np.array([
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227,
], dtype=np.float32)

Q_MAX = np.array([
     1.0472, 3.4907, -0.83776,
     1.0472, 3.4907, -0.83776,
     1.0472, 4.5379, -0.83776,
     1.0472, 4.5379, -0.83776,
], dtype=np.float32)


def _u8(x):
    return int(x) & 0xFF


def _u16(x):
    return int(x) & 0xFFFF


def _u32(x):
    return int(x) & 0xFFFFFFFF


def pack_lowcmd(msg: LowCmd) -> bytes:
    """
    Pack unitree_go/msg/LowCmd using the 4-byte-aligned Go2 layout.

    Total packed size is 812 bytes. CRC occupies the final uint32 and
    is excluded by compute_lowcmd_crc().
    """
    fmt = "<4B4IH2x" + ("B3x5f3I" * 20) + "4B" + "55Bx2I"
    values = []

    values.extend([_u8(msg.head[0]), _u8(msg.head[1])])
    values.append(_u8(msg.level_flag))
    values.append(_u8(msg.frame_reserve))

    values.extend([_u32(x) for x in msg.sn])
    values.extend([_u32(x) for x in msg.version])
    values.append(_u16(msg.bandwidth))

    for i in range(20):
        m = msg.motor_cmd[i]
        values.append(_u8(m.mode))
        values.extend([
            float(m.q),
            float(m.dq),
            float(m.tau),
            float(m.kp),
            float(m.kd),
        ])
        values.extend([_u32(x) for x in m.reserve])

    values.append(_u8(msg.bms_cmd.off))
    values.extend([_u8(x) for x in msg.bms_cmd.reserve])

    values.extend([_u8(x) for x in msg.wireless_remote])
    values.extend([_u8(x) for x in msg.led])
    values.extend([_u8(x) for x in msg.fan])
    values.append(_u8(msg.gpio))

    values.append(_u32(msg.reserve))
    values.append(_u32(msg.crc))

    packed = struct.pack(fmt, *values)
    if len(packed) != 812:
        raise RuntimeError(f"Unexpected LowCmd packed size: {len(packed)}")
    return packed


def crc32_unitree_words(words) -> int:
    """
    Unitree CRC32 over uint32 words:
    init=0xFFFFFFFF, polynomial=0x04C11DB7, MSB first.
    """
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7

    for current in words:
        current = int(current) & 0xFFFFFFFF
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF

            if current & bit:
                crc ^= polynomial
            bit >>= 1

    return crc & 0xFFFFFFFF


def compute_lowcmd_crc(msg: LowCmd) -> int:
    packed = pack_lowcmd(msg)
    # Official Go2 CRC excludes the last 4-byte CRC field itself.
    body = packed[:-4]
    word_count = len(body) // 4
    words = struct.unpack("<" + ("I" * word_count), body)
    return crc32_unitree_words(words)


class LowCmdBridge(Node):
    def __init__(self) -> None:
        super().__init__("lowcmd_bridge")

        self.declare_parameter("joint_target_topic", "/slr_joint_target")
        self.declare_parameter("lowcmd_topic", "/lowcmd")
        self.declare_parameter("publish_hz", 200.0)

        # Safety: stays completely silent until explicitly armed.
        self.declare_parameter("armed", False)
        self.declare_parameter("target_timeout_sec", 0.10)

        self.declare_parameter("kp", 40.0)
        self.declare_parameter("kd", 1.0)
        self.declare_parameter("damping_kd", 1.0)
        self.declare_parameter("joint_limit_margin", 0.03)

        target_topic = str(self.get_parameter("joint_target_topic").value)
        lowcmd_topic = str(self.get_parameter("lowcmd_topic").value)

        self.target_sub = self.create_subscription(
            Float32MultiArray,
            target_topic,
            self.target_callback,
            10,
        )
        self.lowcmd_pub = self.create_publisher(LowCmd, lowcmd_topic, 10)

        self.latest_target = None
        self.last_target_time = None
        self.last_mode = None

        publish_hz = float(self.get_parameter("publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        self.timer = self.create_timer(1.0 / publish_hz, self.publish_step)

        # Allocate the message once and update it in place.
        self.cmd = LowCmd()
        self.init_lowcmd()

        self.get_logger().warn(
            "LowCmd bridge started DISARMED. "
            "Use --ros-args -p armed:=true only after verifying policy targets."
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def init_lowcmd(self) -> None:
        # Match Unitree's low-level example: all 20 motor slots use FOC mode.
        for i in range(20):
            m = self.cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(POS_STOP_F)
            m.dq = float(VEL_STOP_F)
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def target_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 12:
            self.get_logger().error(
                f"Rejected joint target length {len(msg.data)}; expected 12."
            )
            return

        q = np.asarray(msg.data, dtype=np.float32)
        if not np.all(np.isfinite(q)):
            self.get_logger().error("Rejected NaN/Inf joint target.")
            return

        margin = max(0.0, float(self.get_parameter("joint_limit_margin").value))
        q = np.clip(q, Q_MIN + margin, Q_MAX - margin)

        self.latest_target = q
        self.last_target_time = self.now_sec()

    def set_position_command(self, q_des: np.ndarray) -> None:
        kp = float(self.get_parameter("kp").value)
        kd = float(self.get_parameter("kd").value)

        for i in range(12):
            m = self.cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(q_des[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = kp
            m.kd = kd

        # Motors 12..19 are unused for standard Go2 legs.
        for i in range(12, 20):
            m = self.cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(POS_STOP_F)
            m.dq = float(VEL_STOP_F)
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def set_damping_command(self) -> None:
        damping_kd = float(self.get_parameter("damping_kd").value)

        # q disabled, desired velocity 0, viscous damping active.
        for i in range(12):
            m = self.cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(POS_STOP_F)
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = damping_kd

        for i in range(12, 20):
            m = self.cmd.motor_cmd[i]
            m.mode = 0x01
            m.q = float(POS_STOP_F)
            m.dq = float(VEL_STOP_F)
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def publish_step(self) -> None:
        armed = bool(self.get_parameter("armed").value)

        # Important: when disarmed, do not publish /lowcmd at all.
        # This avoids taking low-level control unintentionally.
        if not armed:
            if self.last_mode != "disarmed":
                self.get_logger().info("DISARMED: /lowcmd is not being published.")
                self.last_mode = "disarmed"
            return

        now = self.now_sec()
        timeout = float(self.get_parameter("target_timeout_sec").value)

        fresh = (
            self.latest_target is not None
            and self.last_target_time is not None
            and (now - self.last_target_time) <= timeout
        )

        if fresh:
            self.set_position_command(self.latest_target)
            mode = "position"
        else:
            self.set_damping_command()
            mode = "damping"

        if mode != self.last_mode:
            if mode == "position":
                self.get_logger().info("ARMED: fresh SLR target -> position control.")
            else:
                self.get_logger().warn(
                    "ARMED but SLR target timed out -> damping mode."
                )
            self.last_mode = mode

        self.cmd.crc = 0
        self.cmd.crc = compute_lowcmd_crc(self.cmd)
        self.lowcmd_pub.publish(self.cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LowCmdBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
