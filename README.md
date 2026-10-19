## Stride-Conditioned Quadruped Locomotion for the Unitree Go2

A fork of [**SLR: Learning Quadruped Locomotion without Privileged Information**](https://11chens.github.io/SLR/) (CoRL 2024) that extends the original policy with an explicit **stride-length command**, ports the environment to **Isaac Lab**, and carries the result onto a **physical Unitree Go2**.

**[📊 Slides](https://docs.google.com/presentation/d/1-2oEnoUHSeUYajZTcCplVR4od0JSFyb2/edit)** · **[📁 Videos and data](https://drive.google.com/drive/folders/14LWqdY1KWC_8lPcHA7mI6th4X4ARux2f)**

> **This is a modified version of the original work.** See [What this fork changes](#what-this-fork-changes) for the full list of modifications. The upstream code, method and figures are the work of Chen et al.; please cite their paper (see [Citation](#citation)) and respect the [non-commercial licence](#licence).

---

### Goals

The original SLR policy learns to track a velocity command without privileged information, estimating base linear velocity from a history of proprioception. This fork pursues three further goals:

1. **Track a velocity command reliably on flat and gently uneven ground**, including coming to a genuine stop when the command is zero.
2. **Make the stride length commandable at run time**, so the gait can be modulated independently of speed.
3. **Deploy the result on a real Unitree Go2** over ROS 2, and verify that what the simulator promises is what the hardware delivers.

---

### What this fork changes

#### 1. Stride length as a command

A fifth command channel carries a target stride length in metres, sampled from `[0.1, 0.4]`. It is appended to the proprioceptive observation **unscaled**, after the three velocity commands:

| Block | Dim | Scaling |
|---|---|---|
| `base_lin_vel` | 3 | × 2.0 (dropped by the actor — see note) |
| `base_ang_vel` | 3 | × 0.25 |
| projected gravity | 3 | — |
| `commands[0:3]` (vx, vy, ωz) | 3 | × (2, 2, 0.25) |
| **`commands[4]` (stride length)** | **1** | **raw, in metres** |
| `dof_pos − default` | 12 | — |
| `dof_vel` | 12 | × 0.05 |
| previous action | 12 | — |

Total `n_proprio = 49`, with a history of 10 frames. `ActorCriticSLR` drops the first three elements (`num_props − 3`), so the actor sees 46 dimensions and the encoder estimates base linear velocity from the history — the core idea of SLR, left intact.

Stride is **measured**, not assumed: each foot's touchdown is detected as a rising edge on the contact sensor, and the stride is the world-frame XY distance from that foot's previous touchdown. The reward is an exponential on the tracking error with its **own** `tracking_sigma_stride`, separate from the shared velocity `tracking_sigma`. The shared value (0.25) was tuned for errors on the order of 1 m/s and left the stride term nearly indifferent across a 0.1–0.4 m range; a worst-case 0.3 m error still scored ≈ 0.70.

A **curriculum** holds the stride reward at zero until velocity tracking reaches 80 % of a perfect rate. A robot that cannot hold a commanded velocity has no gait to modulate, so the stride term is only noise before that point.

#### 2. Honest stopping

Three changes make a zero velocity command mean *stop*:

- **Stride is forced to zero whenever the velocity command is zeroed**, and vice versa. Previously a standing robot could still collect stride reward by marching in place, which scored higher than standing still.
- A **`stand_still` penalty** charges for foot motion while the command is zero, so marching in place is distinguishable from standing.
- **`stand_still_prob = 0.08`** injects explicit zero commands during sampling. As the velocity curriculum widens its range, naturally-sampled near-zero commands become rare (12.5 % → 1.4 %); this keeps them at roughly 19 % early and 9 % late in training.

#### 3. Torque-limit termination

The hard clip stays at 100 % of the URDF effort rating. Separately, `soft_torque_limit = 0.8` **ends the episode** when any joint's pre-clip torque demand exceeds 80 % of its limit, with a one-time `termination` penalty. Ending on base or head contact is a separate, milder `fall` penalty, since early policies always fall and a large shared penalty drowns out every positive term.

#### 4. Isaac Lab port

The environment exists twice, driven by one shared config:

- `envs/legged_robot.py` — the original Isaac Gym `LeggedRobot`.
- `envs_lab/go2_direct_env.py` — an Isaac Lab `DirectRLEnv` (`Go2DirectEnv`).

`configs/go2_config.py` is the single source of truth for both. Rewards, curricula, termination conditions and command sampling are mirrored, so a checkpoint trained in one simulator plays back in the other.

Porting surfaced two silent inconsistencies that are now fixed: Isaac Lab ignored `soft_torque_limit`, and it had no contact-based termination at all. A PhysX contact-patch buffer overflow (`gpu_max_rigid_patch_count`) was also raised — the default silently dropped contacts, which corrupts both the stride measurement and the collision reward.

#### 5. Walking-focused terrain

The curriculum was retuned away from stair climbing. Stairs have a fixed tread width that fights the stride command, and the torque required to climb triggered `soft_torque_limit` termination disproportionately often.

```
[smooth_slope, rough_slope, stairs_up, stairs_down, discrete, flat]
 [0.3,          0.3,         0.0,       0.0,         0.1,      0.3]
```

---

### Repository layout

```
configs/go2_config.py      shared config: rewards, commands, curricula, domain randomisation
envs/                      Isaac Gym environment (original)
envs_lab/                  Isaac Lab DirectRLEnv port
modules/                   ActorCriticSLR and the state estimator
ros_utils/                 ROS 2 deployment nodes for the physical robot
train.py  / play.py        Isaac Gym entry points
train_lab.py / play_lab.py Isaac Lab entry points
replay_lowstate.py         kinematic replay of a real-robot LowState log
resources/go2/             URDF and meshes
```

---

### Getting started

#### Isaac Gym (original pipeline)

```bash
conda create -n slr python=3.8
conda activate slr
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Isaac Gym Preview 4: https://developer.nvidia.com/isaac-gym
cd isaacgym/python && pip install -e .

cd SLR-master-pbl && pip install .
python train.py
python play.py
```

It is worth reading through [RSL's Legged Gym](https://github.com/leggedrobotics/legged_gym) first to understand the Isaac-LeggedGym-RslRL framework.

#### Isaac Lab

Entry points are run through the Isaac Lab launcher, from the Isaac Lab installation directory:

```bash
conda activate isaaclab
cd ~/IsaacLab

# train
./isaaclab.sh -p /path/to/SLR-master-pbl/train_lab.py --headless --num_envs 512

# play a checkpoint on flat ground, side camera, 0.25 m stride command
./isaaclab.sh -p /path/to/SLR-master-pbl/play_lab.py \
    --num_envs 1 --steps 400 \
    --checkpoint logs/rough_go2_isaaclab/<run>/model_<n>.pt \
    --terrain_type flat --stride_length 0.25 --no_camera_follow
```

`play_lab.py` draws a straight reference line with one-metre ticks, drops a marker at every touchdown so stride is readable by eye, and can export per-step state with `--csv` (commands, base pose and velocity, measured stride, and every joint's angle, velocity and torque), printing velocity and stride tracking RMSE at the end. Terrain types: `flat`, `smooth_slope`, `rough_slope`, `stairs_up`, `stairs_down`, `discrete`.

---

### Real-robot deployment

The policy runs at 50 Hz on the robot over ROS 2. `ros_utils/` holds the two nodes that sit between the policy and the hardware:

| File | Role |
|---|---|
| `slr_policy_node.py` | Builds the observation from `/lowstate`, runs the exported TorchScript policy, publishes a joint-position target |
| `lowcmd_bridge.py` | Turns that target into a Unitree `LowCmd` at fixed `kp`/`kd`, with arming and timeout guards |
| `slr_go2.launch.py` | Brings both up together |

**Policy node.** Subscribes to `/lowstate`, `/cmd_vel` and `/slr_stride_length`; publishes `/slr_joint_target` and `/slr_raw_action`. It assembles the 46-dimensional observation the actor expects — IMU quaternion → projected gravity, gyroscope × 0.25 → base angular velocity, velocity command × (2, 2, 0.25), the stride command **unscaled in metres**, joint positions relative to the default pose, joint velocities × 0.05, and the previous raw action — keeps a 10-frame history for the encoder, and reproduces SLR's action pipeline exactly (low-pass `0.2·prev + 0.8·current`, `action_scale = 0.25`, hip scale reduction 0.5).

The exported `.jit` interface is checked at startup with a dummy forward pass (encoder 460 → 20, actor 66 → 12), so a checkpoint with a mismatched observation layout fails immediately rather than on the robot.

Note that the simulator does **not** feed the actor ground-truth orientation: it runs a Madgwick filter over simulated gyro and accelerometer readings, exactly so that the observation matches what the robot's own attitude filter produces. The node therefore uses `imu_state.quaternion` from `/lowstate` directly.

**LowCmd bridge.** Publishes at 200 Hz. Guards, in order: a target length and NaN check, a joint-limit clamp with margin, an `armed` parameter that must be set before any command leaves the node, and a `target_timeout_sec` after which the robot falls back to pure damping (`kp = 0`) rather than holding a stale target.

**Joint order.** Unitree's LowState orders the motors FR, FL, RR, RL × (hip, thigh, calf), which is already the policy's joint order — no reindexing is needed anywhere in the chain.

**Gains.** The bridge defaults to `kp = 40`, while training uses `kp = 45`. Set the bridge's `kp` parameter to 45 to match; see [Measuring target-angle tracking](#measuring-target-angle-tracking) for why this matters less than it looks.

> `ros_utils/` holds the node sources only — there is no `package.xml` or `setup.py` here, so it is not a buildable ROS 2 package as it stands. Drop the files into a `slr_ros2` package (the name the launch file expects) in your own workspace.

#### Replaying a real-robot log

`replay_lowstate.py` plays a recorded `LowState` CSV back in Isaac Sim. A LowState log holds measurements only — no `LowCmd`, and no base position or linear velocity — so it cannot drive the simulator in the usual sense, but it can be played back exactly:

- logged joint angles are written straight into the articulation, and the IMU quaternion becomes the base orientation, with gravity zeroed so nothing the simulator computes can fight the data;
- base position is reconstructed by **contact odometry** — while a foot is loaded its world position is fixed, so the base moved by minus that foot's displacement, with forward kinematics from the logged joint angles giving the foot position in the base frame. Height comes from putting the lowest loaded foot on the ground.

```bash
./isaaclab.sh -p /path/to/SLR-master-pbl/replay_lowstate.py \
    --csv_in lowstate.csv --start 12 --end 30 --footprints --loop
```

It also reports the stride each foot actually took, measured with the same rising-edge rule the training environment uses, so the numbers are directly comparable to the stride command.

---

### Measuring target-angle tracking

Neither the simulator nor the robot logs a tracking error directly, but both run the same PD law

```
tau = kp (q_des − q) − kd · dq      ⟹      q_des − q = (tau + kd · dq) / kp
```

so the error follows from the applied torque and joint velocity alone. On hardware this reconstruction was validated two ways: the autocorrelation of the reconstructed target peaks at exactly the 50 Hz policy period, and its noise floor while standing is 0.09°.

Simulation, flat ground, domain randomisation disabled, 800 control steps after a 2 s settling period:

| vx cmd | stride cmd | measured vx | measured stride | error RMS | error max |
|---|---|---|---|---|---|
| 0.4 | 0.1 | 0.53 | 0.126 | 6.0° | 18.7° |
| 0.4 | 0.4 | 0.69 | 0.334 | 7.6° | 55.4° |
| 0.6 | 0.1 | 0.71 | 0.130 | 6.3° | 23.6° |
| 0.6 | 0.3 | 0.83 | 0.325 | 7.1° | 23.2° |
| 0.6 | 0.4 | 0.90 | 0.379 | 7.7° | 53.7° |

Two findings worth knowing before tuning anything:

- **The calf joints realise roughly a third of their commanded amplitude at short strides** (amplitude ratio 0.21–0.40), rising to 0.37–0.77 at long strides. Much of the policy's output is absorbed by the PD controller rather than realised as motion. The effective action is what survives the PD, not what the network emits — worth remembering before changing `action_scale`.
- **The same is true on hardware**, at the same magnitude (6.7° RMS versus 6.5° in simulation, calf worst in both). This is a consequence of a soft position gain carrying the robot's weight, not a hardware fault, and it is not a sim-to-real gap.

---

### Licence

This codebase is licensed under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en), with inherited licenses from [Legged Gym](https://github.com/leggedrobotics/legged_gym) and [RSL RL](https://github.com/leggedrobotics/rsl_rl) by *ETH Zurich, Nikita Rudin* and *NVIDIA CORPORATION & AFFILIATES*. **Commercial use of the material is prohibited.** Modifications made in this fork are described in [What this fork changes](#what-this-fork-changes) and are released under the same terms.

---

### Citation

If this work has helped you, please cite the original SLR paper:

```bibtex
@inproceedings{
chen2024slr,
title={{SLR}: Learning Quadruped Locomotion without Privileged Information},
author={Shiyi Chen and Zeyu Wan and Shiyang Yan and Chun Zhang and Weiyi Zhang and Qiang Li and Debing Zhang and Fasih Ud Din Farrukh},
booktitle={8th Annual Conference on Robot Learning},
year={2024},
url={https://openreview.net/forum?id=RMkdcKK7jq}
}
```

---

### Acknowledgements

This project leverages components from several open-source repositories:

- [SLR](https://github.com/11chens/SLR-master) — the work this fork is built on
- [Legged Gym](https://github.com/leggedrobotics/legged_gym)
- [RSL RL](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) and [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [LocomotionWithNP3O](https://github.com/zeonsunlightyu/LocomotionWithNP3O)
- [ABS](https://github.com/LeCAR-Lab/ABS)

Thanks to the authors and contributors for their work.
