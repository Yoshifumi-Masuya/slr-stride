from configs.go2_config import Go2RoughCfg, Go2RoughCfgPPO
import cv2
import os

from isaacgym import gymapi
from envs import LeggedRobot
from modules import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger
from configs import *
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR

from PIL import Image as im

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    #env_cfg.terrain.mesh_type = 'plane'
    env_cfg.domain_rand.push_robots = False
    #env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.control.use_filter = True
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_kpkd = False
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)

    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticSLR  = actor_critic_class(
                                            num_props=env.cfg.env.n_proprio,
                                            num_hist=env.cfg.env.history_len,
                                            num_actions=env.num_actions,
                                            **policy_cfg_dict)
    print(policy)
    #model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/rough_go2/Jul25_19-48-30_slr/model_109800.pt'), map_location='cuda:0')
    #model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/rough_go2/Sep19_01-32-30_slr/model_25600.pt'), map_location='cuda:0')
    model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/rough_go2/Sep20_22-42-28_slr/model_65000.pt'), map_location='cuda:0')
    policy.load_state_dict(model_dict['model_state_dict'])
    policy.eval()
    # policy = policy.to(env.device)
    policy = policy.to('cpu')
    policy.save_torch_jit_policy()
    policy = policy.to(env.device)

    # clear images under frames folder
    # frames_path = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    # delete_files_in_directory(frames_path)

    # set rgba camera sensor for debug and doudle check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    video_duration = 80
    num_frames = int(video_duration / env.dt)
    print(f'gathering {num_frames} frames')
    video = None

    #torch.sum(self.last_actions - self.actions, dim=1)
    # self.base_lin_vel[:, 2]
    #torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0

    # torque monitoring: running max |torque| per joint, and which step it happened at
    max_torque_per_joint = torch.zeros(env.num_dofs, device=env.device)
    max_torque_overall = 0.0
    max_torque_step = -1
    max_torque_joint_name = None

    for i in range(num_frames):
        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        env.commands[:,0] = 1
        env.commands[:,1] = 0
        env.commands[:,2] = 0
        env.commands[:,3] = 0
        env.commands[:,4] = STRIDE_LENGTH

        actions = policy.act_inference(obs)

        obs, privileged_obs, rewards,dones, infos = env.step(actions)
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)

        # torque monitoring
        cur_torques = env.torques[0].abs()
        max_torque_per_joint = torch.maximum(max_torque_per_joint, cur_torques)
        cur_max, cur_idx = cur_torques.max(dim=0)
        if cur_max.item() > max_torque_overall:
            max_torque_overall = cur_max.item()
            max_torque_step = i
            max_torque_joint_name = env.dof_names[cur_idx.item()]
        if i % 50 == 0:
            print(f"step {i:4d} | torque [N*m] max={cur_max.item():6.2f} mean={cur_torques.mean().item():6.2f}"
                  f" | limit={env.torque_limits[cur_idx.item()].item():.2f} (joint {env.dof_names[cur_idx.item()]})")

        if RECORD_FRAMES:
            img = env.gym.get_camera_image(env.sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR).reshape((512,512,4))[:,:,:3]
            if video is None:
                video = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(1 / env.dt), (img.shape[1],img.shape[0]))
            video.write(img)
            img_idx += 1
    print("action rate:",action_rate/num_frames)
    print("z vel:",z_vel/num_frames)
    print("xy_vel:",xy_vel/num_frames)
    print("feet air reward",feet_air_time/num_frames)
    print("-" * 60)
    print(f"max torque overall: {max_torque_overall:.2f} N*m, joint={max_torque_joint_name}, step={max_torque_step}")
    for name, val, limit in zip(env.dof_names, max_torque_per_joint.tolist(), env.torque_limits.tolist()):
        print(f"  {name:<16}{val:8.2f} N*m  (limit {limit:.2f}, {'SATURATED' if val >= limit - 1e-3 else ''})")
    if RECORD_FRAMES:
      video.release()


if __name__ == '__main__':
    task_registry.register("go2",LeggedRobot,Go2RoughCfg(),Go2RoughCfgPPO())
    RECORD_FRAMES = False
    STRIDE_LENGTH = 0.2  # commanded stride length [m], range 0.1-0.4 (configs/go2_config.py commands.ranges.stride_length)
    args = get_args()
    play(args)
