"""Adapts a Go2DirectEnv (Isaac Lab DirectRLEnv) to this repo's VecEnv interface
(envs/vec_env.py) so that algorithm/ppo.py, modules/ and runner/on_policy_runner.py
can be reused unchanged.
"""

import torch

from configs.go2_config import Go2RoughCfg


class IsaacLabVecEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.cfg = Go2RoughCfg()

        self.num_envs = env.num_envs
        self.num_obs = env.cfg.observation_space
        self.num_privileged_obs = None
        self.num_actions = env.cfg.action_space
        self.max_episode_length = env.max_episode_length
        self.device = env.device

        self.privileged_obs_buf = None
        self.obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        obs_dict, _ = self.env.reset()
        self.obs_buf = obs_dict["policy"]

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def step(self, actions: torch.Tensor):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        self.obs_buf = obs_dict["policy"]
        self.rew_buf = rew
        self.reset_buf = (terminated | truncated).to(dtype=torch.long)
        self.extras = extras
        self.extras["time_outs"] = truncated
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self, env_ids=None):
        obs_dict, extras = self.env.reset()
        self.obs_buf = obs_dict["policy"]
        self.extras = extras
        return self.obs_buf

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf
