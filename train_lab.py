"""Isaac Lab training entry point for the Unitree Go2 (Phase 2: terrain curriculum).

Usage:
    conda activate isaaclab
    cd ~/my_workspace/IsaacLab_23
    ./isaaclab.sh -p /home/huro-lab/my_workspace/SLR-master-g/train_lab.py --headless --num_envs 64

Note: this always runs headless-safe: the terrain curriculum grid (default
10x20 = 200 tiles, --terrain_rows/--terrain_cols to override) is generated
regardless of --num_envs. Keep --headless for training; see play_lab.py's
docstring if you need a lighter-weight GUI check.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Go2 (Isaac Lab, Phase 2: terrain curriculum)")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override number of training iterations.")
parser.add_argument("--run_name", type=str, default="isaaclab_phase2", help="Run name suffix for logging.")
parser.add_argument(
    "--terrain_rows", type=int, default=None, help="Override terrain curriculum rows (levels). Default: Go2RoughCfg value (10)."
)
parser.add_argument(
    "--terrain_cols", type=int, default=None, help="Override terrain curriculum cols (types). Default: Go2RoughCfg value (20)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below needs the simulation app running before it can be imported ---
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_config import ROOT_DIR
from runner import OnPolicyRunner
from configs.go2_config import Go2RoughCfgPPO

from envs_lab import Go2DirectEnv, Go2DirectEnvCfg, IsaacLabVecEnvWrapper
from envs_lab.utils_lab import class_to_dict


def main():
    env_cfg = Go2DirectEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.terrain_rows is not None:
        env_cfg.terrain.terrain_generator.num_rows = args_cli.terrain_rows
    if args_cli.terrain_cols is not None:
        env_cfg.terrain.terrain_generator.num_cols = args_cli.terrain_cols
    env_cfg.terrain.max_init_terrain_level = min(
        env_cfg.terrain.max_init_terrain_level, env_cfg.terrain.terrain_generator.num_rows - 1
    )

    env = Go2DirectEnv(cfg=env_cfg, render_mode=None)
    wrapped_env = IsaacLabVecEnvWrapper(env)

    train_cfg = Go2RoughCfgPPO()
    if args_cli.max_iterations is not None:
        train_cfg.runner.max_iterations = args_cli.max_iterations
    train_cfg.runner.experiment_name = "rough_go2_isaaclab"
    train_cfg.runner.run_name = args_cli.run_name

    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    log_dir = os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name)

    runner = OnPolicyRunner(wrapped_env, class_to_dict(train_cfg), log_dir, args=args_cli, device="cuda:0")
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
