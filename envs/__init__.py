from .vec_env import VecEnv

# BaseTask/LeggedRobot depend on isaacgym; keep them optional so envs.vec_env
# (used by runner/algorithm/modules, and by envs_lab/ under Isaac Lab) stays
# importable without isaacgym installed.
try:
    from .base_task import BaseTask
    from .legged_robot import LeggedRobot
except ModuleNotFoundError:
    pass
