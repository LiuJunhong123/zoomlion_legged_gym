from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .base.legged_robot_walk import LeggedRobotWalk
from .base.legged_robot_stand import LeggedRobotStand
from .t4.t4_stand import T4Stand
from .t4.t4_stand_config import T4StandCfg, T4StandCfgPPO
import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("t4_stand", T4Stand, T4StandCfg(), T4StandCfgPPO())
