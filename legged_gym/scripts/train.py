import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 必须在 import torch 之前！
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from record_config import record_config

# torch.cuda.set_device(0)  # 因为 CUDA_VISIBLE_DEVICES="1" 后，cuda:0 对应物理 GPU 1
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    record_config(log_root=log_dir, name=args.task)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
