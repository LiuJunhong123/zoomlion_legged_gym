import os
from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path, update_cfg_from_args
from isaacgym.torch_utils import *

from rsl_rl.modules import ActorCritic
import torch

def export_policy(args):
    # prepare environment
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    actor_critic_class = eval(train_cfg_dict["runner"]["policy_class_name"])
    actor_critic: ActorCritic = actor_critic_class( env_cfg.env.num_observations,
                                                    env_cfg.env.num_privileged_obs,
                                                    env_cfg.env.num_actions,
                                                    **policy_cfg)

    # load policy
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported_data')
    model_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
    print("Load model from:", model_path)
    loaded_dict = torch.load(model_path)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])

    # export policy as jit module (used to run it from C++)
    save_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
    export_policy_as_jit(actor_critic, save_path)
    print('Exported policy as jit script to: ', save_path)

if __name__ == '__main__':
    """
    python legged_gym/scripts/export_policy.py --task=t4_stand
    """
    args = get_args()
    export_policy(args)