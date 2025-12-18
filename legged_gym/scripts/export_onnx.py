import os
from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args

import torch

def export_onnx(args):
    # prepare environment
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    # load jit
    model_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies', 'policy_1.pt')
    print("Load model from:", model_path)
    jit_model = torch.jit.load(model_path)
    jit_model.eval()

    # export policy as onnx module (used to run it from C++)
    test_input_tensor = torch.randn(1,env_cfg.env.num_observations)  
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'onnx')
    os.makedirs(root_path, exist_ok=True)
    save_path = os.path.join(root_path, "onnx_1.onnx")
    torch.onnx.export(jit_model,
                    test_input_tensor,
                    save_path,
                    export_params=True,   
                    opset_version=11,     
                    do_constant_folding=True,  
                    input_names=['input'],    
                    output_names=['output'],  
                    )
    print('Exported policy as onnx script to: ', save_path)

if __name__ == '__main__':
    """
    python legged_gym/scripts/export_onnx.py --task=t4_stand
    """
    args = get_args()
    export_onnx(args)