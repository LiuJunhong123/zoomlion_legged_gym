import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  Logger
import torch
import pygame
from threading import Thread
import matplotlib.pyplot as plt
import select
import sys
import time
import csv
import os
from pynput import keyboard

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 0.0, 0.0, 0.0
x_vel_min, y_vel_min, yaw_vel_min = 0.0, 0.0, 0.0

joystick_use = True
joystick_opened = False

if joystick_use:

    pygame.init()

    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"cannot open joystick device:{e}")

    exit_flag = False

    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        
        def clamp(value, min_val, max_val):
            return max(min_val, min(value, max_val))
        
        while not exit_flag:
            pygame.event.get()

            x_vel_cmd = clamp(-joystick.get_axis(1) * x_vel_max, x_vel_min, x_vel_max)
            y_vel_cmd = clamp(-joystick.get_axis(0) * y_vel_max, y_vel_min, y_vel_max)
            yaw_vel_cmd = clamp(-joystick.get_axis(3) * yaw_vel_max, yaw_vel_min, yaw_vel_max)

            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()
class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0
    
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = q[:3]
    foot_positions = []
    foot_forces = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if 'foot_link' in body_name: 
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces.append(data.cfrc_ext[i][2].copy().astype(np.double)) 
    
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp + (target_dq - dq)* kd
    return torque_out

def setup_keyboard_listener():
    """
    Set up keyboard event listener for user control input.
    """
    
    def on_press(key):
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
        vel_x_resolution = 0.1
        vel_y_resolution = 0.1
        vel_yaw_resolution = 0.1
        try:
            if key.char == "8":
                x_vel_cmd = min(x_vel_cmd + vel_x_resolution, x_vel_max)
            elif key.char == "2":
                x_vel_cmd = max(x_vel_cmd - vel_x_resolution, x_vel_min)
            elif key.char == "4":
                y_vel_cmd = min(y_vel_cmd + vel_y_resolution, y_vel_max)
            elif key.char == "6":
                y_vel_cmd = max(y_vel_cmd - vel_y_resolution, y_vel_min)
            elif key.char == "7":  
                yaw_vel_cmd = min(yaw_vel_cmd + vel_yaw_resolution, yaw_vel_max)
            elif key.char == "9":  
                yaw_vel_cmd = max(yaw_vel_cmd - vel_yaw_resolution, yaw_vel_min)
            elif key.char == "0":
                x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
            elif key.char == "3":
                x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.8, 0.0, 0.0
            elif key.char == "1":
                x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.2, 0.0, 0.0
        except AttributeError:
            pass

    return keyboard.Listener(on_press=on_press)

def run_mujoco(policy, cfg, env_cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    model.opt.timestep = cfg.sim_config.dt
    
    data = mujoco.MjData(model)
    num_actuated_joints = env_cfg.env.num_actions  # This should match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    data.qpos[:3] = cfg.robot_config.init_pos

    mujoco.mj_step(model, data)
    
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer._render_every_frame = False
    # viewer._paused = True
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] =np.array([0.0,-0.25,0.824])

    target_q = np.zeros((env_cfg.env.num_actions), dtype=np.double)
   
    action = np.zeros((env_cfg.env.num_actions), dtype=np.double)

    mujoco_to_isaac_idx = [
        0,  # J_arm_l_01
        1,  # J_arm_l_02
        2,  # J_arm_l_03
        3,  # J_arm_l_04
        4,  # J_arm_l_05
        5,  # J_arm_r_01
        6,  # J_arm_r_02
        7,  # J_arm_r_03
        8,  # J_arm_r_04
        9,  # J_arm_r_05
        10,  # J_waist_yaw
        11,  # J_hip_l_pitch
        12,  # J_hip_l_roll
        13,  # J_hip_l_yaw
        14,  # J_knee_l_pitch
        15,  # J_ankle_l_pitch
        16,  # J_ankle_l_roll
        17,  # J_hip_r_pitch
        18,  # J_hip_r_roll
        19,  # J_hip_r_yaw
        20,  # J_knee_r_pitch
        21,  # J_ankle_r_pitch
        22,  # J_ankle_r_roll
    ]
    isaac_to_mujoco_idx = [
        0,  # J_arm_l_01
        1,  # J_arm_l_02
        2,  # J_arm_l_03
        3,  # J_arm_l_04
        4,  # J_arm_l_05
        5,  # J_arm_r_01
        6,  # J_arm_r_02
        7,  # J_arm_r_03
        8,  # J_arm_r_04
        9,  # J_arm_r_05
        10,  # J_waist_yaw
        11,  # J_hip_l_pitch
        12,  # J_hip_l_roll
        13,  # J_hip_l_yaw
        14,  # J_knee_l_pitch
        15,  # J_ankle_l_pitch
        16,  # J_ankle_l_roll
        17,  # J_hip_r_pitch
        18,  # J_hip_r_roll
        19,  # J_hip_r_yaw
        20,  # J_knee_r_pitch
        21,  # J_ankle_r_pitch
        22,  # J_ankle_r_roll
    ]

    hist_obs = deque()
    for _ in range(env_cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, env_cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 1
    logger = Logger(cfg.sim_config.dt)
    
    stop_state_log = 4000

    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    
    base_positions = []
    joint_positions = []
    joint_vel = []
    joint_tau = []
    target_positions = []
    base_ang_vel = []
    feet_z = []
    feet_force_z = []
    timesteps = []
    actions = []

    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd

    listener = setup_keyboard_listener()
    listener.start()

    try:
        # print(f"Set command (x, y, yaw): ")
        for i in range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)):
            start_time = time.time()

            # Obtain an observation
            q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
            q = q[-env_cfg.env.num_actions:]
            dq = dq[-env_cfg.env.num_actions:]
            
            base_positions.append(base_pos.copy())
            joint_positions.append(q.copy())
            joint_vel.append(dq.copy())
            base_ang_vel.append(omega.copy())
            feet_z.append(foot_positions)
            feet_force_z.append(foot_forces)
            timesteps.append(i * cfg.sim_config.dt)
            
            base_z = base_pos[2]
            foot_z = foot_positions
            foot_force_z = foot_forces

            # 1000hz -> 100hz
            if count_lowlevel % cfg.sim_config.decimation == 0:

                if env_cfg.commands.use_gait_phase and env_cfg.commands.stand_gait_phase:
                    vel_norm = np.sqrt(x_vel_cmd**2 + y_vel_cmd**2 + yaw_vel_cmd**2)
                    if vel_norm <= env_cfg.commands.stand_commands_threshold:
                        count_lowlevel = 0

                obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                cycle_time = env_cfg.rewards.cycle_time
                vel_ids = int(x_vel_cmd / env_cfg.rewards.track_vel_threshold)
                if vel_ids >= len(cycle_time):
                    vel_ids = len(cycle_time) - 1
                
                obs[0, 0] = x_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * env_cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * env_cfg.normalization.obs_scales.ang_vel
                obs[0, 3:3+env_cfg.env.num_actions] = (q - cfg.robot_config.default_dof_pos)[mujoco_to_isaac_idx] * env_cfg.normalization.obs_scales.dof_pos
                obs[0, 3+env_cfg.env.num_actions:3+2*env_cfg.env.num_actions] = dq[mujoco_to_isaac_idx] * env_cfg.normalization.obs_scales.dof_vel
                obs[0, 3+2*env_cfg.env.num_actions:3+3*env_cfg.env.num_actions] = action * env_cfg.normalization.obs_scales.actions
                obs[0, 3+3*env_cfg.env.num_actions:3+3*env_cfg.env.num_actions+3] = omega * env_cfg.normalization.obs_scales.ang_vel
                # obs[0, 75:78] = eu_ang * cfg.normalization.obs_scales.quat
                obs[0, 3+3*env_cfg.env.num_actions+3:3+3*env_cfg.env.num_actions+6] = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])) * env_cfg.normalization.obs_scales.gravity
                # obs[0, 75:78] = gvec * cfg.normalization.obs_scales.gravity
                if env_cfg.commands.use_gait_phase:
                    obs[0, 3+3*env_cfg.env.num_actions+6:3+3*env_cfg.env.num_actions+7] = math.sin(2*math.pi*count_lowlevel*cfg.sim_config.dt/cycle_time[vel_ids])
                    obs[0, 3+3*env_cfg.env.num_actions+7:3+3*env_cfg.env.num_actions+8] = math.cos(2*math.pi*count_lowlevel*cfg.sim_config.dt/cycle_time[vel_ids])

                obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
                for i in range(env_cfg.env.frame_stack):
                    policy_input[0, i * env_cfg.env.num_single_obs : (i + 1) * env_cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)

                target_q = action * env_cfg.control.action_scale
                action_save = target_q[isaac_to_mujoco_idx] + cfg.robot_config.default_dof_pos
                actions.append(action_save)

            target_dq = np.zeros((env_cfg.env.num_actions), dtype=np.double)

            target_positions.append(target_q[isaac_to_mujoco_idx] + cfg.robot_config.default_dof_pos)

            # Generate PD control
            tau = pd_control(target_q[isaac_to_mujoco_idx], q, cfg.robot_config.kps,
                             target_dq[isaac_to_mujoco_idx], dq, cfg.robot_config.kds, cfg)  # Calc torques
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            joint_tau.append(tau.copy())
            
            data.ctrl = tau
            applied_tau = data.actuator_force

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.qpos.astype(np.float32)[0:3]
            viewer.render()
            count_lowlevel += 1
            idx = 4
            dof_pos_target = target_q + cfg.robot_config.default_dof_pos
            if i < stop_state_log:
                logger.log_states(
                    {   
                        'base_height' : base_z,
                        'foot_z_l' : foot_z[0],
                        'foot_z_r' : foot_z[1],
                        'foot_forcez_l' : foot_force_z[0],
                        'foot_forcez_r' : foot_force_z[1],
                        'base_vel_x': v[0],
                        'command_x': 1,
                        'base_vel_y': v[1],
                        'command_y': y_vel_cmd,
                        'base_vel_z': v[2],
                        'base_vel_yaw': omega[2],
                        'command_yaw': yaw_vel_cmd,
                        'dof_pos_target': dof_pos_target[idx] ,
                        'dof_pos': q[idx],
                        'dof_vel': dq[idx],
                        'dof_torque': applied_tau[idx],
                        'cmd_dof_torque': tau[idx],
                        'dof_pos_target[0]': dof_pos_target[0].item(),
                        'dof_pos_target[1]': dof_pos_target[1].item(),
                        'dof_pos_target[2]': dof_pos_target[2].item(),
                        'dof_pos_target[3]': dof_pos_target[3].item(),
                        'dof_pos_target[4]': dof_pos_target[4].item(),
                        'dof_pos_target[5]': dof_pos_target[5].item(),
                        'dof_pos_target[6]': dof_pos_target[6].item(),
                        'dof_pos_target[7]': dof_pos_target[7].item(),
                        'dof_pos_target[8]': dof_pos_target[8].item(),
                        'dof_pos_target[9]': dof_pos_target[9].item(),
                        'dof_pos_target[10]': dof_pos_target[10].item(),
                        'dof_pos_target[11]': dof_pos_target[11].item(),
                        'dof_pos':    q[0].item(),
                        'dof_pos[0]': q[0].item(),
                        'dof_pos[1]': q[1].item(),
                        'dof_pos[2]': q[2].item(),
                        'dof_pos[3]': q[3].item(),
                        'dof_pos[4]': q[4].item(),
                        'dof_pos[5]': q[5].item(),
                        'dof_pos[6]': q[6].item(),
                        'dof_pos[7]': q[7].item(),
                        'dof_pos[8]': q[8].item(),
                        'dof_pos[9]': q[9].item(),
                        'dof_pos[10]': q[10].item(),
                        'dof_pos[11]': q[11].item(),
                        'dof_torque': applied_tau[0].item(),
                        'dof_torque[0]': applied_tau[0].item(),
                        'dof_torque[1]': applied_tau[1].item(),
                        'dof_torque[2]': applied_tau[2].item(),
                        'dof_torque[3]': applied_tau[3].item(),
                        'dof_torque[4]': applied_tau[4].item(),
                        'dof_torque[5]': applied_tau[5].item(),
                        'dof_torque[6]': applied_tau[6].item(),
                        'dof_torque[7]': applied_tau[7].item(),
                        'dof_torque[8]': applied_tau[8].item(),
                        'dof_torque[9]': applied_tau[9].item(),
                        'dof_torque[10]': applied_tau[10].item(),
                        'dof_torque[11]': applied_tau[11].item(),
                        'dof_vel': dq[0].item(),
                        'dof_vel[0]': dq[0].item(),
                        'dof_vel[1]': dq[1].item(),
                        'dof_vel[2]': dq[2].item(),
                        'dof_vel[3]': dq[3].item(),
                        'dof_vel[4]': dq[4].item(),
                        'dof_vel[5]': dq[5].item(),
                        'dof_vel[6]': dq[6].item(),
                        'dof_vel[7]': dq[7].item(),
                        'dof_vel[8]': dq[8].item(),
                        'dof_vel[9]': dq[9].item(),
                        'dof_vel[10]': dq[10].item(),
                        'dof_vel[11]': dq[11].item(),
                    }
                    )
            
            elif i == stop_state_log:
                logger.plot_states()

            elapsed = time.time() - start_time
            sleep_time = cfg.sim_config.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if viewer is not None:
            listener.stop()
            viewer.close()
            print("Viewer closed successfully")
        
        global exit_flag
        exit_flag = True
        
        base_positions = np.array(base_positions)
        joint_positions = np.array(joint_positions)
        joint_vel = np.array(joint_vel)
        joint_tau = np.array(joint_tau)
        target_positions = np.array(target_positions)
        base_ang_vel = np.array(base_ang_vel)
        feet_z = np.array(feet_z)
        feet_force_z = np.array(feet_force_z)
        actions = np.array(actions)

        if args.save:
            print(f"Saving data to {cfg.sim_config.action_path}")

            directory = os.path.dirname(cfg.sim_config.action_path)

            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

            with open(cfg.sim_config.action_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(actions)):
                    writer.writerow(actions[i].tolist())
        
        num_joints = joint_positions.shape[1]
        fig, axes = plt.subplots(num_joints + 3, 1, figsize=(17, 3 * (num_joints + 2)))
        
        if num_joints + 1 == 1:
            axes = [axes]
        
        axes[0].plot(timesteps, base_positions[:, 0], label='Base X')
        axes[0].plot(timesteps, base_positions[:, 1], label='Base Y')
        axes[0].plot(timesteps, base_positions[:, 2], label='Base Z')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Position (m)')
        axes[0].set_title('Base Position Over Time')
        axes[0].legend()
        axes[0].grid(True)
              
        for i in range(num_joints):
            axes[i+1].plot(timesteps, joint_positions[:, i], label=f'real {i}')
            axes[i+1].plot(timesteps, target_positions[:, i], label=f'target {i}')
            axes[i+1].set_xlabel('Time (s)')
            axes[i+1].set_ylabel('Joint Angle (rad)')
            axes[i+1].set_title(f'Joint {i} Position')
            axes[i+1].legend()
            axes[i+1].grid(True)
        
        axes[num_joints + 1].plot(timesteps, base_ang_vel[:, 0], label='Omega X')
        axes[num_joints + 1].plot(timesteps, base_ang_vel[:, 1], label='Omega Y')
        axes[num_joints + 1].plot(timesteps, base_ang_vel[:, 2], label='Omega Z')
        axes[num_joints + 1].set_xlabel('Time (s)')
        axes[num_joints + 1].set_ylabel('Ang Vel (rad/s)')
        axes[num_joints + 1].set_title('Base Ang Vel Over Time')
        axes[num_joints + 1].legend()
        axes[num_joints + 1].grid(True)

        axes[num_joints + 2].plot(timesteps, feet_z[:, 0], label='left foot')
        axes[num_joints + 2].plot(timesteps, feet_z[:, 1], label='right foot')
        axes[num_joints + 2].set_xlabel('Time (s)')
        axes[num_joints + 2].set_ylabel('Feet Z (m)')
        axes[num_joints + 2].set_title('Feet z Over Time')
        axes[num_joints + 2].legend()
        axes[num_joints + 2].grid(True)

        plt.tight_layout()

        fig, axes = plt.subplots(num_joints, 1, figsize=(17, 3 * num_joints))

        for i in range(num_joints):
            axes[i].plot(timesteps, joint_vel[:, i], label=f'joint {i}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Vel (rad/s)')
            axes[i].set_title(f'Joint {i} Vel Over Time')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()

        fig, axes = plt.subplots(num_joints + 1, 1, figsize=(17, 3 * (num_joints + 1)))

        for i in range(num_joints):
            axes[i].plot(timesteps, joint_tau[:, i], label=f'joint {i}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Torque (Nm)')
            axes[i].set_title(f'Joint {i} Torques Over Time')
            axes[i].legend()
            axes[i].grid(True)

        axes[num_joints].plot(timesteps, feet_force_z[:, 0], label='left foot force')
        axes[num_joints].plot(timesteps, feet_force_z[:, 1], label='right foot force')
        axes[num_joints].set_xlabel('Time (s)')
        axes[num_joints].set_ylabel('Feet Force Z (N)')
        axes[num_joints].set_title('Feet Force z Over Time')
        axes[num_joints].legend()
        axes[num_joints].grid(True)

        plt.tight_layout()

        plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--task', type=str, required=True, help='task name.')
    parser.add_argument('--load_model', type=str, required=True, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--save', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    # TODO: change to X1 style
    class Sim2simCfg():
        global x_vel_max, y_vel_max, yaw_vel_max, x_vel_min, y_vel_min, yaw_vel_min
        x_vel_max = env_cfg.commands.ranges.lin_vel_x_limit[1]
        x_vel_min = env_cfg.commands.ranges.lin_vel_x_limit[0]
        y_vel_max = env_cfg.commands.ranges.lin_vel_y_limit[1]
        y_vel_min = env_cfg.commands.ranges.lin_vel_y_limit[0]
        yaw_vel_max = env_cfg.commands.ranges.ang_vel_yaw_limit[1]
        yaw_vel_min = env_cfg.commands.ranges.ang_vel_yaw_limit[0]

        class sim_config:
            mujoco_model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            action_path = env_cfg.asset.save_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sim_duration = 300.0
            dt = env_cfg.sim.dt
            decimation = env_cfg.control.decimation

        class robot_config:
            kps = np.array([env_cfg.control.stiffness[joint] for joint in env_cfg.control.stiffness.keys()], dtype=np.double)
            kds = np.array([env_cfg.control.damping[joint] for joint in env_cfg.control.damping.keys()], dtype=np.double)
            tau_limit = 200. * np.ones(env_cfg.env.joint_num, dtype=np.double)
            default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))
            init_pos = env_cfg.init_state.pos

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), env_cfg)