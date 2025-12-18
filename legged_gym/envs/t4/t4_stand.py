from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from legged_gym.envs import LeggedRobotStand

from legged_gym.utils.terrain import  HumanoidTerrain

class T4Stand(LeggedRobotStand):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        # TODO: remove gait_start ?
        if self.cfg.commands.stand_gait_phase:
            stand_envs = torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_commands_threshold
            self.gait_process[stand_envs] = 0
            if self.cfg.domain_rand.randomize_gait_start:
                phase = (self.gait_process * self.dt / self.cycle_time + self.gait_start) * (~stand_envs)
            else:
                phase = (self.gait_process * self.dt / self.cycle_time) * (~stand_envs)
        else:
            if self.cfg.domain_rand.randomize_gait_start:
                phase = self.gait_process * self.dt / self.cycle_time + self.gait_start
            else:
                phase = self.gait_process * self.dt / self.cycle_time
        return phase

    def _get_stance_mask(self):
        phase = self._get_phase()
        phases = []
        offset = self.cfg.rewards.gait_offset
        for offset_ in offset:
            leg_phase = (phase + offset_) % 1.0
            phases.append(leg_phase)
        leg_phases = torch.stack(phases, dim=-1)
        stance_mask = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device)
        for i in range(len(self.feet_indices)):
            stance_mask[:, i] = leg_phases[:, i] < self.stance_threshold_val[:]
            # stance_mask[:, i] = leg_phases[:, i] < self.cfg.rewards.stance_threshold

        # set still mask
        if not self.cfg.commands.stand_gait_phase:
            return stance_mask
        still_mask = torch.norm(self.commands[:, :3], dim=1) <= self.cfg.commands.stand_commands_threshold
        stance_mask[still_mask, :] = 1.
        return stance_mask


    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:3] = 0.  # commands
        noise_vec[3:3+self.num_actions] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[3+self.num_actions:3+2*self.num_actions] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[3+2*self.num_actions:3+3*self.num_actions] = 0.  # previous actions
        noise_vec[3+3*self.num_actions:3+3*self.num_actions+3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[3+3*self.num_actions+3:3+3*self.num_actions+6] = noise_scales.gravity  # gravity x,y,z        
        # noise_vec[75: 78] = noise_scales.quat * self.obs_scales.quat  # euler x,y,z
        if self.cfg.commands.use_gait_phase:
            noise_vec[3+3*self.num_actions+6:3+3*self.num_actions+8] = 0.  # gait phase
        return noise_vec


    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action

        return super().step(actions)

    def compute_observations(self):
        phase = self._get_phase()

        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_stance_mask()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_threshold

        self.command_input = self.commands[:, :3] * self.commands_scale

        # TODO: not use rand_push...
        if self.cfg.commands.use_gait_phase:
            self.privileged_obs_buf = torch.cat((
                self.command_input,  # 3
                (self.dof_pos - self.default_joint_pd_target) * \
                self.obs_scales.dof_pos,  # 23
                self.dof_vel * self.obs_scales.dof_vel,  # 23
                self.actions * self.obs_scales.actions,  # 23
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity * self.obs_scales.gravity,
                # self.base_euler_xyz * self.obs_scales.quat,  # 3
                sin_phase,
                cos_phase,
                self.rand_push_force[:, :2],  # 2 
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 10.,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ), dim=-1)
        else:
            self.privileged_obs_buf = torch.cat((
                self.command_input,  # 3
                (self.dof_pos - self.default_joint_pd_target) * \
                self.obs_scales.dof_pos,  # 23
                self.dof_vel * self.obs_scales.dof_vel,  # 23
                self.actions * self.obs_scales.actions,  # 23
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.projected_gravity * self.obs_scales.gravity,
                # self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 2 
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 10.,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ), dim=-1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        if self.cfg.domain_rand.randomize_obs_motor_latency:
            self.obs_motor = self.obs_motor_latency_buffer[torch.arange(self.num_envs), :, self.obs_motor_latency_simstep.long()]
        else:
            self.obs_motor = torch.cat((q, dq), 1)

        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.obs_imu = self.obs_imu_latency_buffer[torch.arange(self.num_envs), :, self.obs_imu_latency_simstep.long()]
        else:              
            # self.obs_imu = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.base_euler_xyz * self.obs_scales.quat), 1)
            self.obs_imu = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.projected_gravity * self.obs_scales.gravity), 1)

        if self.cfg.commands.use_gait_phase:
            obs_buf = torch.cat((
                self.command_input,  # 3
                self.obs_motor,
                self.actions * self.obs_scales.actions,   # 23
                self.obs_imu,
                sin_phase,
                cos_phase,
            ), dim=-1)
        else:
            obs_buf = torch.cat((
                self.command_input,  # 3
                self.obs_motor,
                self.actions * self.obs_scales.actions,   # 23
                self.obs_imu,
            ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((obs_buf, heights), dim=-1)

        if self.add_noise:  
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # TODO: fix amp obs
    def get_amp_obs_for_expert_trans(self):
        self.base_pos_z = self.root_states[:, 2:3]
        self.left_hand_local_vec = torch.tensor(self.cfg.asset.left_hand_local_vec, device=self.device).repeat((self.num_envs, 1))
        self.right_hand_local_vec = torch.tensor(self.cfg.asset.right_hand_local_vec, device=self.device).repeat((self.num_envs, 1))
        self.left_foot_local_vec = torch.tensor(self.cfg.asset.left_foot_local_vec, device=self.device).repeat((self.num_envs, 1))
        self.right_foot_local_vec = torch.tensor(self.cfg.asset.right_foot_local_vec, device=self.device).repeat((self.num_envs, 1))

        self.left_arm_dof_pos = self.dof_pos[:, self.left_arm_ids]
        self.right_arm_dof_pos = self.dof_pos[:, self.right_arm_ids]
        self.waist_dof_pos = self.dof_pos[:, self.waist_ids]
        self.left_leg_dof_pos = self.dof_pos[:, self.left_leg_ids]
        self.right_leg_dof_pos = self.dof_pos[:, self.right_leg_ids]

        self.left_arm_dof_vel = self.dof_vel[:, self.left_arm_ids]
        self.right_arm_dof_vel = self.dof_vel[:, self.right_arm_ids]
        self.waist_dof_vel = self.dof_vel[:, self.waist_ids]
        self.left_leg_dof_vel = self.dof_vel[:, self.left_leg_ids]
        self.right_leg_dof_vel = self.dof_vel[:, self.right_leg_ids]

        left_hand_pos_w = (
            self.rigid_state[:, self.hand_body_ids[0], :3]
            - self.root_states[:, 0:3]
            + quat_apply(self.rigid_state[:, self.hand_body_ids[0], 3:7], self.left_hand_local_vec)
        )
        left_hand_pos_b = quat_rotate_inverse(self.root_states[:, 3:7], left_hand_pos_w)
        right_hand_pos_w = (
            self.rigid_state[:, self.hand_body_ids[1], :3]
            - self.root_states[:, 0:3]
            + quat_apply(self.rigid_state[:, self.hand_body_ids[1], 3:7], self.right_hand_local_vec)
        )
        right_hand_pos_b = quat_rotate_inverse(self.root_states[:, 3:7], right_hand_pos_w)
        left_foot_pos_w = (
            self.rigid_state[:, self.feet_body_ids[0], :3] 
            - self.root_states[:, 0:3]
            + quat_apply(self.rigid_state[:, self.feet_body_ids[0], 3:7], self.left_foot_local_vec)
        )
        left_foot_pos_b = quat_rotate_inverse(self.root_states[:, 3:7], left_foot_pos_w)
        right_foot_pos_w = (
            self.rigid_state[:, self.feet_body_ids[1], :3] 
            - self.root_states[:, 0:3]
            + quat_apply(self.rigid_state[:, self.feet_body_ids[1], 3:7], self.right_foot_local_vec)
        )
        right_foot_pos_b = quat_rotate_inverse(self.root_states[:, 3:7], right_foot_pos_w)

        return torch.cat(
            (
                self.base_pos_z,
                self.projected_gravity,
                self.base_lin_vel,
                self.base_ang_vel,
                self.left_arm_dof_pos,
                self.right_arm_dof_pos,
                self.waist_dof_pos,
                self.left_leg_dof_pos,
                self.right_leg_dof_pos,
                self.left_arm_dof_vel,
                self.right_arm_dof_vel,
                self.waist_dof_vel,
                self.left_leg_dof_vel,
                self.right_leg_dof_vel,
                left_hand_pos_b,
                right_hand_pos_b,
                left_foot_pos_b,
                right_foot_pos_b,
            ),
            dim=-1,
        )

# ================================================ Rewards ================================================== #

    def _reward_track_lin_vel_xy(self):
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_track_ang_vel_z(self):
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_alive(self):
        return 1.0
    
    def _reward_base_linear_velocity(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_base_angular_velocity(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_joint_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_joint_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_vel_limits(self):
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _reward_energy(self):
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=-1)
    
    def _reward_joint_deviation_arms(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        joint_diff[:, self.elbows_indices] += self.elbow_val[:].unsqueeze(1)
        arms_diff = joint_diff[:, self.arms_indices]
        return torch.sum(torch.abs(arms_diff), dim=1)
    
    def _reward_joint_deviation_swing_arms(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        if self.cfg.rewards.swing_arm:
            phase = self._get_phase()
            sin_pos = torch.sin(2 * torch.pi * phase)
            sin_pos_l = sin_pos.clone()
            sin_pos_r = sin_pos.clone()
            sin_pos_l[sin_pos_l < 0] = 0
            joint_diff[:, self.shoulders_indices[0]] += self.swing_val * sin_pos_l
            sin_pos_r[sin_pos_r > 0] = 0
            joint_diff[:, self.shoulders_indices[1]] -= self.swing_val * sin_pos_r
        swing_arms_diff = joint_diff[:, self.swing_arms_indices]
        return torch.sum(torch.abs(swing_arms_diff), dim=1)
    
    def _reward_joint_deviation_waists(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        waists_diff = joint_diff[:, self.waists_indices]
        return torch.sum(torch.abs(waists_diff), dim=1)
    
    def _reward_joint_deviation_legs(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        legs_diff = joint_diff[:, self.legs_indices]
        return torch.sum(torch.abs(legs_diff), dim=1)
    
    def _reward_flat_orientation_l2(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_base_height(self):
        return torch.square(self.root_states[:, 2] - self.cfg.rewards.base_height_target)
    
    def _reward_gait(self):
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.contact_threshold
        stance_mask = self._get_stance_mask()
        for i in range(len(self.feet_indices)):
            reward += torch.where(contact[:, i] == stance_mask[:, i], 1, 0)
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward * (cmd_norm > self.cfg.commands.stand_commands_threshold)

    def _reward_feet_slide(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > self.cfg.rewards.contact_threshold
        feet_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        return torch.sum(feet_speed_norm * contact, dim=1)

    def _reward_feet_clearance(self):
        feet_z_target_error = torch.square(self.rigid_state[:, self.feet_indices, 2] - self.feet_height_val[:].unsqueeze(1))
        # feet_z_target_error = torch.square(self.rigid_state[:, self.feet_indices, 2] - self.cfg.rewards.target_feet_height)
        feet_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        feet_vel_tanh = torch.tanh(self.cfg.rewards.tanh_mult * feet_speed_norm)
        return torch.exp(-torch.sum(feet_z_target_error * feet_vel_tanh, dim=1) / self.cfg.rewards.clearance_sigma)
    
    def _reward_feet_contact_forces(self):
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_slip(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > self.cfg.rewards.contact_threshold
        contact_feet_vel = self.rigid_state[:, self.feet_indices, 7:10] * contact.unsqueeze(-1)
        penalize = torch.sum(torch.square(contact_feet_vel[:, :, :3]), dim=2)
        return torch.sum(penalize, dim=1)

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > self.cfg.rewards.contact_threshold
        pos_error = torch.square(self.rigid_state[:, self.feet_indices, 2] - self.feet_height_val[:].unsqueeze(1))
        return torch.sum(pos_error * ~contact, dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_torque_tiredness(self):
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=1)

    def _reward_undesired_contacts(self):
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > self.cfg.rewards.contact_threshold, dim=1)

    def _reward_stand_still(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        stand_diff = joint_diff[:, self.stand_indices]
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return torch.sum(torch.abs(stand_diff), dim=1) * (cmd_norm <= self.cfg.commands.stand_commands_threshold)

    def _reward_key_frame(self):
        pass