from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class T4StandCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 5
        c_frame_stack = 5
        num_single_obs = 78
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 92
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 23
        num_envs = 4096
        episode_length_s = 20  # episode length in seconds
        use_ref_actions = False
        joint_num = 23
        ref_state_init = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 1.0

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/zoomlion_robot_description-main/T4_v2/urdf/t4_v2.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/zoomlion_robot_description-main/T4_v2/xml/t4_v2.xml'
        save_file = '{LEGGED_GYM_ROOT_DIR}/data/t4/T4_stand.csv'

        name = "t4_stand"
        foot_name = "foot_link"  # TODO: ["left_foot_link", "right_foot_link"]
        feet_name = ["left_foot_link", "right_foot_link"]
        knee_name = "Shank"
        imu_name = ""  # TODO: add imu name

        arms_name = ['J_arm_l_02', 'J_arm_l_03', 'J_arm_l_04', 'J_arm_l_05',
                     'J_arm_r_02', 'J_arm_r_03', 'J_arm_r_04', 'J_arm_r_05']
        swing_arms_name = ['J_arm_l_01', 'J_arm_r_01']
        shoulders_name = ['J_arm_l_01', 'J_arm_r_01']
        elbows_name = ['J_arm_l_04', 'J_arm_r_04']
        waists_name = ['J_waist']
        legs_name = ['J_hip_l_roll', 'J_hip_l_yaw',
                     'J_hip_r_roll', 'J_hip_r_yaw']
        stand_name = ['J_arm_l_01', 'J_arm_l_02', 'J_arm_l_03', 'J_arm_l_04', 'J_arm_l_05',
                      'J_arm_r_01', 'J_arm_r_02', 'J_arm_r_03', 'J_arm_r_04', 'J_arm_r_05',
                      'J_waist',
                      'J_hip_l_pitch', 'J_hip_l_roll', 'J_hip_l_yaw', 'J_knee_l_pitch', 'J_ankle_l_pitch',
                      'J_ankle_l_roll',
                      'J_hip_r_pitch', 'J_hip_r_roll', 'J_hip_r_yaw', 'J_knee_r_pitch', 'J_ankle_r_pitch',
                      'J_ankle_r_roll']

        left_arm = ['J_arm_l_01', 'J_arm_l_02', 'J_arm_l_03', 'J_arm_l_04', 'J_arm_l_05']
        right_arm = ['J_arm_r_01', 'J_arm_r_02', 'J_arm_r_03', 'J_arm_r_04', 'J_arm_r_05']
        waist = ['J_waist']
        left_leg = ['J_hip_l_pitch', 'J_hip_l_roll', 'J_hip_l_yaw', 'J_knee_l_pitch', 'J_ankle_l_pitch',
                    'J_ankle_l_roll']
        right_leg = ['J_hip_r_pitch', 'J_hip_r_roll', 'J_hip_r_yaw', 'J_knee_r_pitch', 'J_ankle_r_pitch',
                     'J_ankle_r_roll']

        hand_body = ['AL5', 'AR5']
        feet_body = ["left_foot_link", "right_foot_link"]

        left_hand_local_vec = [0.0, 0.0, -0.1405]
        right_hand_local_vec = [0.0, 0.0, -0.1405]
        left_foot_local_vec = [0.0, 0.0, 0.0]
        right_foot_local_vec = [0.0, 0.0, 0.0]

        terminate_after_contacts_on = ['Trunk']
        penalize_contacts_on = ["Trunk"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # TODO: how to set trimesh cfg
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 1.0  # TODO: 0.6
        dynamic_friction = 1.0  # TODO: 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            lin_vel = 0.2
            quat = 0.05
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'J_arm_l_01': 0.2,
            'J_arm_l_02': 0.12,
            'J_arm_l_03': 0,
            'J_arm_l_04': -0.5,
            'J_arm_l_05': 0,
            'J_arm_r_01': 0.2,
            'J_arm_r_02': -0.12,
            'J_arm_r_03': 0,
            'J_arm_r_04': -0.5,
            'J_arm_r_05': 0,
            'J_waist_yaw': 0,
            'J_hip_l_pitch': -0.2,
            'J_hip_l_roll': 0,
            'J_hip_l_yaw': 0,
            'J_knee_l_pitch': 0.4,
            'J_ankle_l_pitch': -0.2,
            'J_ankle_l_roll': 0,
            'J_hip_r_pitch': -0.2,
            'J_hip_r_roll': 0,
            'J_hip_r_yaw': 0,
            'J_knee_r_pitch': 0.4,
            'J_ankle_r_pitch': -0.2,
            'J_ankle_r_roll': 0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        # TODO: order is same as mujoco joint order
        stiffness = {'J_arm_l_01': 40, 'J_arm_l_02': 40, 'J_arm_l_03': 40, 'J_arm_l_04': 40, 'J_arm_l_05': 40,
                     'J_arm_r_01': 40, 'J_arm_r_02': 40, 'J_arm_r_03': 40, 'J_arm_r_04': 40, 'J_arm_r_05': 40,
                     'J_waist_yaw': 200,
                     'J_hip_l_pitch': 100, 'J_hip_l_roll': 100, 'J_hip_l_yaw': 100, 'J_knee_l_pitch': 150,
                     'J_ankle_l_pitch': 40, 'J_ankle_l_roll': 40,
                     'J_hip_r_pitch': 100, 'J_hip_r_roll': 100, 'J_hip_r_yaw': 100, 'J_knee_r_pitch': 150,
                     'J_ankle_r_pitch': 40, 'J_ankle_r_roll': 40}
        damping = {'J_arm_l_01': 1, 'J_arm_l_02': 1, 'J_arm_l_03': 1, 'J_arm_l_04': 1, 'J_arm_l_05': 1,
                   'J_arm_r_01': 1, 'J_arm_r_02': 1, 'J_arm_r_03': 1, 'J_arm_r_04': 1, 'J_arm_r_05': 1,
                   'J_waist_yaw': 5,
                   'J_hip_l_pitch': 2, 'J_hip_l_roll': 2, 'J_hip_l_yaw': 2, 'J_knee_l_pitch': 4,
                   'J_ankle_l_pitch': 2, 'J_ankle_l_roll': 2,
                   'J_hip_r_pitch': 2, 'J_hip_r_roll': 2, 'J_hip_r_yaw': 2, 'J_knee_r_pitch': 4,
                   'J_ankle_r_pitch': 2, 'J_ankle_r_roll': 2}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        # TODO: fix the play.py bug
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        num_buckets = 64

        randomize_friction = True
        friction_range = [0.2, 1.3]

        randomize_restitution = False
        restitution_range = [0.0, 0.4]

        push_robots = True
        push_interval_s = 5  # every this second, push robot
        # TODO: in t4_stand.pt use self.cfg.num_steps_per_env change 24
        update_step = 2000 * 24  # after this count, increase push_duration index, iterations * num_steps_per_env
        push_duration = [0]  # [0, 0.05, 0.1, 0.15, 0.2, 0.25] # increase push duration during training
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_base_mass = True
        added_base_mass_range = [-4.0, 4.0]

        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]

        randomize_calculated_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_joint_friction = True
        joint_friction_range = [0.01, 1.15]

        randomize_joint_damping = True
        joint_damping_range = [0.3, 1.5]

        randomize_joint_armature = True
        joint_armature_range = [0.008, 0.06]

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [1, 5]

        add_obs_latency = True  # no latency for obs_action
        randomize_obs_motor_latency = True
        randomize_obs_imu_latency = True
        range_obs_motor_latency = [1, 5]
        range_obs_imu_latency = [1, 5]

        randomize_gait_start = True

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        use_gait_phase = False  # TODO: add gait phase
        stand_gait_phase = True  # stand still when use gait phase
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        low_commands_proportion = 0.
        low_commands_threshold = 0.3
        stand_commands_proportion = 0.
        stand_commands_threshold = 0.1
        curriculum_lin_threshold = 0.8
        curriculum_ang_threshold = 0.8

        class ranges:
            # TODO: step 1: 1.0, step 2: 3.0
            lin_vel_x = [-0.1, 0.1]  # min max [m/s]
            lin_vel_y = [-0.1, 0.1]  # min max [m/s]
            ang_vel_yaw = [-0.1, 0.1]  # min max [rad/s]
            heading = [-3.14, 3.14]
            lin_vel_x_limit = [-0.5, 1.0]
            lin_vel_y_limit = [-0.5, 0.5]
            ang_vel_yaw_limit = [-1.0, 1.0]

    class rewards:
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.72
        stance_threshold = [0.55, 0.55, ]  # 0.5, 0.45, 0.4, 0.35]  # TODO: [0.55]
        target_feet_height = [0.1, 0.15, ]  # 0.2, 0.25, 0.3, 0.35]  # TODO: [0.1]
        swing_arm = True
        target_swing = [0.1, 0.15, ]  # 0.2, 0.25, 0.3, 0.35]
        target_elbow = [0.0, 0.0, ]  # 0.2, 0.4, 0.6, 0.8]
        gait_offset = [0.0, 0.5]
        cycle_time = [0.8, 0.8,] # 0.7, 0.6, 0.5, 0.4]  # TODO: [0.8]
        cycle_range = [0.95, 1.05]
        only_positive_rewards = True
        tracking_sigma = 0.25
        tanh_mult = 2.
        clearance_sigma = 0.05
        track_vel_threshold = 0.5
        contact_threshold = 1.  # TODO: 5.
        max_contact_force = 500

        class scales:
            # task
            track_lin_vel_xy = 1.
            track_ang_vel_z = 0.5
            alive = 0.15
            # base
            base_linear_velocity = -2.
            base_angular_velocity = -0.05
            joint_vel = -0.001
            joint_acc = -2.5e-7
            action_rate = -0.05
            dof_pos_limits = -5.
            energy = -2e-5
            joint_deviation_arms = -0.5
            joint_deviation_swing_arms = -0.05  # TODO: step 1: -0.5, step 2: -0.05
            joint_deviation_waists = -1.
            joint_deviation_legs = -1.
            # robot
            flat_orientation_l2 = -5.
            base_height = -10.
            # feet
            gait = 0.5
            feet_slide = -0.2
            feet_clearance = 1.
            # other
            undesired_contacts = -1.

    class normalization:
        class obs_scales:
            lin_vel = 1.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 1.
            actions = 1.
            quat = 1.
            gravity = 1.
            height_measurements = 1.

        clip_observations = 100.
        clip_actions = 100.


class T4StandCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'  # DWLOnPolicyRunner

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        sym_loss = True
        # TODO: base_euler_xyz vs projected_gravity
        obs_permutation = [
            # commands
            0.0001, -1, -2,  # x, y, yaw
            # dof pos related
            8, -9, -10, 11, -12,  # right arm
            3, -4, -5, 6, -7,  # left arm
            -13,  # waist
            20, -21, -22, 23, 24, -25,  # right leg
            14, -15, -16, 17, 18, -19,  # left leg
            # dof vel related
            31, -32, -33, 34, -35,  # right arm
            26, -27, -28, 29, -30,  # left arm
            -36,  # waist
            43, -44, -45, 46, 47, -48,  # right leg
            37, -38, -39, 40, 41, -42,  # left leg
            # action related
            54, -55, -56, 57, -58,  # right arm
            49, -50, -51, 52, -53,  # left arm
            -59,  # waist
            66, -67, -68, 69, 70, -71,  # right leg
            60, -61, -62, 63, 64, -65,  # left leg
            # base related
            -72, 73, -74,  # base_ang_vel, roll, pitch, yaw
            75, -76, 77,  # base_projected_gravity, x, y, z
            # -75, 76, -77,  # base_eu_ang, roll, pitch, yaw
            # -78, -79,  # gait phase
        ]
        act_permutation = [
            # action related
            5, -6, -7, 8, -9,  # right arm
            0.0001, -1, -2, 3, -4,  # left arm
            -10,  # waist
            17, -18, -19, 20, 21, -22,  # right leg
            11, -12, -13, 14, 15, -16,  # left leg
        ]
        frame_stack = 5
        sym_coef = 1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 't4_stand_ppo'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt


