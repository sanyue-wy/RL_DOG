from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO

# RC 机器人环境配置
class RCBlindPlaneCfg(LeggedRobotCfg):
    # 环境基础配置
    class env(LeggedRobotCfg.env):
        num_envs = 4096  # 并行环境数量；越大采样越快，但显存和仿真负载也越高
        num_one_step_observations = 46  # 单个时刻提供给策略的基础观测维度；额外加入 1 维高度命令
        num_observations = num_one_step_observations * 6  # 最终策略观测维度；这里表示堆叠 6 帧历史观测
        num_one_step_privileged_obs = 46 + 3 + 3 + 187 # 单个时刻的特权观测维度；额外包含 base_lin_vel、external_forces、scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # critic 使用的特权观测总维度；若为 None 则 step() 不返回 privileged_obs
        num_actions = 12  # 动作维度；对应 12 个可控关节
        env_spacing = 3.  # 环境间距；仅在 plane 这类简单地面上有意义，heightfield/trimesh 下通常不使用
        send_timeouts = True # 是否把超时终止信息传给算法；用于区分“超时结束”和“失败结束”
        episode_length_s = 20 # 单个 episode 的最大时长，单位秒

    # 地形配置
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # 地形网格类型；这里使用三角网格地形，适合生成坡、楼梯、高台、坑洞等复杂盲走地形
        horizontal_scale = 0.1 # 地形横向分辨率，单位米；值越小，坡面边缘、踏块边界、沟壑轮廓会越细致
        vertical_scale = 0.005 # 地形纵向分辨率，单位米；值越小，高台、楼梯、坑洞的高度量化越精细
        border_size = 25 # 地形外围平地区域宽度，单位米；用于给整张训练地形留出边界缓冲，避免机器人刷在地图边缘
        curriculum = True # 是否启用地形课程学习；开启后会从低难地形逐步过渡到更陡的坡、更高的台阶、更难的障碍
        static_friction = 1.0 # 地面静摩擦系数；影响坡地、楼梯、高台边缘起步和站立时是否容易打滑
        dynamic_friction = 1.0 # 地面动摩擦系数；影响机器人在粗糙坡、不平路、离散踏块上滑动时的阻力
        restitution = 0. # 地面恢复系数；控制脚踩到地形后的弹性反弹程度，复杂地形训练一般保持较低以减少弹跳
        # rough terrain only:
        measure_heights = True # 是否启用高度扫描观测；盲走 rough terrain、楼梯、高台、坑洞地形时通常必须开启
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 高度扫描在机体前后方向的采样点；用于感知前方坡面、台阶边、高台落差和坑洞位置
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 高度扫描在机体左右方向的采样点；用于感知侧坡、横向错位踏块、左右不平路和边缘落差
        selected = False # 是否固定只生成一种地形；False 表示按比例混合多类地形，适合综合地形训练
        terrain_kwargs = None # 当 selected=True 时，用来给指定单一地形传参数；综合训练下通常不用
        max_init_terrain_level = 5 # 初始可出生的最大难度等级；只影响训练早期起始地形有多难，不改变全局最高难度上限
        terrain_length = 8. # 单块子地形长度，单位米；决定每个坡、楼梯、高台区块在前进方向上的可用距离
        terrain_width = 8. # 单块子地形宽度，单位米；决定每个侧坡、踏块区、坑洞区在横向上的覆盖范围
        num_rows= 10 # 地形难度层数；行数越多，课程学习的难度等级越细，最高难地形也会覆盖到更多层级，决定最高等级上限
        num_cols = 25 # 地形类型列数；列数越多，同一难度下可并行生成的坡、楼梯、高台、踏块、坑洞等类型越丰富
        # terrain types: [smooth slope, rough slope, stairs up/down, discrete obstacles, stepping stones, gaps, pits]
        terrain_proportions = [0.1, 0.2, 0.3, 0.2, 0.2] # 各类地形生成比例；依次控制平滑坡、粗糙坡、上下楼梯、离散高台/方块障碍、踏石路、沟壑和坑洞在训练地图中的占比
        # trimesh only:
        slope_treshold = 0.75 # 坡度修正阈值；超过该阈值的高度场边缘会更接近陡坎/立边，主要影响陡坡、台阶边和高台侧壁的形状

    # 指令采样配置
    class commands(LeggedRobotCfg.commands):
        curriculum = True # 是否启用指令课程学习；训练中逐渐放宽命令范围
        max_curriculum = 1.0 # 指令课程学习的最大放宽尺度
        num_commands = 5 # 指令维度；依次为 lin_vel_x、lin_vel_y、ang_vel_yaw、heading、base_height
        resampling_time = 10. # 指令重采样时间间隔，单位秒；每隔该时间重新抽样一次命令
        heading_command = True # 是否启用 heading 模式；开启后会由 heading 误差自动换算出 yaw 角速度命令
        height_command = True # 是否启用高度命令；关闭后退回旧版 4 维命令和 45 维单帧 actor 观测
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # 前向/后向线速度命令范围，单位 m/s
            lin_vel_y = [-0.5, 0.5]   # 横向线速度命令范围，单位 m/s
            ang_vel_yaw = [-2.5, 2.5]    # 偏航角速度命令范围，单位 rad/s
            heading = [-3.14, 3.14]  # 目标朝向角范围，单位 rad；仅 heading_command=True 时有意义
            height = [0.10, 0.26]  # 机身目标高度命令范围，单位米；平地训练围绕该范围跟踪

    # 初始状态配置
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42] # 机器人初始基座位置 [x, y, z]，单位米
        default_joint_angles = { # 默认关节角；当策略动作 action=0 时，各关节的目标角度就是这里的值
            'FL_hip_joint': 0.,   # 左前髋外展/内收关节默认角度，单位 rad
            'RL_hip_joint': 0.,   # 左后髋外展/内收关节默认角度，单位 rad
            'FR_hip_joint': -0. ,  # 右前髋外展/内收关节默认角度，单位 rad
            'RR_hip_joint': -0.,   # 右后髋外展/内收关节默认角度，单位 rad

            'FL_thigh_joint': 0.8,     # 左前大腿关节默认角度，单位 rad
            'RL_thigh_joint': 1.,   # 左后大腿关节默认角度，单位 rad
            'FR_thigh_joint': 0.8,     # 右前大腿关节默认角度，单位 rad
            'RR_thigh_joint': 1.,   # 右后大腿关节默认角度，单位 rad

            'FL_calf_joint': -1.5,   # 左前小腿关节默认角度，单位 rad
            'RL_calf_joint': -1.5,    # 左后小腿关节默认角度，单位 rad
            'FR_calf_joint': -1.5,  # 右前小腿关节默认角度，单位 rad
            'RR_calf_joint': -1.5,    # 右后小腿关节默认角度，单位 rad
        }

    # 关节控制器配置
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P' 
        stiffness = {'joint': 40.0}  # 位置刚度 Kp，单位 N*m/rad；
        damping = {'joint': 1.0}     # 速度阻尼 Kd，单位 N*m*s/rad；
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # 动作缩放系数；策略输出会先乘该值，再叠加到默认关节角上作为目标角
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 控制降采样比；每个策略动作会维持 4 个仿真步
        hip_reduction = 1.0  # 髋关节动作缩放附加系数；1.0 表示不额外缩小髋关节动作幅度

    # 机器人资产与碰撞配置
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/RCV3/urdf/RCV3.urdf'  # URDF 机器人模型路径
        name = "rc"  # 机器人资产名称；创建 actor 时使用
        foot_name = "foot"  # 足端刚体名称匹配关键字；用于寻找 feet_indices
        penalize_contacts_on = ["thigh", "calf", "base"]  # 这些刚体发生接触时计入碰撞惩罚
        terminate_after_contacts_on = ["base"]  # 这些刚体接触后立即判定 episode 终止
        privileged_contacts_on = ["base", "thigh", "calf"]  # 这些接触信息会作为 critic 的特权信息使用
        self_collisions = 1 # 自碰撞开关；1 表示禁用自碰撞，0 表示启用自碰撞
        flip_visual_attachments = False # 是否把部分视觉网格从 y-up 翻转到 z-up；取决于模型网格坐标系


    # 域随机化配置
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = True  # 是否随机化机体额外负载质量
        payload_mass_range = [-1, 1.5]  # 额外负载质量随机范围，单位 kg

        randomize_com_displacement = True  # 是否随机化质心偏移
        com_displacement_range = [-0.05, 0.05]  # 质心偏移随机范围，单位米

        randomize_link_mass = True  # 是否随机化各连杆质量
        link_mass_range = [0.9, 1.1]  # 连杆质量缩放范围；1.0 表示原始质量
        
        randomize_friction = True  # 是否随机化摩擦系数
        friction_range = [0.2, 1.25]  # 摩擦系数随机范围
        
        randomize_restitution = True  # 是否随机化恢复系数也就是碰撞弹性
        restitution_range = [0., 1.0]  # 恢复系数随机范围
        
        randomize_motor_strength = True  # 是否随机化电机输出强度
        motor_strength_range = [0.9, 1.1]  # 电机强度缩放范围
        
        randomize_kp = True  # 是否随机化刚度 Kp
        kp_range = [0.8, 1.2]  # Kp 缩放范围
        
        randomize_kd = True  # 是否随机化阻尼 Kd
        kd_range = [0.8, 1.2]  # Kd 缩放范围
        
        randomize_initial_joint_pos = True  # 是否随机化初始关节姿态
        initial_joint_pos_range = [0.5, 1.5]  # 初始关节角缩放范围
        
        disturbance = False  # 是否施加外部三维随机方向扰动力
        disturbance_range = [-30.0, 30.0]  # 外部扰动力范围，单位 N 或 N*m，取决于实现
        disturbance_interval = 8  # 外部扰动施加间隔，单位秒
        
        push_robots = True  # 是否周期性二维随机方向推机器人，可优先开
        push_interval_s = 16  # 推机器人时间间隔，单位秒
        max_push_vel_xy = 1.  # 推动后在平面内引入的最大速度扰动，单位 m/s

        delay = True  # 是否引入动作执行延迟

    # 奖励函数配置
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0  # 终止惩罚权重；当前为 0，表示不单独惩罚失败终止
            tracking_lin_vel = 3.0  # 线速度跟踪奖励权重；鼓励机体跟踪给定 x/y 速度命令
            tracking_ang_vel = 2.0  # 角速度跟踪奖励权重；鼓励机体跟踪给定 yaw 角速度命令
            lin_vel_z = -1.8  # z 方向线速度惩罚权重；抑制机体上下乱跳
            ang_vel_xy = -0.08  # x/y 方向角速度惩罚权重；抑制滚转和俯仰角速度过大
            orientation = -0.6  # 姿态惩罚权重；鼓励机身保持接近平衡姿态
            orientation_pitch = -1.2 # 俯仰角惩罚权重；鼓励机身保持接近平衡姿态
            dof_acc = -2.5e-7  # 关节加速度惩罚权重；抑制关节剧烈加减速
            joint_power = -2e-5  # 关节功率项权重；当前为 0，表示不启用功率约束
            base_height = -3.0  # 机身高度惩罚权重；鼓励机身高度接近目标值
            base_height_encourage = 2.0 # 机身高度鼓励奖励权重；鼓励机身高度接近目标值
            foot_clearance = -0.25  # 足端净空高度相关项权重；当前实现是对偏离目标高度的运动进行惩罚
            action_rate = -0.04  # 动作变化率惩罚权重；抑制相邻两步动作变化过大
            smoothness = -0.02  # 二阶平滑惩罚权重；抑制动作序列出现明显拐点和高频抖动
            feet_air_time = 0.0  # 足端腾空时间奖励权重；鼓励形成更清晰的摆动相
            collision = -2.5  # 碰撞惩罚权重；惩罚不希望发生接触的刚体碰撞
            stumble = -0.0  # 绊腿惩罚权重；惩罚脚撞到近似垂直障碍物
            stand_still = -2.0 # 静止站立惩罚权重；
            torques = -0.0001  # 力矩惩罚权重；当前为 0，不约束关节输出力矩大小
            dof_vel = -2.5e-6  # 关节速度惩罚权重；当前为 0，不约束关节转速
            dof_pos_limits = -0.0  # 关节位置限位惩罚权重；当前为 0，不额外惩罚逼近位置极限
            dof_vel_limits = -0.0  # 关节速度限位惩罚权重；当前为 0，不额外惩罚逼近速度极限
            torque_limits = -0.12  # 力矩限位惩罚权重；惩罚逼近力矩极限
            hip_abduction_deviation = -0.8  # 髋外展/内收关节偏离默认姿态惩罚权重；用于约束横向开腿过大
            foot_drag = -0.05 # 足端拖地惩罚权重；鼓励足端在摆动相腾空，减少磨擦和能量损失

            similar_legged = 2.0 # 左右腿动作相似性惩罚权重；鼓励左右腿动作对称，减少不稳定的横向摆动
            vel_y_zero_penalize = -0.1 # 横向速度为零惩罚权重；鼓励机器人在横向有一定速度，减少横向静止
            low_height_thigh_horizontal = 1.0 
        only_positive_rewards = False # 是否把总奖励裁剪为非负；可避免训练早期大量负奖励导致学习不稳定
        tracking_sigma = 0.25 # 速度跟踪奖励的高斯宽度；越小表示对跟踪误差越敏感
        soft_dof_pos_limit = 0.9 # 软位置限位比例；超过 URDF 极限 90% 后开始进入惩罚区
        soft_dof_vel_limit = 0.9  # 软速度限位比例；超过速度极限 90% 后开始进入惩罚区
        soft_torque_limit = 0.9  # 软力矩限位比例；超过力矩极限 90% 后开始进入惩罚区
        base_height_target = 0.26  # 机身目标高度，单位米；base_height 奖励围绕该值计算
        max_contact_force = 100. # 最大允许接触力阈值；超过后可进入接触力惩罚
        clearance_height_target = -0.16  # 足端目标净空高度，单位米；foot_clearance 奖励围绕该值计算

    # 观测与动作归一化配置
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0  # 线速度观测缩放系数
            ang_vel = 0.25  # 角速度观测缩放系数
            height = 4.0  # 高度命令观测缩放系数
            dof_pos = 1.0  # 关节位置观测缩放系数
            dof_vel = 0.05  # 关节速度观测缩放系数
            height_measurements = 5.0  # 地形高度观测缩放系数
        clip_observations = 100.  # 观测裁剪阈值；超过该范围的观测会被截断
        clip_actions = 100.  # 动作裁剪阈值；超过该范围的策略输出会被截断

    # 观测噪声配置
    class noise(LeggedRobotCfg.noise):
        add_noise = True  # 是否向观测注入噪声
        noise_level = 1.5 # 全局噪声缩放系数；会统一放大或缩小各类噪声幅度
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.015  # 关节位置观测噪声幅度,gai
            dof_vel = 2.5  # 关节速度观测噪声幅度,gai
            lin_vel = 0.1  # 线速度观测噪声幅度
            ang_vel = 0.35  # 角速度观测噪声幅度,gai
            gravity = 0.05  # 重力方向观测噪声幅度
            height_measurements = 0.1  # 地形高度观测噪声幅度

    # viewer camera:
    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0  # 可视化时跟踪的参考环境编号
        pos = [10, 0, 6]  # 相机位置 [x, y, z]，单位米
        lookat = [11., 5, 3.]  # 相机朝向目标点 [x, y, z]，单位米

    # 仿真器配置
    class sim(LeggedRobotCfg.sim):
        dt =  0.005  # 仿真时间步长，单位秒
        substeps = 1  # 每个仿真步内部的子步数
        gravity = [0., 0. ,-9.81]  # 重力加速度，单位 m/s^2
        up_axis = 1  # 世界坐标系向上轴；0 表示 y 轴向上，1 表示 z 轴向上

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10  # PhysX CPU 线程数
            solver_type = 1  # 求解器类型；0 为 PGS，1 为 TGS
            num_position_iterations = 4  # 位置约束迭代次数；越大接触/约束越稳定但更慢
            num_velocity_iterations = 0  # 速度约束迭代次数
            contact_offset = 0.01  # 接触偏移距离，单位米；物体接近到该范围内开始生成接触
            rest_offset = 0.0   # 静止接触偏移距离，单位米
            bounce_threshold_velocity = 0.5 # 反弹阈值速度；低于该速度时通常不产生明显弹跳
            max_depenetration_velocity = 1.0  # 最大去穿透修正速度；限制物体互穿后的分离速度
            max_gpu_contact_pairs = 2**23 # GPU 最大接触对数量上限；环境数很多时需要足够大
            default_buffer_size_multiplier = 5  # PhysX 默认缓冲区扩展倍率
            contact_collection = 2 # 接触采集模式；0 从不采集，1 仅最后子步，2 采集所有子步

# PPO 训练配置
class RCBlindPlaneCfgPPO(LeggedRobotCfgPPO):
    seed = 1  # 随机种子；用于保证训练可复现性
    runner_class_name = 'HIMOnPolicyRunner'  # 训练 runner 类名；指定使用哪种训练驱动器
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0  # 策略初始动作噪声标准差；影响探索强度
        actor_hidden_dims = [512, 256, 128]  # actor MLP 隐藏层维度配置
        critic_hidden_dims = [512, 256, 128]  # critic MLP 隐藏层维度配置
        activation = 'elu' # 网络激活函数；可选 elu、relu、selu、crelu、lrelu、tanh、sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'  # 若使用循环策略，这里指定 RNN 类型
        # rnn_hidden_size = 512  # 若使用循环策略，这里指定隐藏层维度
        # rnn_num_layers = 1  # 若使用循环策略，这里指定 RNN 层数
        
    # PPO 算法超参数
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # training params
        value_loss_coef = 1.0  # value loss 权重
        use_clipped_value_loss = True  # 是否对 value loss 使用 clipping
        clip_param = 0.2  # PPO 策略裁剪阈值 epsilon
        entropy_coef = 0.01  # 熵奖励系数；越大探索越强
        num_learning_epochs = 5  # 每批采样数据重复学习的轮数
        num_mini_batches = 4 # mini-batch 数；单个 mini-batch 大小 = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 # 优化器学习率
        schedule = 'adaptive' # 学习率调度方式；可选 adaptive 或 fixed
        gamma = 0.99  # 折扣因子
        lam = 0.95  # GAE 参数 lambda
        desired_kl = 0.01  # 目标 KL；adaptive 调度时常用来调学习率
        max_grad_norm = 1.  # 梯度裁剪上限

    # 训练运行器配置
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'  # 策略类名
        algorithm_class_name = 'HIMPPO'  # 算法类名
        num_steps_per_env = 48 # 每次迭代每个环境采样的步数
        max_iterations = 100000 # 最大策略更新迭代次数

        # logging
        save_interval = 100 # 模型保存检查间隔；每这么多次迭代检查一次是否保存
        experiment_name = 'blindplane'  # 实验名称；决定日志主目录名
        run_name = ''  # 当前运行名称；会拼接到日志目录名后面
        # load and resume
        resume = False  # 是否从已有 checkpoint 恢复训练
        load_run = -1 # 要加载的 run；-1 表示自动选择最新 run
        checkpoint = -1 # 要加载的 checkpoint；-1 表示自动选择最新 checkpoint
        resume_path = None # 恢复路径；通常由 load_run 和 checkpoint 自动解析生成
