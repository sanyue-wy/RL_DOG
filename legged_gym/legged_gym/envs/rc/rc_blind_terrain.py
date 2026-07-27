# 父类文件里的导入，只在父类那个模块自己的命名空间里可见。
# 如果子类里写的新函数需要用某些模块/函数/类
# 那子类文件里也要自己导入这些模块/函数/类
from legged_gym.envs.base.legged_robot import LeggedRobot
import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

class RCBlindTerrain(LeggedRobot):
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        ##########################################################
                # 4 个负责内收/外展的 hip 关节索引
        # 如果你的关节名和这里不完全一致，就把下面 4 个名字改成你 URDF / dof_names 里的真实名字
        hip_joint_names = [
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
        ]
        self.hip_indices = torch.tensor(
            [self.dof_names.index(name) for name in hip_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        #########################################################
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        ##################delay加1仿真步（0.005s）##################
        delay_steps = torch.randint(0, self.cfg.control.decimation + 1, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length * 0.5
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    

    ################################REWARD FUNCTIONS#################################
    #腾空时间阈值改成了0.1s
    def _reward_feet_air_time(self):
        # 这个奖励项用来鼓励“脚离地后在空中保持一段时间”，
        # 也就是鼓励步态里形成更清晰的摆动相，而不是脚刚抬起来就很快又落地。
        #
        # 注意，这个奖励不是在脚腾空的每一个 step 连续发放，
        # 而是在“脚重新接触地面”的那一刻，根据这条腿累计的腾空时间一次性结算。
        #
        # self.contact_forces 的来源：
        # - 在 _init_buffers() 里从 net_contact_force_tensor 读取
        # - shape 是 [num_envs, num_bodies, 3]
        # - 这里 self.contact_forces[:, self.feet_indices, 2] 取的是每只脚在世界系 z 方向的接触力
        #
        # > 1.0 表示这只脚当前被认为“已经接触地面”。
        # 这里得到的 contact 的 shape 是 [num_envs, num_feet]，
        # 每个元素都是一个布尔值，表示某个环境里的某只脚此刻是否接触。
        #
        # 之所以要看 z 向接触力，是因为足底与地面接触时，最稳定的接触判据通常就是法向支撑力。
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        # self.last_contacts 保存“上一时刻这只脚是否接触”的布尔状态，shape 同样是 [num_envs, num_feet]。
        # 这里把当前 contact 和上一时刻 last_contacts 做逻辑或，得到 contact_filt。
        #
        # 这么做的目的不是改变物理定义，而是做一个非常轻量的接触滤波：
        # - PhysX 在 mesh 地形上，接触信号可能出现单步抖动
        # - 某只脚刚接触地面时，可能一帧 True、一帧 False 地闪
        # - 用 logical_or(contact, last_contacts) 可以让“刚接触/持续接触”更稳定一些
        #
        # 这样做会稍微保守一点，但可以减少因为接触抖动导致的假起落。
        contact_filt = torch.logical_or(contact, self.last_contacts) 

        # 把这一步未经滤波的 contact 存起来，留给下一步作为 last_contacts 使用。
        # 也就是说，last_contacts 始终记录“上一仿真步原始接触状态”。
        self.last_contacts = contact

        # self.feet_air_time 的来源：
        # - 在 _init_buffers() 里初始化为 [num_envs, num_feet] 的 0 张量
        # - 每条腿都有自己独立的腾空计时器，不是四条腿共用一个值
        #
        # (self.feet_air_time > 0.) 表示“这只脚之前已经有过离地时间累计”，
        # 换句话说，这只脚当前不是一直站在地上不动，而是确实经历过一个空中阶段。
        #
        # first_contact 的含义：
        # - 这只脚之前在空中（air_time > 0）
        # - 这一步又被判成接触（contact_filt = True）
        # 满足这两个条件时，说明“这只脚刚刚结束一次腾空，重新落地”。
        #
        # 这里 first_contact 仍然是 [num_envs, num_feet] 的逐腿布尔掩码，
        # 用来决定哪些腿应该在这一刻结算腾空时间奖励。
        first_contact = (self.feet_air_time > 0.) * contact_filt

        # 每一步先给所有脚的腾空计时器都加上一个策略步长 dt。
        # 这里的 self.dt 在你的工程里不是物理子步长 0.005，而是策略步长：
        # dt = decimation * sim_dt = 4 * 0.005 = 0.02 s
        #
        # 后面接触脚会被清零，所以最终效果是：
        # - 还在空中的脚：air_time 持续累积
        # - 已经接触的脚：本步结算后清零
        self.feet_air_time += self.dt

        # 这里正式计算腾空时间奖励。
        #
        # (self.feet_air_time - 0.1) 的含义：
        # - 0.1 s 是你当前设定的“最低期望腾空时间”
        # - 如果某条腿这次腾空时间 > 0.1 s，那么这条腿本次落地会给正值
        # - 如果某条腿这次腾空时间 < 0.1 s，那么这条腿本次落地会给负值
        #
        # 再乘上 first_contact：
        # - 只有“刚刚落地”的腿才会在这一刻结算
        # - 还在空中的腿虽然 air_time 在累积，但不会每一步都发奖励
        # - 一直站在地上的腿也不会凭空得到这项奖励
        #
        # 最后 dim=1 求和，把同一个环境里四条腿在这一时刻需要结算的值加起来，
        # 得到每个环境一个标量 reward，shape 是 [num_envs]。
        # reward only on first contact with the ground
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)

        # 如果当前速度命令几乎为 0（也就是机器人基本处于静止/站立任务），
        # 那么这项奖励直接置 0，不鼓励“为了刷腾空时间而原地抬脚”。
        #
        # torch.norm(self.commands[:, :2], dim=1) 取的是命令中的平面线速度大小：
        # - commands[:, 0] 是 x 方向速度命令
        # - commands[:, 1] 是 y 方向速度命令
        #
        # 只有当平面速度命令范数 > 0.1 时，这项奖励才生效。
        # no reward for zero command
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1

        # 这一步用来更新逐腿 air_time 计时器：
        # - 对于当前被判成接触的脚，~contact_filt = 0，因此 air_time 会被清零
        # - 对于仍然离地的脚，~contact_filt = 1，因此 air_time 会保留下来继续累积
        #
        # 所以 feet_air_time 始终表示“这只脚从最近一次离地开始，到当前为止累计了多久”。
        self.feet_air_time *= ~contact_filt

        # 返回每个环境一个标量的腾空时间奖励。
        return rew_airTime

    def _reward_foot_clearance(self):
        # 先把足端世界坐标减去机身根节点世界坐标，得到“脚相对机身原点”的位移向量。
        # self.feet_pos 来自 rigid_body_states 中每只脚刚体的世界位置，shape 是 [num_envs, num_feet, 3]。
        # self.root_states[:, 0:3] 是机身根节点在世界系中的位置，shape 是 [num_envs, 3]。
        # 这里 unsqueeze(1) 后会自动广播到四条腿，因此结果 cur_footpos_translated 仍是 [num_envs, num_feet, 3]。
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)

        # 机身在世界系中的线速度，来源是 root_states 的 7:10 切片，单位 m/s，shape 是 [num_envs, 3]。
        # 这是根节点整体平移速度，不包含机身绕自身旋转时，各个足端由于杠杆臂产生的附加线速度。
        base_lin_vel_world = self.root_states[:, 7:10]

        # 机身在世界系中的角速度，来源是 root_states 的 10:13 切片，单位 rad/s，shape 是 [num_envs, 3]。
        # 后面会用它和足端相对机身原点的位置做叉乘，算出“如果脚只是跟着机身刚体转动，本来应该有的线速度”。
        base_ang_vel_world = self.root_states[:, 10:13]

        # 这一项是机身刚体转动在足端位置上诱导出的线速度：v = w x r。
        # 含义是：即使脚相对机身完全不动，只要机身自己在转，世界系里看到的脚也会有速度。
        # 把这一项算出来并减掉，比简单只减 base linear velocity 更准确，能更真实地估计“脚相对机身到底是在上抬还是下摆”。
        rotational_vel_world = torch.cross(
            base_ang_vel_world.unsqueeze(1).expand_as(cur_footpos_translated),
            cur_footpos_translated,
            dim=2,
        )

        # 机身上对应足端位置点的刚体速度 = 机身平移速度 + 机身转动诱导速度。
        # 这个量仍在世界坐标系下，shape 为 [num_envs, num_feet, 3]。
        base_point_vel_world = base_lin_vel_world.unsqueeze(1) + rotational_vel_world

        # 足端相对机身刚体运动的线速度 = 足端世界速度 - 机身对应点的刚体速度。
        # self.feet_vel 同样来自 rigid_body_states，表示每只脚刚体在世界系的线速度。
        # 这样得到的 relative_foot_vel_world 更接近我们真正想要的“脚相对身体是在往上提还是往下放”。
        relative_foot_vel_world = self.feet_vel - base_point_vel_world

        # 下面这两个缓冲区分别保存：
        # - footpos_in_body_frame：脚相对机身原点的位置，转到机身坐标系后的结果
        # - footvel_in_body_frame：脚相对机身刚体的线速度，转到机身坐标系后的结果
        # 两者 shape 都是 [num_envs, num_feet, 3]。
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)

        # 对四条腿逐个做坐标变换，把世界系量转回机身坐标系。
        # 这样后面取 z 分量时，含义就稳定成“相对机身竖直方向的脚高 / 抬脚速度”，不会受机身俯仰滚转直接污染。
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, relative_foot_vel_world[:, i, :])

        # 机身坐标系下的 z 分量就是脚相对机身的高度。
        # 在你的坐标定义里这个值通常是负的；数值越大，表示脚越接近机身，也就是抬得越高。
        foot_height = footpos_in_body_frame[:, :, 2]

        # 机身坐标系下的 z 方向速度，表示脚相对机身是在向上抬还是向下落。
        # 正值越大，说明这只脚当前越明显地处于“往上提”的过程。
        foot_vertical_vel = footvel_in_body_frame[:, :, 2]

        # 接触力来自 self.contact_forces[:, self.feet_indices, 2]，表示每只脚在世界系 z 方向的净接触力。
        # 这里只保留 >= 0 的部分，避免接触求解噪声里的瞬时负值影响门控判断。
        contact_force_z = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)

        # airborne_gate 用来表示“这只脚现在有多像真的已经离开地面了”。
        # 当 z 向接触力接近 0 时，这个值接近 1；当接触力达到约 5N 或以上时，这个值接近 0。
        # 这部分负责覆盖真正的空中段和落脚前的下摆段，不需要再依赖 feet_air_time 才知道它在摆动。
        airborne_gate = 1.0 - torch.clamp(contact_force_z / 5.0, 0.0, 1.0)

        # unloading_gate 用来表示“这只脚是否已经明显卸载、开始准备离地”。
        # 这里把尺度放宽到约 20N，是为了让脚在完全离地前，只要已经明显减载，就能提前进入 clearance 约束。
        # 这一步正是为了解决只用 airtime 时，奖励激活过晚、后腿容易先拖一段再离地的问题。
        unloading_gate = 1.0 - torch.clamp(contact_force_z / 20.0, 0.0, 1.0)

        # upward_gate 用来表示“脚当前有没有明确的向上抬脚趋势”。
        # 低于 0.03 m/s 的小正速度视为噪声或轻微抖动，不激活；
        # 达到约 0.15 m/s 及以上时，这个门控接近 1，认为抬脚趋势很明确。
        upward_gate = torch.clamp((foot_vertical_vel - 0.03) / 0.12, 0.0, 1.0)

        # lift_gate 只有在“已经明显卸载”且“脚确实在向上抬”时才会变大。
        # 这样可以把真正的抬脚起始段提前纳入奖励，又不会让重载支撑腿因为轻微竖向波动就被误判成摆动腿。
        lift_gate = unloading_gate * upward_gate

        # swing_weight 是最终的连续摆动门控，范围在 [0, 1]。
        # - 真正空中段由 airborne_gate 覆盖
        # - 起摆早期由 lift_gate 提前点亮
        # 二者取逐元素最大值，表示“只要满足其一，就认为这只脚当前应该对净空高度负责”。
        swing_weight = torch.maximum(airborne_gate, lift_gate)

        # 为了让 min 聚合更稳定，这里再把连续门控变成一个较松的激活掩码。
        # 只有当 swing_weight 超过 0.05 时，才认为这条腿当前确实处于需要关注净空高度的阶段。
        # 这个阈值很低，主要是为了滤掉几乎为 0 的数值噪声，不是为了做硬切相。
        active_mask = swing_weight > 0.05

        # active_weight_sum 是每个环境里所有“当前需要关注净空”的腿的连续权重之和。
        # 后面做加权均值时用它归一化，比简单除以激活腿数更平滑。
        active_weight_sum = torch.sum(swing_weight, dim=1)

        # active_count 是每个环境里当前被判成活跃腿的条数。
        # 这个计数只用于控制 min 项和“当前是否应该施加 clearance 约束”的开关。
        active_count = torch.sum(active_mask, dim=1)

        # clearance_height_target 在这里被解释成“最低合格脚高”。
        # 只要脚高达到这个值，就认为 clearance 达标；再抬得更高不额外加分，也不额外扣分。
        height_target = self.cfg.rewards.clearance_height_target

        # height_sigma 控制“低于目标多少以后，分数会明显下降”。
        # 0.03 m 对应 3 cm 左右的容忍带：略低一点不会立刻很惨，明显低很多时才会快速掉分。
        height_sigma = 0.03

        # height_deficit 只计算“低于目标”的那部分缺口。
        # - 如果脚已经达到或高于目标，缺口就是 0
        # - 如果脚没抬够，缺口就是 target - current_height
        # 这样就符合你前面定下来的原则：达到目标即可，略高不吃亏，太低才给惩罚。
        height_deficit = torch.clamp(height_target - foot_height, min=0.0)

        # 用单边高斯把高度缺口映射成每条腿的高度分数，范围约在 (0, 1]：
        # - 达标时 deficit = 0，score = 1
        # - 越低于目标，score 越接近 0
        # 这里每条腿单独打分，保证后面可以明确地做逐腿约束，而不是让某两条腿漂亮就掩盖另外两条腿的问题。
        height_score = torch.exp(-torch.square(height_deficit / height_sigma))

        # active_height_score 是“只在当前应关注的腿上保留高度分数”的版本。
        # 支撑腿或几乎不活跃的腿，其 swing_weight 接近 0，因此不会对均值项产生影响。
        active_height_score = height_score * swing_weight

        # mean_score 是所有活跃腿的加权平均高度分数。
        # 这一项保证整体上所有摆动腿都应尽量抬到位，而不是只照顾某一条腿。
        mean_score = torch.sum(active_height_score, dim=1) / torch.clamp(active_weight_sum, min=1.0)

        # 对不活跃的腿，把分数临时替换成 1，再去取最小值。
        # 这样 min_score 只会盯住当前真正活跃的那些腿，不会让支撑腿以低脚高错误拉低最小值。
        masked_score_for_min = torch.where(active_mask, height_score, torch.ones_like(height_score))

        # min_score 表示“当前所有活跃腿里，表现最差的那一条腿的高度分数”。
        # 这是防止前腿或者某一条优势腿代偿其它腿的关键：哪怕平均分还行，只要最差腿太低，整体代价仍然会上来。
        min_score = torch.min(masked_score_for_min, dim=1).values

        # 最终仍然采用 mean + min 聚合：
        # - mean_score 保证整体平均水平
        # - min_score 保证最差腿不能太差
        # 这里让 mean 占 0.7、min 占 0.3，比之前稍微更强调“最差腿”，更适合你当前后腿容易掉队的问题。
        aggregated_score = 0.7 * mean_score + 0.3 * min_score

        # 最后把分数转成代价，便于继续沿用当前配置里 foot_clearance 的负权重写法。
        # 当前没有任何活跃腿时直接返回 0，避免站立或纯支撑阶段被错误施加净空高度约束。
        clearance_cost = torch.where(active_count > 0, 1.0 - aggregated_score, torch.zeros_like(aggregated_score))
        return clearance_cost

    def _reward_foot_drag(self):
        # 这个奖励项专门用来惩罚“拖地”：
        # 这里说的拖地，不是脚离地高度不够，而是脚已经和地面接触/承重了，
        # 但脚在世界坐标系的水平面内仍然有比较明显的滑动速度，
        # 看起来就会像脚贴着地面往前擦、往后拖或者横着划过去。
        #
        # 这项奖励和 _reward_foot_clearance 的职责是分开的：
        # - foot_clearance 负责管“离地腿有没有抬够”
        # - foot_drag      负责管“着地腿有没有在地上滑”
        #
        # 这样分开以后，调参时能清楚区分到底是“没抬起来”还是“落地后在拖地”。

        # self.feet_vel 的来源：
        # - 在 _init_buffers() 里从 rigid_body_states 里取出足端刚体的线速度
        # - shape 是 [num_envs, num_feet, 3]
        # - 最后一个维度分别是世界坐标系下的 vx, vy, vz
        #
        # 这里我们只取 :2，也就是世界系 x/y 平面内的速度，
        # 因为“拖地/打滑”本质上是脚相对地面的水平滑动，而不是上下运动。
        # 用世界系速度而不是机身系速度，是因为地面在世界系里近似静止，
        # 所以世界系 xy 速度最直接对应“脚在地面上有没有滑过去”。
        foot_speed_xy = torch.norm(self.feet_vel[:, :, :2], dim=2)

        # self.contact_forces 的来源：
        # - 在 _init_buffers() 里从 net_contact_force_tensor 读取
        # - shape 是 [num_envs, num_bodies, 3]
        # - 其中 self.contact_forces[:, self.feet_indices, 2] 取到的是每只脚在世界系 z 方向的接触力
        #
        # 对脚来说，z 方向的正向接触力可以近似理解为“这只脚此刻有没有压在地面上、压得有多重”。
        # 这里先 clamp 到 >= 0，只保留向上的支撑接触，
        # 这样可以减少离地边缘、接触求解误差、瞬时负值噪声带来的干扰。
        contact_force_z = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)

        # contact_weight 是一个“软接触门控”：
        # - shape 也是 [num_envs, num_feet]
        # - 数值范围被压到 [0, 1]
        #
        # 为什么不用硬阈值判断 contact=True/False，而要用这个连续权重：
        # 1. 真实接触不是完全干净的 0/1 切换，边界时会有抖动
        # 2. 有些脚只是轻微擦碰地面，不应该和完全承重的支撑腿吃同样强的惩罚
        # 3. 连续门控对 PPO 一般更平滑，训练更稳定
        #
        # 这里 /5.0 的含义：
        # - 当 z 向接触力很小的时候，只给一部分权重
        # - 当 z 向接触力达到约 5N 或更大时，权重饱和到 1
        # - 也就是：轻擦地轻罚，真正踩住地面后如果还在滑就重罚
        contact_weight = torch.clamp(contact_force_z / 5.0, 0.0, 1.0)

        # slip_deadzone 是“速度死区”，单位是 m/s。
        # 它的作用是允许脚在接触地面时存在一点点很小的水平速度，
        # 不把数值抖动、接触求解器的微滑、以及很轻微的正常滚动/擦碰都算成拖地。
        #
        # 如果没有这个死区，哪怕脚在地面上几乎是稳定的，只要有一点点速度噪声也会持续受罚，
        # 容易让 reward 过于敏感，训练出来的步态发紧或者不自然。
        slip_deadzone = 0.05

        # slip_excess 表示“超过死区之后，真正需要惩罚的那部分滑动速度”：
        # - 如果脚的水平速度 <= 0.05 m/s，则认为几乎没在拖地，记为 0
        # - 如果明显大于这个阈值，则超出的部分越大，后面的惩罚越强
        slip_excess = torch.clamp(foot_speed_xy - slip_deadzone, min=0.0)

        # per_foot_drag 是逐脚的拖地代价，shape 为 [num_envs, num_feet]。
        #
        # 公式拆开看：
        # - torch.square(slip_excess)：对超出死区的滑动速度做平方
        #   这样小滑动轻罚，大滑动重罚，比线性更能压制“明显拖地”
        # - contact_weight * ... ：只有在脚真正接触地面时，这个滑动才算拖地
        #   离地腿即使速度很快，也不应该被这一项惩罚
        per_foot_drag = contact_weight * torch.square(slip_excess)

        # contact_weight_sum 是每个环境里“有效接触权重”的总和，shape 为 [num_envs]。
        #
        # 这里不用固定除以 4 条腿，也不用简单除以接触腿数量，
        # 而是除以连续的接触权重和，原因是：
        # - 有的相位只有 2 条腿明显承重
        # - 有的相位可能有 3 条腿部分接触
        # - 用连续权重做归一化，量级会比硬计数更平滑
        contact_weight_sum = torch.sum(contact_weight, dim=1)

        # mean_drag 是“当前所有接触脚的平均拖地程度”，shape 为 [num_envs]。
        # 这一项负责约束整体：希望所有着地脚平均来看都别在地上滑。
        #
        # torch.clamp(..., min=1.0) 是为了避免分母为 0；
        # 真正没有接触脚时，下面 return 里会把整项直接置 0。
        mean_drag = torch.sum(per_foot_drag, dim=1) / torch.clamp(contact_weight_sum, min=1.0)

        # max_drag 取的是“最严重那只拖地脚”的代价，shape 为 [num_envs]。
        # 只看均值会有一个问题：如果 3 条腿都很好，1 条腿严重拖地，
        # 那均值可能 still 不大，看起来像问题不严重。
        # 所以这里额外加一个 max 项，专门盯住“最坏的那条腿”，避免坏腿被平均掉。
        max_drag = torch.max(per_foot_drag, dim=1).values

        # 最终 drag_cost 把整体和最坏腿结合起来：
        # - 0.7 * mean_drag：主要约束整体接触脚的平均拖地水平
        # - 0.3 * max_drag ：补充约束最差那条腿，防止某一条腿长期明显拖地
        #
        # 这个组合和 foot_clearance 里用 mean + min 的思路类似：
        # 既不只看全局平均，也不只盯单条腿，而是在两者之间做折中。
        drag_cost = 0.7 * mean_drag + 0.3 * max_drag

        # 如果当前没有任何脚在有效接触地面（contact_weight_sum == 0），
        # 那么“拖地”这个概念本身就不成立，直接返回 0。
        # 这样可以避免离地阶段被错误地施加拖地惩罚。
        return torch.where(contact_weight_sum > 0.0, drag_cost, torch.zeros_like(drag_cost))
    
    def _reward_hip_abduction_deviation(self):
        # 4 个 hip 相对默认角度的偏差
        err = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]
        # 容忍正负 0.05 rad，超过这个范围才开始惩罚
        excess = torch.clamp(torch.abs(err) - 0.05, min=0.0)
        # 基础代价：偏差平方的均值
        penalty = torch.mean(excess ** 2, dim=1)
        # 侧移和转向命令越大，惩罚越弱，避免和 lateral motion / yaw motion 直接冲突
        lateral_cmd = torch.abs(self.commands[:, 1])
        yaw_cmd = torch.abs(self.commands[:, 2])
        # 当 |command_y| 从 0 增加到 0.5 时，侧移放松强度从 0 增加到 1
        lateral_relax = torch.clamp(lateral_cmd / 0.5, 0.0, 1.0)
        # 当 |command_yaw| 从 0 增加到 1.0 rad/s 时，转向放松强度从 0 增加到 1
        yaw_relax = torch.clamp(yaw_cmd / 1.0, 0.0, 1.0)
        # 两者任一较大时都应明显减弱惩罚，取 max 保持原有放松幅度上限
        relax_strength = torch.maximum(lateral_relax, yaw_relax)
        # 放松系数最小保留到 0.2，避免该项奖励完全失效
        relax = 1.0 - 0.8 * relax_strength
        return 10 * penalty * relax
    
    def _reward_similar_legged(self):
        # 鼓励前右腿-后左腿、前左腿-后右腿动作一致，避免出现三条腿运动同时一条腿腾空
        legged_error_fr_rl = torch.sum(torch.square(self.dof_pos[:,0:3] - self.dof_pos[:,9:12]), dim=1)
        legged_error_fl_rr = torch.sum(torch.square(self.dof_pos[:,3:6] - self.dof_pos[:,6:9]), dim=1)
        legged_error = legged_error_fl_rr + legged_error_fr_rl
        similar_legged_sigma = 0.1
        return torch.exp(-legged_error / similar_legged_sigma)
