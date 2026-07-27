# 父类文件里的导入，只在父类那个模块自己的命名空间里可见。
# 如果子类里写的新函数需要用某些模块/函数/类
# 那子类文件里也要自己导入这些模块/函数/类
from legged_gym.envs.base.legged_robot import LeggedRobot
import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

class RCBlindPlane(LeggedRobot):
    def _parse_cfg(self, cfg):
        self.height_command_enabled = getattr(cfg.commands, "height_command", True)
        if self.height_command_enabled:
            cfg.commands.num_commands = 5
            cfg.env.num_one_step_observations = 46
            cfg.env.num_one_step_privileged_obs = 46 + 3 + 3 + 187
        else:
            cfg.commands.num_commands = 4
            cfg.env.num_one_step_observations = 45
            cfg.env.num_one_step_privileged_obs = 45 + 3 + 3 + 187
        cfg.env.num_observations = cfg.env.num_one_step_observations * 6
        cfg.env.num_privileged_obs = cfg.env.num_one_step_privileged_obs * 1
        super()._parse_cfg(cfg)

    def _get_noise_scale_vec(self, cfg):
        """平地高度控制版本的噪声配置。"""
        num_command_obs = 4 if self.height_command_enabled else 3
        base_obs_dim = 6 + num_command_obs + 3 * self.num_actions
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(base_obs_dim + 187, device=self.device)
        else:
            noise_vec = torch.zeros(base_obs_dim, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:num_command_obs] = 0.  # commands
        noise_vec[num_command_obs:(num_command_obs + 3)] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[(num_command_obs + 3):(num_command_obs + 6)] = noise_scales.gravity * noise_level
        dof_pos_start = num_command_obs + 6
        dof_vel_start = dof_pos_start + self.num_actions
        action_start = dof_vel_start + self.num_actions
        noise_vec[dof_pos_start:dof_vel_start] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[dof_vel_start:action_start] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[action_start:(action_start + self.num_actions)] = 0.
        if self.cfg.terrain.measure_heights:
            noise_vec[(action_start + self.num_actions):(action_start + self.num_actions + 187)] = (
                noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            )
        return noise_vec

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
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading, height
        if self.height_command_enabled:
            self.commands_scale = torch.tensor(
                [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, self.obs_scales.height],
                device=self.device,
                requires_grad=False,
            )
        else:
            self.commands_scale = torch.tensor(
                [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                device=self.device,
                requires_grad=False,
            )
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
        thigh_joint_names = [
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
        ]
        self.thigh_indices = torch.tensor(
            [self.dof_names.index(name) for name in thigh_joint_names],
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

    def compute_observations(self):
        """平地高度控制版本观测：actor 前 46 维包含 3 维速度指令 + 1 维高度命令。"""
        if self.height_command_enabled:
            command_obs = torch.cat((self.commands[:, :3], self.commands[:, 4:5]), dim=1) * self.commands_scale
        else:
            command_obs = self.commands[:, :3] * self.commands_scale
        current_obs = torch.cat((
            command_obs,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        proprio_obs_dim = current_obs.shape[1]
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            heights_noise_start = proprio_obs_dim
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[heights_noise_start:(heights_noise_start + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def get_current_obs(self):
        """返回与当前 actor/critic 观测结构一致的单帧观测。"""
        if self.height_command_enabled:
            command_obs = torch.cat((self.commands[:, :3], self.commands[:, 4:5]), dim=1) * self.commands_scale
        else:
            command_obs = self.commands[:, :3] * self.commands_scale
        current_obs = torch.cat((
            command_obs,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        proprio_obs_dim = current_obs.shape[1]
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            heights_noise_start = proprio_obs_dim
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[heights_noise_start:(heights_noise_start + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return current_obs

    def compute_termination_observations(self, env_ids):
        """平地高度控制版本 termination obs。"""
        if self.height_command_enabled:
            command_obs = torch.cat((self.commands[:, :3], self.commands[:, 4:5]), dim=1) * self.commands_scale
        else:
            command_obs = self.commands[:, :3] * self.commands_scale
        current_obs = torch.cat((
            command_obs,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:current_obs.shape[1]]

        proprio_obs_dim = current_obs.shape[1]
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            heights_noise_start = proprio_obs_dim
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[heights_noise_start:(heights_noise_start + 187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]

    def _resample_commands(self, env_ids):
        """平地高度控制版本：重采样 x/y/yaw/heading/height 5 维命令。"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        if self.height_command_enabled:
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device
            ).squeeze(1)

        high_vel_env_ids = env_ids[(env_ids < (self.num_envs * 0.2)).nonzero(as_tuple=True)]
        self.commands[high_vel_env_ids, 1] *= (torch.abs(self.commands[high_vel_env_ids, 0]) < 1.0)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


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

    ################################REWARD FUNCTIONS#################################
    #腾空时间阈值改成了0.1s
    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        base_lin_vel_world = self.root_states[:, 7:10]
        base_ang_vel_world = self.root_states[:, 10:13]
        rotational_vel_world = torch.cross(
            base_ang_vel_world.unsqueeze(1).expand_as(cur_footpos_translated),
            cur_footpos_translated,
            dim=2,
        )
        base_point_vel_world = base_lin_vel_world.unsqueeze(1) + rotational_vel_world
        relative_foot_vel_world = self.feet_vel - base_point_vel_world
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, relative_foot_vel_world[:, i, :])
        foot_height = footpos_in_body_frame[:, :, 2]
        foot_vertical_vel = footvel_in_body_frame[:, :, 2]
        contact_force_z = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)
        airborne_gate = 1.0 - torch.clamp(contact_force_z / 5.0, 0.0, 1.0)
        unloading_gate = 1.0 - torch.clamp(contact_force_z / 20.0, 0.0, 1.0)
        upward_gate = torch.clamp((foot_vertical_vel - 0.03) / 0.12, 0.0, 1.0)
        lift_gate = unloading_gate * upward_gate
        swing_weight = torch.maximum(airborne_gate, lift_gate)
        active_mask = swing_weight > 0.05
        active_weight_sum = torch.sum(swing_weight, dim=1)
        active_count = torch.sum(active_mask, dim=1)
        height_target = self.cfg.rewards.clearance_height_target
        height_sigma = 0.03
        height_deficit = torch.clamp(height_target - foot_height, min=0.0)
        height_score = torch.exp(-torch.square(height_deficit / height_sigma))
        active_height_score = height_score * swing_weight
        mean_score = torch.sum(active_height_score, dim=1) / torch.clamp(active_weight_sum, min=1.0)
        masked_score_for_min = torch.where(active_mask, height_score, torch.ones_like(height_score))
        min_score = torch.min(masked_score_for_min, dim=1).values
        aggregated_score = 0.7 * mean_score + 0.3 * min_score
        clearance_cost = torch.where(active_count > 0, 1.0 - aggregated_score, torch.zeros_like(aggregated_score))
        base_height = self._get_base_heights()
        low_height_ratio = torch.clamp((0.2 - base_height) / 0.1, 0.0, 1.0)
        # Height >= 0.2m keeps full weight, 0.1m linearly decays to 40%, and below 0.1m stays at 40%.
        height_scale = 1.0 - 0.6 * low_height_ratio
        return clearance_cost * height_scale

    def _reward_foot_drag(self):
        foot_speed_xy = torch.norm(self.feet_vel[:, :, :2], dim=2)
        contact_force_z = torch.clamp(self.contact_forces[:, self.feet_indices, 2], min=0.0)
        contact_weight = torch.clamp(contact_force_z / 5.0, 0.0, 1.0)
        slip_deadzone = 0.05
        slip_excess = torch.clamp(foot_speed_xy - slip_deadzone, min=0.0)
        per_foot_drag = contact_weight * torch.square(slip_excess)
        contact_weight_sum = torch.sum(contact_weight, dim=1)
        mean_drag = torch.sum(per_foot_drag, dim=1) / torch.clamp(contact_weight_sum, min=1.0)
        max_drag = torch.max(per_foot_drag, dim=1).values
        drag_cost = 0.7 * mean_drag + 0.3 * max_drag
        return torch.where(contact_weight_sum > 0.0, drag_cost, torch.zeros_like(drag_cost))
    
    def _reward_hip_abduction_deviation(self):
        err = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]
        # 容忍正负 0.05 rad，超过这个范围才开始惩罚
        excess = torch.clamp(torch.abs(err) - 0.05, min=0.0)
        penalty = torch.mean(excess ** 2, dim=1)
        lateral_cmd = torch.abs(self.commands[:, 1])
        yaw_cmd = torch.abs(self.commands[:, 2])
        lateral_relax = torch.clamp(lateral_cmd / 0.5, 0.0, 1.0)
        yaw_relax = torch.clamp(yaw_cmd / 1.0, 0.0, 1.0)
        relax_strength = torch.maximum(lateral_relax, yaw_relax)
        relax = 1.0 - 0.8 * relax_strength
        # 当机身高度低于 0.2m 后，逐步放松这一项惩罚，允许机器人为了降高度而适度张开髋关节。
        base_height = self._get_base_heights()
        # 在 [0.2m, 0.1m] 区间内先快速放松，接近最低高度时再逐渐放缓。
        low_height_relax = torch.clamp((0.2 - base_height) / 0.1, 0.0, 1.0)
        # 最低保留到 0.4，避免过度放松导致机器人张开髋关节太多。
        height_relax = 0.3 + 0.7 * torch.square(1.0 - low_height_relax)
        relax = relax * height_relax
        return 10 * penalty * relax

    def _reward_base_height(self):
        """高度命令开启时跟随 commands[:, 4]，关闭时退回固定目标高度。"""
        base_height = self._get_base_heights()
        if self.height_command_enabled:
            return 100 * torch.square(base_height - self.commands[:, 4])
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height_encourage(self):
        base_height = self._get_base_heights()
        if self.height_command_enabled:
            target_height = self.commands[:, 4]
        else:
            target_height = self.cfg.rewards.base_height_target
        base_height_error = torch.square(base_height - target_height)
        # 用更陡的高斯型奖励，让高度越接近目标时奖励越明显，偏差拉开后奖励更快下降。
        return torch.exp(-80.0 * base_height_error)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        tracking_reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        base_height = self._get_base_heights()
        low_height_relax = torch.clamp((0.2 - base_height) / 0.1, 0.0, 1.0)
        # 高度从 0.2m 降到 0.1m 时，线速度跟踪奖励从 100% 线性衰减到 30%。
        height_scale = 1.0 - 0.7 * low_height_relax
        return tracking_reward * height_scale

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tracking_reward = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        base_height = self._get_base_heights()
        low_height_relax = torch.clamp((0.2 - base_height) / 0.1, 0.0, 1.0)
        # 高度从 0.2m 降到 0.1m 时，角速度跟踪奖励从 100% 线性衰减到 30%。
        height_scale = 1.0 - 0.7 * low_height_relax
        return tracking_reward * height_scale
    
    def _reward_orientation_pitch(self):
        # Penalize non flat base orientation pitch
        return torch.square(self.projected_gravity[:, 1])

    def _reward_similar_legged(self):
        # 鼓励前右腿-后左腿、前左腿-后右腿动作一致，避免出现三条腿运动同时一条腿腾空
        legged_error_fr_rl = torch.sum(torch.square(self.dof_pos[:,0:3] - self.dof_pos[:,9:12]), dim=1)
        legged_error_fl_rr = torch.sum(torch.square(self.dof_pos[:,3:6] - self.dof_pos[:,6:9]), dim=1)
        legged_error = legged_error_fl_rr + legged_error_fr_rl
        similar_legged_sigma = 0.1
        return torch.exp(-legged_error / similar_legged_sigma)
    
    def _reward_vel_y_zero_penalize(self):
        # 单轴平移命令时，惩罚另一条平移轴上的偏移速度，减少斜着走。
        command_threshold = 0.08
        cmd_x = torch.abs(self.commands[:, 0])
        cmd_y = torch.abs(self.commands[:, 1])

        x_only_cmd = (cmd_x > command_threshold) & (cmd_y < command_threshold)
        y_only_cmd = (cmd_y > command_threshold) & (cmd_x < command_threshold)

        penalize_y_vel_when_x_only = torch.square(self.base_lin_vel[:, 1]) * x_only_cmd.float()
        penalize_x_vel_when_y_only = torch.square(self.base_lin_vel[:, 0]) * y_only_cmd.float()
        return penalize_y_vel_when_x_only + penalize_x_vel_when_y_only

    def _reward_stand_still(self):
        # 仅在机身高度大于 0.2m 时，对零速度命令下偏离默认站姿的行为进行惩罚。
        stand_still_cost = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        zero_motion_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        high_body_gate = (self._get_base_heights() > 0.2).float()
        return stand_still_cost * zero_motion_cmd * high_body_gate

    def _reward_collision(self):
        # Penalize collisions on selected bodies.
        collision_cost = torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )
        base_height = self._get_base_heights()
        low_height_relax = torch.clamp((0.2 - base_height) / 0.1, 0.0, 1.0)
        # 高度从 0.2m 降到 0.1m 时，碰撞惩罚先快速下降，接近最低高度时再逐渐放缓到 10%。
        height_scale = 0.1 + 0.9 * torch.square(1.0 - low_height_relax)
        return collision_cost * height_scale

    def _reward_low_height_thigh_horizontal(self):
        base_height = self._get_base_heights()
        # 在 [0.20m, 0.18m] 区间内平滑激活；低于 0.18m 后完全激活，避免早期训练时长期拿不到信号。
        active_weight = torch.clamp((0.20 - base_height) / 0.04, 0.0, 1.0)
        thigh_target = 1.55
        thigh_error = torch.mean(torch.square(self.dof_pos[:, self.thigh_indices] - thigh_target), dim=1)
        # 用更宽的高斯型奖励，让训练早期即使偏差较大也能得到可见奖励。
        thigh_sigma = 0.5
        thigh_reward = torch.exp(-thigh_error / (thigh_sigma ** 2))
        return thigh_reward * active_weight
