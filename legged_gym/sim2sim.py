import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import argparse
import pygame
import copy

import math
from pathlib import Path


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_body_id(model, body_name="base"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")
    return body_id


def get_base_rotmat(data, body_name="base"):
    body_id = get_body_id(data.model, body_name)
    return data.xmat[body_id].reshape(3, 3).copy()


def get_base_yaw(data, body_name="base"):
    rot = get_base_rotmat(data, body_name)
    return float(math.atan2(rot[1, 0], rot[0, 0]))


def get_projected_gravity(data, body_name="base"):
    # 训练里 projected_gravity 的语义是：世界重力向量投到机体系
    rot = get_base_rotmat(data, body_name)  # body -> world
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    gravity_body = rot.T @ gravity_world
    return gravity_body.astype(np.float32)

# ======================== 手柄配置 ========================
# 轴映射（根据你实际测试结果调整编号）
AXIS_VX   = 1    # 左摇杆上下
AXIS_VY   = 0    # 左摇杆左右
AXIS_YAW  = 2    # 右摇杆左右，直接作为偏航角速度指令

# 速度范围限制
VX_MAX   = 1.0     # 最大前进速度 m/s
VY_MAX   = 1.0    # 最大横移速度 m/s
YAW_MAX  = 2.0    # 最大偏航角速度指令 rad/s

# 死区（手柄摇杆归零时可能不是精确的0）
DEADZONE = 0.1


def init_gamepad():
    """初始化手柄"""
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("⚠️  未检测到手柄，将启用键盘控制")
        return None

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"✅ 手柄已连接: {js.get_name()}")
    print(f"   轴数: {js.get_numaxes()}, 按钮数: {js.get_numbuttons()}")
    return js


def apply_deadzone(value: float, deadzone: float) -> float:
    """应用死区"""
    if abs(value) < deadzone:
        return 0.0
    # 死区外线性映射到 [0, 1]
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def get_gamepad_cmd(js) -> np.ndarray:
    """从手柄读取线速度指令和偏航角速度指令"""
    pygame.event.pump()  # 必须调用，刷新手柄状态

    # 读取摇杆值（范围 -1 ~ 1）
    raw_vx  = -js.get_axis(AXIS_VX)   # 取负，因为摇杆向上是负值
    raw_vy  = -js.get_axis(AXIS_VY)   # 取负，因为摇杆向左是负值（根据实际调整）
    raw_yaw = -js.get_axis(AXIS_YAW)  # 取负，根据实际调整

    # 应用死区
    vx  = apply_deadzone(raw_vx,  DEADZONE) * VX_MAX
    vy  = apply_deadzone(raw_vy,  DEADZONE) * VY_MAX
    yaw = apply_deadzone(raw_yaw, DEADZONE) * YAW_MAX

    return np.array([vx, vy, yaw], dtype=np.float32)


def init_keyboard():
    """初始化键盘输入窗口。"""
    pygame.display.init()
    screen = pygame.display.set_mode((420, 120))
    pygame.display.set_caption("sim2sim Keyboard Control")
    screen.fill((30, 30, 30))
    pygame.display.flip()
    return screen


# ======================== 配置常量（跳跃机依赖） ========================
DEFAULT_HEIGHT_CMD = 0.26  # 默认站立高度目标（米）
# 注意：根据训练配置选择正确的COMMANDS_SCALE
# - 有高度命令(height_command=True): [2.0, 2.0, 0.25, 4.0]
# - 无高度命令(height_command=False): [2.0, 2.0, 0.25]
COMMANDS_SCALE = np.array([2.0, 2.0, 0.25, 4.0], dtype=np.float32)


def get_keyboard_cmd() -> np.ndarray:
    """从键盘读取线速度、偏航角速度指令（3维）。"""
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    vx = 0.0
    vy = 0.0
    yaw = 0.0

    if keys[pygame.K_w]:
        vx += VX_MAX
    if keys[pygame.K_s]:
        vx -= VX_MAX
    if keys[pygame.K_a]:
        vy += VY_MAX
    if keys[pygame.K_d]:
        vy -= VY_MAX
    if keys[pygame.K_q]:
        yaw += YAW_MAX
    if keys[pygame.K_e]:
        yaw -= YAW_MAX

    return np.array([vx, vy, yaw], dtype=np.float32)


# ======================== 跳跃状态机 ========================
class JumpController:
    """
    按住 Z 蹲下蓄力，松开 Z 起跳。用于上楼梯/跨越障碍。

    流程：
      按住 Z  → 蹲伏蓄力（高度 0.12m，可一直按住）
      松开 Z  → 蹬起 → 腾空 → 落地 → 回到正常

    蹬起/腾空/落地 序列（按控制步计数，每步 0.02s）:
      蹬起  8步 (0.16s)  → 高度 0.40m，向前冲
      腾空 10步 (0.20s)  → 高度 0.38m，向前冲
      落地 15步 (0.30s)  → 高度渐回 0.26m
    """
    PHASE_IDLE    = 0
    PHASE_CROUCH  = 1   # 按住 Z 蓄力
    PHASE_LAUNCH  = 2   # 松开 Z 蹬起
    PHASE_AIR     = 3   # 腾空
    PHASE_LAND    = 4   # 落地过渡

    # 蹬起→腾空→落地 的时长（控制步数）
    LAUNCH_STEPS = 8
    AIR_STEPS    = 10
    LAND_STEPS   = 15

    CROUCH_HEIGHT  = 0.12   # 蹲伏高度
    LAUNCH_HEIGHT  = 0.40   # 蹬起高度
    AIR_HEIGHT     = 0.38   # 腾空高度
    LAUNCH_VX      = 1.5    # 蹬起前进速度
    AIR_VX         = 1.2    # 腾空前进速度
    LAND_VX        = 0.5    # 落地前进速度

    def __init__(self, has_height_cmd=True):
        self.phase = self.PHASE_IDLE
        self.step_counter = 0
        self.land_start_height = DEFAULT_HEIGHT_CMD
        self.has_height_cmd = has_height_cmd
        self.cmd_dim = 4 if has_height_cmd else 3

    def update(self, cmd_in: np.ndarray) -> np.ndarray:
        """
        每个控制步调用一次。输入原始 cmd [vx, vy, yaw]，
        输出修改后的 cmd [vx, vy, yaw] 或 [vx, vy, yaw, height_cmd]。
        """
        keys = pygame.key.get_pressed()
        z_held = keys[pygame.K_z]

        cmd = np.zeros(self.cmd_dim, dtype=np.float32)
        cmd[0] = cmd_in[0]
        cmd[1] = cmd_in[1]
        cmd[2] = cmd_in[2]

        # ── IDLE：等待按 Z ──
        if self.phase == self.PHASE_IDLE:
            if z_held:
                self.phase = self.PHASE_CROUCH
                self.step_counter = 0
            if self.has_height_cmd:
                cmd[3] = DEFAULT_HEIGHT_CMD
            return cmd

        # ── CROUCH：按住 Z 蹲伏蓄力 ──
        if self.phase == self.PHASE_CROUCH:
            if self.has_height_cmd:
                cmd[3] = self.CROUCH_HEIGHT
            # 前进速度降低，准备起跳
            cmd[0] = cmd_in[0] * 0.5
            if not z_held:
                # 松开 Z → 起跳
                self.phase = self.PHASE_LAUNCH
                self.step_counter = 0
            return cmd

        # ── LAUNCH / AIR / LAND：自动序列 ──
        self.step_counter += 1

        if self.phase == self.PHASE_LAUNCH:
            if self.has_height_cmd:
                cmd[3] = self.LAUNCH_HEIGHT
            cmd[0] = max(cmd_in[0], self.LAUNCH_VX)
            if self.step_counter >= self.LAUNCH_STEPS:
                self.land_start_height = self.LAUNCH_HEIGHT
                self.phase = self.PHASE_AIR
                self.step_counter = 0

        elif self.phase == self.PHASE_AIR:
            if self.has_height_cmd:
                cmd[3] = self.AIR_HEIGHT
            cmd[0] = max(cmd_in[0], self.AIR_VX)
            if self.step_counter >= self.AIR_STEPS:
                self.land_start_height = self.AIR_HEIGHT
                self.phase = self.PHASE_LAND
                self.step_counter = 0

        elif self.phase == self.PHASE_LAND:
            t = min(self.step_counter / self.LAND_STEPS, 1.0)
            if self.has_height_cmd:
                cmd[3] = self.land_start_height + (DEFAULT_HEIGHT_CMD - self.land_start_height) * t
            cmd[0] = max(cmd_in[0], self.LAND_VX * (1.0 - t))
            if self.step_counter >= self.LAND_STEPS:
                self.phase = self.PHASE_IDLE
                self.step_counter = 0

        return cmd

# ======================== 配置 ========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "resources", "robots", "RCV3", "xml", "scene.xml")


def find_latest_model():
    """自动查找最新的模型文件（按修改时间）"""
    logs_dir = Path(_SCRIPT_DIR) / "logs"
    all_models = []

    for model_file in logs_dir.rglob("model_*.pt"):
        try:
            iteration = int(model_file.stem.split('_')[1])
            all_models.append((model_file, iteration, model_file.stat().st_mtime))
        except (ValueError, IndexError):
            continue

    if not all_models:
        return None

    # 按修改时间排序，返回最新的
    all_models.sort(key=lambda x: x[2], reverse=True)
    return str(all_models[0][0])

DT = 0.02           # 控制频率 50Hz
DECIMATION = 4       # 与训练一致
SIM_DT = DT / DECIMATION  # 仿真步长 0.005s

NUM_ACTIONS = 12
# 注意：根据训练配置选择正确的值
# - 有高度命令(height_command=True): 46
# - 无高度命令(height_command=False): 45
# 当前模型训练时使用46维观测（有高度命令）
NUM_ONE_STEP_OBS = 46   # 与训练配置一致
OBS_HISTORY_LEN = 6
NUM_OBS = NUM_ONE_STEP_OBS * OBS_HISTORY_LEN  # 276

CLIP_OBS = 100.0
CLIP_ACTIONS = 100.0
ACTION_SCALE = 0.25

# 观测缩放（根据你训练配置填写）
OBS_SCALES = {
    'lin_vel':    2.0,
    'ang_vel':    0.25,
    'gravity':    1.0,
    'dof_pos':    1.0,
    'dof_vel':    0.05,
}


# Go2 默认站立关节角度（弧度），顺序与 MJCF 一致
# FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
# RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
DEFAULT_POS = np.array([
     0.1,   0.8,  -1.5,
    -0.1,   0.8,  -1.5,
     0.1,   1.0,  -1.5,
    -0.1,   1.0,  -1.5,
], dtype=np.float32)

# PD 增益：与当前训练配置保持一致
KP = np.full(NUM_ACTIONS, 35.0, dtype=np.float32)
KD = np.full(NUM_ACTIONS, 0.8, dtype=np.float32)

# 真机单关节最大力矩；与 MuJoCo XML 中的 actuator / joint 限幅保持一致
TORQUE_LIMIT = 23.7

# 原始命令输入 [vx, vy, yaw_rate, height_cmd]
# 注意：有高度命令时使用4维
CMD = np.array([0.0, 0.0, 0.0, DEFAULT_HEIGHT_CMD], dtype=np.float32)


# ======================== 工具函数 ========================
def get_obs(data, cmd, last_action, commands_scale=None):
    if commands_scale is None:
        commands_scale = COMMANDS_SCALE
    q = data.qpos.astype(np.float32).copy()
    dq = data.qvel.astype(np.float32).copy()

    gyro = data.sensor('imu_gyro').data.copy().astype(np.float32)
    projected_gravity = get_projected_gravity(data, body_name="base")

    dof_pos = q[7:19]
    dof_vel = dq[6:18]

    obs = np.concatenate([
        cmd.astype(np.float32) * commands_scale,
        gyro * OBS_SCALES["ang_vel"],
        projected_gravity,
        (dof_pos - DEFAULT_POS).astype(np.float32) * OBS_SCALES["dof_pos"],
        dof_vel.astype(np.float32) * OBS_SCALES["dof_vel"],
        last_action.astype(np.float32),
    ]).astype(np.float32)

    return obs


def build_obs_history(obs_history: np.ndarray, new_obs: np.ndarray, num_one_step_obs: int = None) -> np.ndarray:
    """
    维护观测历史，最新帧在前（与训练一致）：
    obs_buf = [t, t-1, t-2, t-3, t-4, t-5]
    """
    if num_one_step_obs is None:
        num_one_step_obs = NUM_ONE_STEP_OBS
    obs_history = np.roll(obs_history, num_one_step_obs)
    obs_history[:num_one_step_obs] = new_obs
    return obs_history



# ======================== 策略加载 ========================
class PolicyExporterHIM(nn.Module):
    """从 checkpoint 加载 actor + estimator.encoder 的推理模块。"""
    def __init__(self, actor, estimator_encoder, num_one_step_obs=46):
        super().__init__()
        self.actor = actor
        self.estimator = estimator_encoder
        self.num_one_step_obs = num_one_step_obs

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, 0:self.num_one_step_obs], vel, z), dim=1))


def load_policy(path, device):
    """加载训练 checkpoint，返回 (policy模块, 模型配置dict)。
    自动从权重维度推断观测配置。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt['model_state_dict']

    def build_sequential(prefix):
        """从 state_dict 按 key 顺序重建 nn.Sequential（跳号 key，如 0,2,4,6）。"""
        pfx = prefix + '.'
        indices = sorted({int(k[len(pfx):].split('.')[0]) for k in sd if k.startswith(pfx) and k.endswith('.weight')})
        layers = []
        for idx, k_idx in enumerate(indices):
            w = sd[f'{prefix}.{k_idx}.weight']
            b = sd[f'{prefix}.{k_idx}.bias']
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            layers[-1].weight.data.copy_(w)
            layers[-1].bias.data.copy_(b)
            if idx < len(indices) - 1:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    actor = build_sequential('actor')
    estimator_encoder = build_sequential('estimator.encoder')

    # 从 estimator 第一层权重推断观测维度
    est_input_dim = estimator_encoder[0].in_features  # e.g. 276 or 270
    history_length = 6  # 固定历史长度
    num_one_step_obs = est_input_dim // history_length  # 46 or 45
    has_height_cmd = (num_one_step_obs == 46)

    model_config = {
        'est_input_dim': est_input_dim,
        'num_one_step_obs': num_one_step_obs,
        'num_obs': est_input_dim,
        'history_length': history_length,
        'has_height_cmd': has_height_cmd,
        'commands_scale': np.array([2.0, 2.0, 0.25, 4.0] if has_height_cmd else [2.0, 2.0, 0.25], dtype=np.float32),
        'cmd_dim': 4 if has_height_cmd else 3,
    }

    policy = PolicyExporterHIM(actor, estimator_encoder, num_one_step_obs)
    policy.to(device)
    policy.eval()
    return policy, model_config


# ======================== 主循环 ========================
def main():
    print("=" * 60)
    print("🚀 Sim2Sim MuJoCo 验证程序启动")
    print("=" * 60)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Sim2Sim MuJoCo验证')
    parser.add_argument('--policy', type=str, default=None,
                       help='策略文件路径（不指定则自动查找最新）')
    args = parser.parse_args()

    # 确定策略路径
    if args.policy:
        policy_path = args.policy
        if not Path(policy_path).exists():
            print(f"❌ 策略文件不存在: {policy_path}")
            sys.exit(1)
    else:
        policy_path = find_latest_model()
        if not policy_path:
            print("❌ 没有找到任何策略文件")
            sys.exit(1)
        print(f"🔍 自动选择最新策略: {policy_path}")

    print(f"🎯 使用策略: {policy_path}")

    # 初始化手柄
    print("🎮 初始化输入设备...")
    gamepad = init_gamepad()
    keyboard_screen = None
    if gamepad is None:
        keyboard_screen = init_keyboard()
        print("⌨️  键盘控制窗口已创建")

    # 加载MuJoCo模型
    print(f"📦 加载MuJoCo模型: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    print("✅ MuJoCo模型加载成功")

    # 加载策略（自动检测模型配置）
    print(f"🧠 加载策略: {policy_path}")
    device = torch.device('cpu')
    policy, model_config = load_policy(policy_path, device)
    policy.eval()

    # 从模型配置中获取参数
    num_one_step_obs = model_config['num_one_step_obs']
    num_obs = model_config['num_obs']
    commands_scale = model_config['commands_scale']
    has_height_cmd = model_config['has_height_cmd']

    print(f"✅ 策略加载成功")
    print(f"   观测维度: {num_one_step_obs}/step, {num_obs} 总计")
    print(f"   高度命令: {'是' if has_height_cmd else '否'}")
    print(f"   命令缩放: {commands_scale}")

    # 初始化状态
    mujoco.mj_resetData(model, data)
    data.qpos[7:] = DEFAULT_POS.copy()
    # 与当前 floating-base XML 中的站立初始高度一致，避免一开始把机身压得过低
    data.qpos[2] = 0.52
    mujoco.mj_forward(model, data)

    # 初始化
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    obs_history = np.zeros(num_obs, dtype=np.float32)
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if has_height_cmd:
        cmd = np.array([0.0, 0.0, 0.0, DEFAULT_HEIGHT_CMD], dtype=np.float32)
    else:
        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    init_obs = get_obs(data, cmd, last_action, commands_scale)
    for i in range(OBS_HISTORY_LEN):
        obs_history = build_obs_history(obs_history, init_obs, num_one_step_obs)

    print("=" * 50)
    if gamepad is not None:
        print("🎮 手柄控制说明:")
        print("   左摇杆 上下 → 前进/后退")
        print("   左摇杆 左右 → 左右横移")
        print("   右摇杆 左右 → 原地转向/偏航角速度")
    else:
        print("⌨️ 键盘控制说明:")
        print("   W / S → 前进 / 后退")
        print("   A / D → 左移 / 右移")
        print("   Q / E → 左转 / 右转")
        print("   Z     → 按住蹲下蓄力，松开起跳上台阶")
        print("   注意：需要先点击一下 pygame 小窗口，确保键盘焦点在控制窗口内")
    print("=" * 50)

    step_count = 0
    ctrl_count = 0
    jump_ctrl = JumpController(has_height_cmd=has_height_cmd)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            if step_count % DECIMATION == 0:
                # 读取手柄/键盘指令
                if gamepad is not None:
                    vel_cmd = get_gamepad_cmd(gamepad)
                    if has_height_cmd:
                        cmd = np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], DEFAULT_HEIGHT_CMD], dtype=np.float32)
                    else:
                        cmd = np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2]], dtype=np.float32)
                else:
                    raw_cmd = get_keyboard_cmd()
                    cmd = jump_ctrl.update(raw_cmd)

                # 获取观测
                current_obs = get_obs(data, cmd, last_action, commands_scale)

                # 更新历史
                obs_history = build_obs_history(obs_history, current_obs, num_one_step_obs)

                # 策略推理
                obs_input = np.clip(obs_history, -CLIP_OBS, CLIP_OBS)
                obs_tensor = torch.from_numpy(obs_input).unsqueeze(0).float()
                with torch.no_grad():
                    action_tensor = policy(obs_tensor)
                action = action_tensor.squeeze(0).numpy().astype(np.float32)
                action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
                last_action = action.copy()

                ctrl_count += 1

            # PD 控制
            target_pos = DEFAULT_POS + action * ACTION_SCALE
            current_pos = data.qpos[7:].copy().astype(np.float32)
            current_vel = data.qvel[6:].copy().astype(np.float32)
            torques = KP * (target_pos - current_pos) - KD * current_vel
            torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
            data.ctrl[:] = torques

            # 仿真步进
            mujoco.mj_step(model, data)
            step_count += 1

            # 同步渲染（相机用鼠标自由控制：左键旋转、滚轮缩放、中键平移）
            viewer.sync()

            # 实时同步
            elapsed = time.time() - step_start
            sleep_time = SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 打印
            if ctrl_count % 50 == 0 and step_count % DECIMATION == 0:
                sim_time = data.time
                real_time = time.time() - start_time
                base_height = data.qpos[2]
                imu_acc = data.sensor('imu_acc').data.copy().astype(np.float32)
                gravity_body = get_projected_gravity(data, body_name="base") * 9.81
                imu_acc_nograv = imu_acc - gravity_body
                print(f"[t={sim_time:6.2f}s] "
                      f"h={base_height:.3f}  "
                      f"cmd=[{cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}]  "
                    f"|a|={np.linalg.norm(action):.3f}  "
                    f"acc_body=[{imu_acc[0]:+.2f}, {imu_acc[1]:+.2f}, {imu_acc[2]:+.2f}]  "
                    f"acc_body_nograv=[{imu_acc_nograv[0]:+.2f}, {imu_acc_nograv[1]:+.2f}, {imu_acc_nograv[2]:+.2f}]")
                # print(f"[t={sim_time:6.2f}s | real={real_time:6.2f}s] "
                #       f"height={base_height:.3f}  "
                #       f"action_norm={np.linalg.norm(action):.3f}")

    # 退出时清理
    if keyboard_screen is not None:
        pygame.display.quit()
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出仿真")
    except Exception as e:
        print(f"\n❌ 仿真出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
