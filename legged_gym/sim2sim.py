import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
import os
import pygame

import math


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


def get_keyboard_cmd() -> np.ndarray:
    """从键盘读取线速度指令和偏航角速度指令。"""
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

# ======================== 配置 ========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "resources", "robots", "RCV3", "xml", "scene.xml")
POLICY_PATH = os.path.join(_SCRIPT_DIR, "logs", "blindplane", "Jul31_14-43-25_",  "model_1200.pt")

DT = 0.02           # 控制频率 50Hz
DECIMATION = 4       # 与训练一致
SIM_DT = DT / DECIMATION  # 仿真步长 0.005s

NUM_ACTIONS = 12
NUM_ONE_STEP_OBS = 45
OBS_HISTORY_LEN = 6
NUM_OBS = NUM_ONE_STEP_OBS * OBS_HISTORY_LEN  # 270

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

# commands 缩放（与训练 commands_scale 一致）
COMMANDS_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

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

# 手柄缺失时的原始命令输入 [vx, vy, yaw_rate]
CMD = np.array([0.0, 0.0, 0.0], dtype=np.float32)


# ======================== 工具函数 ========================
def get_obs(data, cmd, last_action):
    q = data.qpos.astype(np.float32).copy()
    dq = data.qvel.astype(np.float32).copy()

    gyro = data.sensor('imu_gyro').data.copy().astype(np.float32)
    projected_gravity = get_projected_gravity(data, body_name="base")

    dof_pos = q[7:19]
    dof_vel = dq[6:18]

    obs = np.concatenate([
        cmd.astype(np.float32) * COMMANDS_SCALE,
        gyro * OBS_SCALES["ang_vel"],
        projected_gravity,
        (dof_pos - DEFAULT_POS).astype(np.float32) * OBS_SCALES["dof_pos"],
        dof_vel.astype(np.float32) * OBS_SCALES["dof_vel"],
        last_action.astype(np.float32),
    ]).astype(np.float32)

    return obs


def build_obs_history(obs_history: np.ndarray, new_obs: np.ndarray) -> np.ndarray:
    """
    维护观测历史，最新帧在前（与训练一致）：
    obs_buf = [t, t-1, t-2, t-3, t-4, t-5]
    """
    obs_history = np.roll(obs_history, NUM_ONE_STEP_OBS)
    obs_history[:NUM_ONE_STEP_OBS] = new_obs
    return obs_history


# ======================== 主循环 ========================
def main():
    # 初始化手柄
    gamepad = init_gamepad()
    keyboard_screen = None
    if gamepad is None:
        keyboard_screen = init_keyboard()

    # 加载模型
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # 加载策略
    device = torch.device('cpu')
    policy = torch.jit.load(POLICY_PATH, map_location=device)
    policy.eval()

    # 初始化状态
    mujoco.mj_resetData(model, data)
    data.qpos[7:] = DEFAULT_POS.copy()
    # 与当前 floating-base XML 中的站立初始高度一致，避免一开始把机身压得过低
    data.qpos[2] = 0.52
    mujoco.mj_forward(model, data)

    # 初始化
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    obs_history = np.zeros(NUM_OBS, dtype=np.float32)
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    cmd = CMD.copy()

    init_obs = get_obs(data, cmd, last_action)
    for i in range(OBS_HISTORY_LEN):
        obs_history = build_obs_history(obs_history, init_obs)

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
        print("   注意：需要先点击一下 pygame 小窗口，确保键盘焦点在控制窗口内")
    print("=" * 50)

    step_count = 0
    ctrl_count = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            if step_count % DECIMATION == 0:
                # 读取手柄指令
                if gamepad is not None:
                    cmd = get_gamepad_cmd(gamepad)
                else:
                    cmd = get_keyboard_cmd()

                # 获取观测
                current_obs = get_obs(data, cmd, last_action)

                # 更新历史
                obs_history = build_obs_history(obs_history, current_obs)

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

            # ========== 镜头跟随机器人朝向 ==========
            robot_pos = data.qpos[:3].copy()
            robot_quat = data.qpos[3:7].copy()  # [w, x, y, z]

            # 从四元数提取 yaw 角
            w, x, y, z = robot_quat
            yaw = np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z))
            yaw_deg = np.degrees(yaw)

            # 更新 lookat
            viewer.cam.lookat[0] = robot_pos[0]
            viewer.cam.lookat[1] = robot_pos[1]
            viewer.cam.lookat[2] = robot_pos[2]

            # 摄像机方位角跟随 yaw（偏移0度表示从背后看）
            viewer.cam.azimuth = yaw_deg + 90.0
            viewer.cam.elevation = -20.0
            viewer.cam.distance = 3.0

            # 同步渲染
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
    main()
