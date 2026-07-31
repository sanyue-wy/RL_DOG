# 12-DOF Quadruped Robot — RL Training Codebase

基于 Isaac Gym 仿真器，使用 **HIM (History-based Internal Model)** 框架 + **PPO** 算法训练 12 自由度四足机器狗。

B站演示视频：【四足复活】https://www.bilibili.com/video/BV1geTe6YETa/

部署代码见：https://github.com/kvoy202311/rc-sim2real

---

## 项目结构总览

```
12dof-dog-main/
├── legged_gym/          # 🐕 机器人环境（仿真 + 配置 + 奖励）
├── rsl_rl/              # 🧠 强化学习算法（PPO + 网络 + 训练循环）
├── IsaacGym/            # 🎮 NVIDIA 物理仿真引擎（底层依赖）
└── assets/              # 论文与图片资源
```

---

## 一、legged_gym — 机器人环境层

### 入口脚本

| 文件 | 作用 |
|---|---|
| `legged_gym/scripts/train.py` | **训练入口**。创建环境 → 创建 Runner → 调用 `ppo_runner.learn()` 开始训练循环 |
| `legged_gym/scripts/play.py` | **推理/可视化入口**。加载训练好的模型，跑 Isaac Gym 可视化，可导出 JIT 策略 |
| `legged_gym/sim2sim.py` | **Sim-to-Sim 验证**。把 Isaac Gym 训练出的策略搬到 **MuJoCo** 里跑，验证策略迁移性。支持手柄/键盘控制 |

### 任务注册机制

`legged_gym/envs/__init__.py` 是注册中心：

```python
task_registry.register("rc", RCBlindPlane, RCBlindPlaneCfg(), RCBlindPlaneCfgPPO())
```

把"任务名" → "环境类 + 配置" 绑定在一起。`train.py` 传入 `--task rc` 就能自动找到对应的类和配置。

### 环境类继承链

```
BaseTask                         # Isaac Gym 基础封装（创建 sim、管理 GPU buffer）
  └── LeggedRobot                # 四足机器人通用逻辑（step、PD 控制、reward 计算框架）
        ├── RCBlindPlane         # ✅ 主力环境：平地盲走，含高度命令
        ├── RCBlindTerrain       # 复杂地形盲走版本
        └── A1 / Go1 / AlienGo  # 其他机器人型号
```

### 核心文件说明

| 文件 | 作用 |
|---|---|
| `envs/base/base_task.py` | 调用 isaacgym API 创建仿真世界、actor、GPU tensor |
| `envs/base/legged_robot.py` | 核心 `step()` 函数：接收 action → PD 控制 → 物理仿真 → 计算观测/奖励/终止 |
| `envs/base/legged_robot_config.py` | 所有默认配置的基类（env、terrain、control、rewards、domain_rand...） |
| `envs/rc/rc_blind_plane.py` | **当前使用的环境**：重写了观测计算、奖励函数、命令采样、动作延迟等 |
| `envs/rc/rc_blind_plane_config.py` | **当前使用的配置**：4096 并行环境、12 关节、5 维命令、PPO 超参数等 |

### 工具文件

| 文件 | 作用 |
|---|---|
| `utils/task_registry.py` | 任务注册表，负责根据名字创建环境和算法 Runner |
| `utils/terrain.py` | 地形生成器（坡道、楼梯、离散障碍等） |
| `utils/math.py` | 数学工具（四元数旋转、角度归一化等） |
| `utils/helpers.py` | 命令行参数解析、配置转换、模型加载路径等 |
| `utils/logger.py` | 日志/画图工具 |

---

## 二、rsl_rl — 强化学习算法层

### Runner（训练驱动器）

| 文件 | 作用 |
|---|---|
| `runners/on_policy_runner.py` | 标准 PPO 训练循环（rollout → 计算 returns → update） |
| `runners/him_on_policy_runner.py` | **HIM 版训练循环**，多了 estimator 的 loss 和 termination obs 处理 |

### 网络模块

| 文件 | 作用 |
|---|---|
| `modules/him_actor_critic.py` | **Actor-Critic 网络**：Actor 用 estimator 输出 + 当前观测生成动作；Critic 用特权观测估计价值 |
| `modules/him_estimator.py` | **HIM Estimator**：从 6 帧历史观测中估计隐速度和 latent code（对比学习 + prototype） |
| `modules/actor_critic.py` | 标准 Actor-Critic（非 HIM 版本） |

### 算法

| 文件 | 作用 |
|---|---|
| `algorithms/him_ppo.py` | **HIM-PPO**：标准 PPO + estimator 的 estimation_loss + swap_loss |
| `algorithms/ppo.py` | 标准 PPO 算法 |

### 存储

| 文件 | 作用 |
|---|---|
| `storage/rollout_storage.py` | 存储 rollout 数据（obs, actions, rewards, dones...） |
| `storage/him_rollout_storage.py` | HIM 版存储，额外存了 critic_obs 用于 estimator 训练 |

---

## 三、数据流详解

### 训练数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练主循环                                │
│                                                                 │
│  ┌──────────┐    obs    ┌──────────────┐   action  ┌─────────┐ │
│  │ IsaacGym │ ────────→ │ HIMActorCritic│ ────────→ │  Env    │ │
│  │  4096个   │           │              │           │ .step() │ │
│  │ 并行环境  │ ←──────── │  PPO Update  │ ←──────── │         │ │
│  └──────────┘  reward   └──────────────┘  reward   └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**单次迭代详细流程：**

```
1. env.get_observations()
   → 返回 obs_buf: [4096, 270]  (45维单帧 × 6帧历史)

2. alg.act(obs, critic_obs)
   ├── obs → HIMEstimator → 估计 vel(3维) + latent(16维)
   ├── 拼接: [当前45维观测, vel(3), latent(16)] = 64维 → Actor MLP → action(12维)
   └── critic_obs → Critic MLP → value(1维)

3. env.step(actions)  ← 进入 LeggedRobot.step()
   ├── clip actions
   ├── 动作延迟随机化 (domain_rand.delay)
   ├── for _ in range(decimation=4):  # 4个仿真步
   │     ├── PD控制: torque = Kp*(target-pos) - Kd*vel
   │     ├── gym.set_dof_actuation_force_tensor(torque)
   │     └── gym.simulate() → 物理步进 0.005s
   ├── post_physics_step()
   │     ├── 刷新状态 tensor
   │     ├── compute_observations() → 拼接新观测到 obs_buf
   │     ├── compute_reward() → 计算各项奖励求和
   │     └── check_termination() → 判断是否需要 reset
   └── 返回 (obs, privileged_obs, rewards, dones, infos)

4. alg.update()  # PPO 更新
   ├── compute_returns() → GAE 计算优势函数
   ├── mini-batch 梯度下降 × 5 epochs
   │     ├── actor loss (clipped surrogate)
   │     ├── critic loss (value function)
   │     ├── estimator estimation_loss (速度预测 MSE)
   │     └── estimator swap_loss (对比学习)
   └── 更新网络参数
```

### 观测空间（单帧 45 维）

```
[0:3]   commands × scale      → 速度指令 (vx, vy, yaw_rate) × 缩放
[3:6]   base_ang_vel × scale  → 机体角速度 (陀螺仪)
[6:9]   projected_gravity     → 重力在机体坐标系的投影
[9:21]  (dof_pos - default) × scale → 12个关节位置偏差
[21:33] dof_vel × scale       → 12个关节速度
[33:45] last_actions          → 上一步的动作
```

最终观测 = 6帧历史拼接 = **270 维**

### 特权观测（单帧 239 维，仅 Critic 使用）

```
[0:45]   同 actor 观测
[45:48]  base_lin_vel × scale → 机体线速度（真值，部署时不可用）
[48:51]  disturbance           → 外部扰动力
[51:238] height_measurements   → 187维地形高度扫描
```

### 奖励函数

| 奖励项 | 权重 | 含义 |
|---|---|---|
| `tracking_lin_vel` | +3.0 | 跟踪 xy 线速度指令 |
| `tracking_ang_vel` | +2.0 | 跟踪 yaw 角速度指令 |
| `base_height_encourage` | +2.0 | 鼓励达到目标高度 |
| `similar_legged` | +2.0 | 鼓励对角腿动作一致 |
| `low_height_thigh_horizontal` | +1.0 | 低姿态时大腿趋于水平 |
| `base_height` | -3.0 | 惩罚偏离目标高度 |
| `collision` | -2.5 | 惩罚不该碰的部位碰撞 |
| `lin_vel_z` | -1.8 | 惩罚上下乱跳 |
| `orientation_pitch` | -1.2 | 惩罚俯仰不平 |
| `hip_abduction_deviation` | -0.8 | 惩罚髋关节外展过大 |
| `dof_acc` | -2.5e-7 | 惩罚关节加速度过大 |
| `action_rate` | -0.04 | 惩罚相邻步动作变化过大 |
| `smoothness` | -0.02 | 惩罚动作序列出现拐点 |
| `stand_still` | -2.0 | 零指令时惩罚偏离站姿 |
| `foot_clearance` | -0.25 | 足端净空高度约束 |
| `foot_drag` | -0.05 | 惩罚足端拖地 |
| `vel_y_zero_penalize` | -0.1 | 单轴运动时惩罚另一轴偏移 |
| `orientation` | -0.6 | 姿态平整约束 |
| `ang_vel_xy` | -0.08 | 抑制滚转/俯仰角速度 |
| `torque_limits` | -0.12 | 惩罚逼近力矩极限 |
| `torques` | -0.0001 | 力矩大小约束 |
| `dof_vel` | -2.5e-6 | 关节速度约束 |

### 域随机化（Domain Randomization）

训练时随机化以下参数来提高 sim-to-real 迁移能力：

| 参数 | 范围 | 说明 |
|---|---|---|
| 摩擦系数 | [0.2, 1.25] | 地面摩擦 |
| 负载质量 | [-1, 1.5] kg | 额外负载 |
| 连杆质量 | [0.9, 1.1] 倍 | 各连杆质量缩放 |
| 电机强度 | [0.9, 1.1] 倍 | 电机输出缩放 |
| Kp | [0.8, 1.2] 倍 | 位置刚度缩放 |
| Kd | [0.8, 1.2] 倍 | 速度阻尼缩放 |
| 动作延迟 | 0~4 步随机 | 模拟通信延迟 |
| 推机器人 | 16s 间隔，最大 1m/s | 周期性外部扰动 |
| 质心偏移 | [-0.05, 0.05] m | 质心位置随机偏移 |
| 恢复系数 | [0, 1.0] | 碰撞弹性 |

---

## 四、Sim2Sim 验证流程

```
Isaac Gym 训练                    MuJoCo 验证
     │                                │
     │  导出 JIT policy.pt            │
     │ ──────────────────────────→    │
     │                                │
     │                    ┌───────────┴───────────┐
     │                    │ mujoco 加载 scene.xml  │
     │                    │ 加载 policy.pt         │
     │                    │                       │
     │                    │ 主循环:                │
     │                    │  手柄/键盘 → cmd       │
     │                    │  读传感器 → obs        │
     │                    │  6帧历史 → policy      │
     │                    │  action → PD控制       │
     │                    │  mj_step()            │
     │                    │  渲染 + 镜头跟随       │
     │                    └───────────────────────┘
```

---

## 五、快速索引

| 你想做什么 | 看哪个文件 |
|---|---|
| 修改机器人参数（关节角、Kp/Kd、高度目标） | `envs/rc/rc_blind_plane_config.py` |
| 修改奖励函数 | `envs/rc/rc_blind_plane.py` 底部的 `_reward_*` 方法 |
| 修改观测空间 | `envs/rc/rc_blind_plane.py` 的 `compute_observations()` |
| 修改网络结构 | `modules/him_actor_critic.py` |
| 修改 estimator（速度估计器） | `modules/him_estimator.py` |
| 修改 PPO 超参数 | `envs/rc/rc_blind_plane_config.py` 底部的 `RCBlindPlaneCfgPPO` |

### 常用命令

```bash
# 训练
python legged_gym/scripts/train.py --task rc

# 可视化推理
python legged_gym/scripts/play.py --task rc

# Sim2Sim 验证（需修改 sim2sim.py 中的 MODEL_PATH 和 POLICY_PATH）
python legged_gym/sim2sim.py
```
