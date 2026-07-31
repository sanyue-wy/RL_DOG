# 训练监控看板 UI 改版技术报告 V3

> 版本：V3 | 日期：2026-07-31 | 状态：**已实施完成**

---

## 一、项目概述

### 1.1 改版目标

对 12-DOF 四足机器人训练监控看板进行前端布局改版，聚焦数据呈现方式优化：

- **不动**：TrainingControl（按钮区）位置与逻辑、TrainingLog（日志区）位置与逻辑
- **只改**：数据展示区域的布局、图表类型、信息密度
- **可滚动**：页面不再挤在一个屏幕内，右侧主内容区可纵向滚动

### 1.2 改版范围

| 维度 | 范围 |
|------|------|
| 后端 | 扩展指标解析（新增 7 个字段）、历史数据长度限制 |
| 前端布局 | 从"左右双栏"改为"左侧固定控制 + 右侧可滚动数据区" |
| 前端组件 | 新增 4 个组件、重写 1 个组件、保留 2 个组件 |
| 技术栈 | 不变（降级 Vite 5 适配 Node 22.5.1） |

---

## 二、技术栈

| 层级 | 技术 | 版本 | 备注 |
|------|------|------|------|
| 后端框架 | FastAPI | 0.109.0 | 不变 |
| ASGI 服务器 | Uvicorn | 0.27.0 | 不变 |
| 实时通信 | python-socketio | 5.11.0 | 不变 |
| 前端框架 | React | 18.3.1 | 从 v19 降级适配 |
| 构建工具 | Vite | 5.4.x | 从 v8 降级适配 Node 22.5.1 |
| CSS 框架 | Tailwind CSS | 3.4.x | 从 v4 降级，使用 PostCSS 模式 |
| 图表库 | Recharts | 2.12.x | 从 v3 降级适配 React 18 |
| 图标库 | lucide-react | 0.400.x | 适配 React 18 |
| 通信客户端 | socket.io-client | 4.7.x | 不变 |

---

## 三、数据流架构

### 3.1 整体数据流

```
┌─────────────────────────────────────────────────────────────────┐
│  训练进程 (train.py → HIMOnPolicyRunner.learn())                │
│  stdout 输出格式:                                                │
│    Learning iteration 164/100000                                │
│    Computation: 1250 steps/s (collection: 2.35s, learning 1.82s)│
│    Value function loss: 0.1234                                  │
│    Surrogate loss: 0.5678                                       │
│    Mean action noise std: 0.50                                  │
│    Mean reward: 45.67                                           │
│    Mean episode rew_tracking_lin_vel: 0.89                      │
│    ... (23 个 rew_* 分项)                                       │
│    Total timesteps: 3952000                                     │
│    Iteration time: 4.17s                                        │
│    Total time: 683.45s                                          │
│    ETA: 12345.6s                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ stdout (逐行)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  TrainManager._read_output() (后台线程)                          │
│    ↓ _parse_and_emit() 正则解析                                  │
│    ├── iteration_update  → { current, total }                   │
│    ├── metrics_update    → { mean_reward, value_loss, rew_*,    │
│    │                         collection_time, learning_time,     │
│    │                         steps_per_sec, iteration_time,      │
│    │                         eta, total_timesteps, ... }         │
│    └── train_output      → { data: raw_log_line }               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Socket.IO 事件
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  React App (App.jsx)                                            │
│    ├── status state        → TrainingControl, StatusBar          │
│    ├── metrics state       → CoreCharts, LossCharts              │
│    ├── rewards state       → RewardBreakdown, RewardGrid         │
│    └── logs state          → TrainingLog                         │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 后端解析正则（已修正）

> **关键修正**：初始版本的正则与训练脚本实际输出格式不匹配，导致部分字段无数据。
> 修正依据来自 `rsl_rl/rsl_rl/runners/him_on_policy_runner.py:154-227` 的 `log()` 方法。

| 字段 | 正则 | 输出行示例 |
|------|------|-----------|
| `mean_reward` | `Mean reward:\s*([-\d.]+)` | `Mean reward:                                    45.67` |
| `mean_episode_length` | `Mean episode length:\s*([-\d.]+)` | `Mean episode length:                            200.00` |
| `value_loss` | `Value function loss:\s*([-\d.]+)` | `Value function loss:                             0.1234` |
| `surrogate_loss` | `Surrogate loss:\s*([-\d.]+)` | `Surrogate loss:                                  0.5678` |
| `estimation_loss` | `Estimation loss:\s*([-\d.]+)` | `Estimation loss:                                 0.9012` |
| `swap_loss` | `Swap loss:\s*([-\d.]+)` | `Swap loss:                                       0.3456` |
| `action_noise_std` | `Mean action noise std:\s*([-\d.]+)` | `Mean action noise std:                            0.50` |
| `total_time` | `Total time:\s*([-\d.]+)s` | `Total time:                                    683.45s` |
| `iteration_time` | `Iteration time:\s*([-\d.]+)s` | `Iteration time:                                  4.17s` |
| `eta` | `ETA:\s*([-\d.]+)s` | `ETA:                                          12345.6s` |
| `total_timesteps` | `Total timesteps:\s*(\d+)` | `Total timesteps:                               3952000` |
| `steps_per_sec` | 复合正则（见下） | `Computation: 1250 steps/s (collection: 2.35s, learning 1.82s)` |
| `collection_time` | 复合正则（见下） | 同上 |
| `learning_time` | 复合正则（见下） | 同上 |
| `rew_*` | `Mean episode rew_(\w+):\s*([-\d.]+)` | `Mean episode rew_tracking_lin_vel:               0.89` |

**复合正则**（从 `Computation:` 行提取三个字段）：

```python
comp_match = re.search(
    r'Computation:\s*(\d+)\s*steps/s\s*\(collection:\s*([\d.]+)s,\s*learning\s*([\d.]+)s\)',
    line
)
# group(1) → steps_per_sec
# group(2) → collection_time
# group(3) → learning_time
```

---

## 四、页面布局设计

### 4.1 布局对比

**改版前**：
```
┌─────────────────────────────────────────────┐
│              Header (标题 + 连接状态)          │
├───────────────┬─────────────────────────────┤
│  左侧 1/3     │        右侧 2/3              │
│               │                             │
│  TrainingCtrl │  RewardChart (2个折线图)      │
│  MetricsPanel │  RewardBreakdown (条形列表)   │
│  LossPanel    │  TrainingLog (终端日志)       │
│               │                             │
└───────────────┴─────────────────────────────┘
问题：所有内容挤在一屏，数据密度低，无法滚动
```

**改版后**：
```
┌──────────────────────────────────────────────────────────┐
│                    Header (标题 + 连接状态)                 │
├──────────────┬───────────────────────────────────────────┤
│  左侧固定     │  右侧可滚动                                │
│  w-72        │                                           │
│              │  ┌───────────────────────────────────┐    │
│  TrainingCtrl│  │  StatusBar (进度条 + 7个数值卡片)    │    │
│  (按钮不变)   │  ├─────────────────┬─────────────────┤    │
│              │  │  CoreCharts     │  LossCharts     │    │
│              │  │  (3个折线图)     │  (4个折线图)     │    │
│              │  ├─────────────────┴─────────────────┤    │
│              │  │  RewardBreakdown (23项条形图)       │    │
│              │  ├───────────────────────────────────┤    │
│              │  │  RewardGrid (6×4 小型趋势图矩阵)    │    │
│              │  ├───────────────────────────────────┤    │
│              │  │  TrainingLog (日志，位置不变)        │    │
│              │  └───────────────────────────────────┘    │
└──────────────┴───────────────────────────────────────────┘
```

### 4.2 区域详细规格

#### 区域 1：顶部状态栏 (StatusBar)

| 项目 | 说明 |
|------|------|
| 位置 | 右侧顶部，通栏 |
| 内容 | 进度条 + 7 个数值卡片横向排列 |
| 进度条 | 渐变绿色，显示 `current / total` 及百分比 |
| 卡片列表 | 速度(steps/s)、采集耗时、学习耗时、迭代耗时、总步数、运行时间、ETA |
| 字体 | 数值使用等宽字体 `font-mono`，大字号 `text-base` |

#### 区域 2：左侧核心曲线区 (CoreCharts)

| 项目 | 说明 |
|------|------|
| 位置 | 右侧中部左列，占 2/3 宽度 |
| 内容 | 3 个 Recharts LineChart 垂直堆叠 |
| 子图 | Mean Reward (绿)、Action Noise Std (紫)、Episode Length (蓝) |
| 交互 | 共享 XAxis，底部 Brush 组件支持区域缩放 |
| 高度 | 150px / 120px / 120px |

#### 区域 3：右侧损失曲线区 (LossCharts)

| 项目 | 说明 |
|------|------|
| 位置 | 右侧中部右列，占 1/3 宽度 |
| 内容 | 4 个小型折线图垂直堆叠 |
| 子图 | Value Loss (橙)、Surrogate Loss (蓝)、Estimation Loss (紫)、Swap Loss (粉) |
| 高度 | 每个 90px |
| 特点 | 标题行右侧显示当前最新值 |

#### 区域 4：最新奖励条形图 (RewardBreakdown)

| 项目 | 说明 |
|------|------|
| 位置 | 右侧中下部 |
| 内容 | 水平条形图，23 个 rew_* 指标 |
| 颜色 | 正值绿色，负值红色 |
| 排序 | 支持"按数值"和"按名称"两种排序切换 |
| 滚动 | `max-h-[300px] overflow-y-auto` |

#### 区域 5：奖励细节网格 (RewardGrid)

| 项目 | 说明 |
|------|------|
| 位置 | 右侧下部 |
| 内容 | 6列 × 4行 CSS Grid，每个格子包含极简 SVG 折线图 |
| 分组 | 按语义分组排列（速度跟踪→姿态维持→足端运动→能耗平滑→身高碰撞→其他约束） |
| 颜色 | 每个分组使用不同颜色标识 |
| 交互 | 点击格子弹出放大 Modal，显示详细曲线 + 统计值 |
| 底部 | 分组颜色图例 |

---

## 五、组件架构

### 5.1 组件清单

| 组件 | 文件 | 状态 | 功能 |
|------|------|------|------|
| `TrainingControl` | `components/TrainingControl.jsx` | **保留不变** | 训练启停控制 + 进度条 |
| `TrainingLog` | `components/TrainingLog.jsx` | **保留不变** | 终端日志滚动窗口 |
| `StatusBar` | `components/StatusBar.jsx` | **新建** | 顶部状态栏：进度条 + 7 个数值卡片 |
| `CoreCharts` | `components/CoreCharts.jsx` | **新建** | 核心曲线：3 个折线图 + Brush 缩放 |
| `LossCharts` | `components/LossCharts.jsx` | **新建** | 损失曲线：4 个小型折线图 |
| `RewardBreakdown` | `components/RewardBreakdown.jsx` | **重写** | 从列表改为水平条形图 + 排序切换 |
| `RewardGrid` | `components/RewardGrid.jsx` | **新建** | 6×4 SVG 小图矩阵 + 点击放大 |
| `MetricsPanel` | `components/MetricsPanel.jsx` | **弃用** | 功能合并到 StatusBar |
| `LossPanel` | `components/LossPanel.jsx` | **弃用** | 替换为 LossCharts |
| `RewardChart` | `components/RewardChart.jsx` | **弃用** | 替换为 CoreCharts |

### 5.2 组件依赖关系

```
App.jsx
├── TrainingControl      (props: status, onStart, onStop, onPause)
├── StatusBar             (props: status, currentMetrics)
├── CoreCharts            (props: metrics)
├── LossCharts            (props: metrics)
├── RewardBreakdown       (props: rewards, currentMetrics)
├── RewardGrid            (props: rewards)
└── TrainingLog           (props: logs)
```

### 5.3 性能优化

- 所有新增组件使用 `React.memo` 包裹，避免无关 state 变化导致重渲染
- 历史数据限制 200 条（后端 + 前端双重限制）
- RewardGrid 的 MiniSparkline 使用原生 SVG 绘制，不依赖 Recharts，减少渲染开销
- 点击放大 Modal 使用 `fixed` 定位 + 背景遮罩，不触发父组件重排

---

## 六、后端改造详情

### 6.1 train_manager.py 变更

| 变更项 | 说明 |
|--------|------|
| `metrics_history` | 新增 `action_noise_std`、`collection_time`、`learning_time`、`steps_per_sec` 字段 |
| `start_time` | 新增实例变量，记录训练开始时间戳 |
| `get_status()` | 新增返回 `start_time` 和 `elapsed_seconds` |
| `_parse_and_emit()` | 修正正则表达式，新增复合正则解析 `Computation:` 行 |
| 历史数据限制 | 所有 history 列表限制 200 条，奖励分项同理 |

### 6.2 app.py 变更

| 变更项 | 说明 |
|--------|------|
| 静态文件挂载 | 新增 `/assets` 路径挂载，解决构建产物引用路径问题 |

---

## 七、前端改造详情

### 7.1 依赖降级（适配 Node 22.5.1）

| 包 | 原版本 | 新版本 | 原因 |
|----|--------|--------|------|
| vite | ^8.2.0 | ^5.4.0 | Vite 8 要求 Node >=22.12.0 |
| @vitejs/plugin-react | ^6.0.4 | ^4.3.0 | 适配 Vite 5 |
| react | ^19.2.8 | ^18.3.1 | 适配 Recharts 2 |
| react-dom | ^19.2.8 | ^18.3.1 | 同上 |
| recharts | ^3.10.1 | ^2.12.0 | 适配 React 18 |
| tailwindcss | ^4.3.3 | ^3.4.0 | Vite 5 + PostCSS 模式 |
| lucide-react | ^1.28.0 | ^0.400.x | 适配 React 18 |

### 7.2 Tailwind CSS 配置

从 v4 的 `@import "tailwindcss"` 改为 v3 的标准三件套：

```
tailwind.config.js   → 自定义颜色变量映射
postcss.config.js    → PostCSS 插件配置
index.css            → @tailwind base/components/utilities + CSS 变量
```

### 7.3 App.jsx 布局重构

```jsx
// 核心布局结构
<div className="flex gap-4 p-4">
  {/* 左侧：固定宽度控制区 */}
  <div className="w-72 flex-shrink-0">
    <TrainingControl ... />
  </div>

  {/* 右侧：可滚动数据区 */}
  <div className="flex-1 space-y-4 min-w-0">
    <StatusBar ... />
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2"><CoreCharts ... /></div>
      <div className="lg:col-span-1"><LossCharts ... /></div>
    </div>
    <RewardBreakdown ... />
    <RewardGrid ... />
    <TrainingLog ... />
  </div>
</div>
```

---

## 八、构建与部署

### 8.1 构建流程

```bash
cd monitor/frontend
npm install          # 安装依赖
npm run build        # Vite 构建 → dist/
```

### 8.2 部署流程

```bash
# 复制构建产物到 static 目录
cp dist/index.html ../static/index.html
mkdir -p ../static/assets
cp dist/assets/* ../static/assets/
```

### 8.3 目录结构

```
monitor/
├── backend/
│   ├── app.py              # FastAPI + Socket.IO 服务
│   ├── train_manager.py    # 训练进程管理 + 指标解析
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # 主布局
│   │   ├── index.css       # Tailwind + CSS 变量
│   │   └── components/
│   │       ├── StatusBar.jsx
│   │       ├── CoreCharts.jsx
│   │       ├── LossCharts.jsx
│   │       ├── RewardBreakdown.jsx
│   │       ├── RewardGrid.jsx
│   │       ├── TrainingControl.jsx  (保留)
│   │       └── TrainingLog.jsx      (保留)
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
└── static/
    ├── index.html
    └── assets/
        ├── index-*.js
        └── index-*.css
```

---

## 九、已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 速度/采集耗时/学习耗时/ETA 无数据 | 正则与实际输出格式不匹配 | 使用复合正则从 `Computation:` 行提取；新增 `Iteration time`/`ETA`/`Total timesteps` 解析 |
| Vite 8 构建失败 | Node 22.5.1 < 要求的 22.12.0 | 降级到 Vite 5 + React 18 + Recharts 2 |
| Tailwind v4 插件不兼容 | `@tailwindcss/vite` 依赖 Vite 8 | 改用 Tailwind v3 + PostCSS 模式 |
| 静态资源 404 | 构建产物引用 `/assets/` 路径 | 后端新增 `/assets` 静态文件挂载 |
| 23 个 MiniChart 性能 | Recharts 组件开销大 | RewardGrid 使用原生 SVG 绘制 |

---

## 十、后续优化方向

| 方向 | 说明 |
|------|------|
| 数据持久化 | 训练数据写入文件/数据库，刷新页面不丢失历史 |
| 多训练对比 | 支持同时监控多个训练任务 |
| 图表联动 | 所有折线图共享 Brush 缩放状态 |
| 暗色主题 | 已有 CSS 变量基础，需添加主题切换按钮 |
| 代码分割 | 使用 `React.lazy` 按需加载组件，减小首屏 bundle |
| 响应式优化 | 底部网格在小屏幕自动调整列数 |
