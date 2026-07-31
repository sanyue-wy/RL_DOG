# 12-DOF 四足机器人训练监控系统 - 技术报告

> 项目地址：`~/VScode_files/12dof-dog-main/monitor`
> 最后更新：2026-07-31

---

## 项目概述

基于 Isaac Gym 仿真器的 12 自由度四足机器人训练监控系统，提供实时训练数据可视化和训练进程管理功能。

### 技术栈

| 组件 | 技术 | 说明 |
|---|---|---|
| 后端 | FastAPI + Socket.IO | 异步 Web 框架 + 实时通信 |
| 前端 | HTML + JavaScript | 原生实现，无构建依赖 |
| 通信 | WebSocket | Socket.IO 协议 |
| 数据源 | subprocess | 训练进程输出解析 |

---

## 版本历史

| 版本 | 日期 | 状态 | 说明 |
|---|---|---|---|
| v1.0 | 2026-07-31 | ✅ 完成 | 初始版本，基础功能实现 |
| v1.1 | 2026-07-31 | ✅ 完成 | Socket.IO 修复，数据流修复 |
| v2.0 | - | 📋 计划 | 功能增强版本 |

---

## 报告列表

### 已发布

- [v1.0 初始版本报告](v1.0-initial.md) - 基础功能实现
- [v1.1 Socket.IO 修复报告](v1.1-fix-socketio.md) - 数据流问题修复

### 计划中

- [v2.0 功能增强计划](v2.0-planned.md) - 下一版本功能规划

### 模板

- [报告模板](templates/report-template.md) - 新报告撰写模板

---

## 快速链接

| 资源 | 路径 |
|---|---|
| 后端代码 | `monitor/backend/` |
| 前端代码 | `monitor/static/` |
| 启动脚本 | `monitor/backend/app.py` |
| 训练管理 | `monitor/backend/train_manager.py` |

---

## 启动方式

```bash
cd ~/VScode_files/12dof-dog-main/monitor/backend
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rl
python app.py
```

访问地址：http://localhost:8000

---

## 版本更新流程

1. 创建新报告文件：`v[x.x]-[描述].md`
2. 使用 [报告模板](templates/report-template.md) 撰写
3. 更新本 README.md 的版本历史表
4. 更新报告列表链接
