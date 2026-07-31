#!/usr/bin/env python3
"""训练进程管理器 - 修复版"""

import os
import sys
import subprocess
import signal
import threading
import re
import asyncio
from pathlib import Path
from datetime import datetime


class TrainManager:
    """训练进程管理器"""

    def __init__(self, socketio=None):
        self.process = None
        self.paused = False
        self.running = False
        self.render_mode = False
        self.project_root = Path(__file__).parent.parent.parent
        self.socketio = socketio

        # Conda环境配置 - rl环境有mujoco等依赖
        self.conda_python = Path.home() / "miniconda3" / "envs" / "rl" / "bin" / "python"
        if not self.conda_python.exists():
            # 如果rl环境不存在，使用系统python
            self.conda_python = Path(sys.executable)
        print(f"[TrainManager] Using Python: {self.conda_python}")
        self.current_iteration = 0
        self.total_iterations = 0
        self.loop = None
        self.metrics_history = {
            'iterations': [],
            'mean_reward': [],
            'mean_episode_length': [],
            'action_noise_std': [],
            'value_loss': [],
            'surrogate_loss': [],
            'estimation_loss': [],
            'swap_loss': [],
            'collection_time': [],
            'learning_time': [],
            'steps_per_sec': [],
            'rewards': {}
        }
        self.start_time = None  # 训练开始时间戳

        # 新增：策略配置
        self.available_tasks = {
            'rc': {
                'name': 'RC Blind Plane',
                'description': '12-DOF 四足机器人平地训练',
                'config': 'RCBlindPlaneCfg',
                'ppo_config': 'RCBlindPlaneCfgPPO',
                'env_class': 'RCBlindPlane'
            },
            'rc_terrain': {
                'name': 'RC Blind Terrain',
                'description': '12-DOF 四足机器人地形训练',
                'config': 'RCBlindTerrainCfg',
                'ppo_config': 'RCBlindTerrainCfgPPO',
                'env_class': 'RCBlindTerrain'
            },
            'a1': {
                'name': 'Unitree A1',
                'description': 'A1 四足机器人训练',
                'config': 'A1RoughCfg',
                'ppo_config': 'A1RoughCfgPPO',
                'env_class': 'LeggedRobot'
            },
            'go1': {
                'name': 'Unitree Go1',
                'description': 'Go1 四足机器人训练',
                'config': 'Go1RoughCfg',
                'ppo_config': 'Go1RoughCfgPPO',
                'env_class': 'LeggedRobot'
            }
        }
        self.current_task = 'rc'  # 默认任务

        # 新增：Sim2Real 进程
        self.sim2real_process = None
        self.sim2real_running = False

    def _emit_sync(self, event, data):
        """在线程中同步发送 Socket.IO 事件"""
        if self.socketio and self.loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.socketio.emit(event, data),
                    self.loop
                )
            except Exception as e:
                print(f"[Emit Error] {event}: {e}")

    def get_available_tasks(self):
        """获取可用任务列表"""
        return self.available_tasks

    def set_current_task(self, task_name):
        """设置当前任务"""
        if task_name in self.available_tasks:
            self.current_task = task_name
            return True, f"已切换到任务: {self.available_tasks[task_name]['name']}"
        return False, f"未知任务: {task_name}"

    def start(self, render=False, task=None):
        """启动训练（扩展版）"""
        if self.running:
            return False, "训练已在运行中"

        # 使用指定任务或当前任务
        target_task = task or self.current_task

        if target_task not in self.available_tasks:
            return False, f"未知任务: {target_task}"

        # 获取事件循环
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        cmd = [
            sys.executable,
            "-u",  # 无缓冲模式
            str(self.project_root / "legged_gym" / "legged_gym" / "scripts" / "train.py"),
            "--task", target_task
        ]

        if not render:
            cmd.append("--headless")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,  # 无缓冲
                cwd=str(self.project_root),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            self.running = True
            self.render_mode = render
            self.paused = False
            self.current_task = target_task
            self.start_time = datetime.now()

            # 启动输出读取线程
            thread = threading.Thread(target=self._read_output, daemon=True)
            thread.start()

            task_name = self.available_tasks[target_task]['name']
            print(f"[TrainManager] 训练已启动 (task={target_task}, render={render})")
            return True, f"训练已启动 - {task_name} (渲染: {'开启' if render else '关闭'})"
        except Exception as e:
            print(f"[TrainManager] 启动失败: {e}")
            return False, f"启动失败: {str(e)}"

    def stop(self):
        """停止训练"""
        if not self.running:
            return False, "训练未运行"

        try:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.running = False
            self.paused = False
            return True, "训练已停止"
        except Exception as e:
            return False, f"停止失败: {str(e)}"

    def toggle_pause(self):
        """切换暂停状态"""
        if not self.running:
            return False, "训练未运行"

        try:
            if self.paused:
                self.process.send_signal(signal.SIGCONT)
                self.paused = False
                return True, "训练已恢复"
            else:
                self.process.send_signal(signal.SIGSTOP)
                self.paused = True
                return True, "训练已暂停"
        except Exception as e:
            return False, f"操作失败: {str(e)}"

    def get_status(self):
        """获取训练状态（扩展版）"""
        elapsed = 0
        if self.start_time and self.running:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            'running': self.running,
            'paused': self.paused,
            'render_mode': self.render_mode,
            'current_iteration': self.current_iteration,
            'total_iterations': self.total_iterations,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'elapsed_seconds': round(elapsed, 1),
            # 新增
            'current_task': self.current_task,
            'current_task_name': self.available_tasks.get(self.current_task, {}).get('name', 'Unknown'),
            'sim2real_running': self.sim2real_running
        }

    def start_sim2real(self, policy_path=None):
        """启动 Sim2Real 验证（支持指定策略路径）"""
        if self.sim2real_running:
            return False, "Sim2Real 验证已在运行中"

        # 获取事件循环（用于发送Socket.IO事件）
        if not self.loop:
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

        # 如果没有指定策略，自动查找最新
        if not policy_path:
            policy_path = self.find_latest_policy()
            if not policy_path:
                return False, "没有找到可用的策略文件"
            print(f"[TrainManager] 自动选择最新策略: {policy_path}")

        # 验证策略文件存在
        if not Path(policy_path).exists():
            return False, f"策略文件不存在: {policy_path}"

        sim2sim_path = self.project_root / "legged_gym" / "sim2sim.py"

        # 通过命令行参数传递策略路径
        cmd = [
            str(self.conda_python),
            "-u",
            str(sim2sim_path),
            "--policy", str(policy_path)
        ]

        try:
            self.sim2real_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=0,
                cwd=str(self.project_root / "legged_gym"),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            self.sim2real_running = True

            # 启动输出读取线程
            thread = threading.Thread(target=self._read_sim2real_output, daemon=True)
            thread.start()

            policy_name = Path(policy_path).name
            print(f"[TrainManager] Sim2Real 验证已启动 (policy={policy_name})")
            return True, f"Sim2Real 验证已启动: {policy_name}"
        except Exception as e:
            print(f"[TrainManager] Sim2Real 启动失败: {e}")
            return False, f"Sim2Real 启动失败: {str(e)}"

    def stop_sim2real(self):
        """停止 Sim2Real 验证"""
        if not self.sim2real_running:
            return False, "Sim2Real 验证未运行"

        try:
            if self.sim2real_process:
                self.sim2real_process.terminate()
                try:
                    self.sim2real_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.sim2real_process.kill()
            self.sim2real_running = False
            return True, "Sim2Real 验证已停止"
        except Exception as e:
            return False, f"停止失败: {str(e)}"

    def _read_sim2real_output(self):
        """读取 Sim2Real 输出"""
        try:
            print("[TrainManager] Sim2Real 输出读取线程已启动")
            for line in iter(self.sim2real_process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    print(f"[Sim2Real] {line[:100]}")
                    self._emit_sync('sim2real_output', {'data': line})

            self.sim2real_running = False
            self._emit_sync('sim2real_status', {
                'status': 'finished',
                'message': 'Sim2Real 验证已完成',
                'sim2real_running': False
            })
            print("[TrainManager] Sim2Real 输出读取线程已结束")
        except Exception as e:
            print(f"[TrainManager] Sim2Real 输出读取错误: {e}")
            self._emit_sync('sim2real_output', {'data': f'Error: {str(e)}'})
            self.sim2real_running = False
            self._emit_sync('sim2real_status', {
                'status': 'error',
                'message': f'Sim2Real 错误: {str(e)}',
                'sim2real_running': False
            })

    def get_available_policies(self):
        """扫描logs目录，返回所有可用的策略文件"""
        policies = []
        logs_dir = self.project_root / "legged_gym" / "logs"

        if not logs_dir.exists():
            print(f"[TrainManager] logs目录不存在: {logs_dir}")
            return policies

        # 遍历实验目录（blindplane, blindrough3等）
        for exp_dir in logs_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # 遍历训练运行目录（Jul31_14-43-25_等）
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # 扫描模型文件（model_0.pt, model_100.pt等）
                model_files = list(run_dir.glob("model_*.pt"))

                for model_file in model_files:
                    try:
                        iteration = int(model_file.stem.split('_')[1])
                        policies.append({
                            'path': str(model_file),
                            'experiment': exp_dir.name,
                            'run': run_dir.name,
                            'iteration': iteration,
                            'filename': model_file.name,
                            'display_name': f"{exp_dir.name}/{run_dir.name}/{model_file.name}"
                        })
                    except (ValueError, IndexError):
                        continue

        # 排序：先按实验名，再按运行时间，最后按迭代次数
        policies.sort(key=lambda x: (x['experiment'], x['run'], x['iteration']))

        print(f"[TrainManager] 找到 {len(policies)} 个策略文件")
        return policies

    def find_latest_policy(self):
        """查找最新的策略文件（按修改时间）"""
        policies = self.get_available_policies()
        if not policies:
            return None

        # 按修改时间排序，返回最新的
        policies_with_mtime = []
        for p in policies:
            try:
                mtime = Path(p['path']).stat().st_mtime
                policies_with_mtime.append((p, mtime))
            except:
                continue

        if policies_with_mtime:
            policies_with_mtime.sort(key=lambda x: x[1], reverse=True)
            return policies_with_mtime[0][0]['path']

        return policies[-1]['path'] if policies else None

    def get_metrics_history(self):
        """获取指标历史"""
        return self.metrics_history

    def _read_output(self):
        """读取训练输出"""
        try:
            print("[TrainManager] 输出读取线程已启动")
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    # 调试：打印前100字符
                    print(f"[Output] {line[:100]}")
                    self._emit_sync('train_output', {'data': line})
                    self._parse_and_emit(line)

            self.running = False
            self._emit_sync('train_status', {'status': 'finished'})
            print("[TrainManager] 输出读取线程已结束")
        except Exception as e:
            print(f"[TrainManager] 输出读取错误: {e}")
            self._emit_sync('train_output', {'data': f'Error: {str(e)}'})

    def _parse_and_emit(self, line):
        """解析训练输出并发送数据"""
        try:
            metrics = {}

            # 解析迭代信息
            match = re.search(r'Learning iteration (\d+)/(\d+)', line)
            if match:
                self.current_iteration = int(match.group(1))
                self.total_iterations = int(match.group(2))
                self.metrics_history['iterations'].append(self.current_iteration)
                if len(self.metrics_history['iterations']) > 200:
                    self.metrics_history['iterations'] = self.metrics_history['iterations'][-200:]
                self._emit_sync('iteration_update', {
                    'current': self.current_iteration,
                    'total': self.total_iterations
                })

            # 解析各项指标
            patterns = {
                'mean_reward': r'Mean reward:\s*([-\d.]+)',
                'mean_episode_length': r'Mean episode length:\s*([-\d.]+)',
                'value_loss': r'Value function loss:\s*([-\d.]+)',
                'surrogate_loss': r'Surrogate loss:\s*([-\d.]+)',
                'estimation_loss': r'Estimation loss:\s*([-\d.]+)',
                'swap_loss': r'Swap loss:\s*([-\d.]+)',
                'action_noise_std': r'Mean action noise std:\s*([-\d.]+)',
                'total_time': r'Total time:\s*([-\d.]+)s',
                'iteration_time': r'Iteration time:\s*([-\d.]+)s',
                'eta': r'ETA:\s*([-\d.]+)s',
                'total_timesteps': r'Total timesteps:\s*(\d+)',
            }

            # 特殊处理: 从 Computation 行解析 steps/s, collection_time, learning_time
            comp_match = re.search(r'Computation:\s*(\d+)\s*steps/s\s*\(collection:\s*([\d.]+)s,\s*learning\s*([\d.]+)s\)', line)
            if comp_match:
                metrics['steps_per_sec'] = float(comp_match.group(1))
                metrics['collection_time'] = float(comp_match.group(2))
                metrics['learning_time'] = float(comp_match.group(3))

            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1))
                    metrics[key] = value
                    if key in self.metrics_history:
                        self.metrics_history[key].append(value)
                    # 限制历史数据长度，防止内存溢出
                    if key in self.metrics_history and len(self.metrics_history[key]) > 200:
                        self.metrics_history[key] = self.metrics_history[key][-200:]

            # 解析奖励项
            rew_match = re.search(r'Mean episode rew_(\w+):\s*([-\d.]+)', line)
            if rew_match:
                rew_name = rew_match.group(1)
                rew_value = float(rew_match.group(2))
                metrics[f'rew_{rew_name}'] = rew_value
                if rew_name not in self.metrics_history['rewards']:
                    self.metrics_history['rewards'][rew_name] = []
                self.metrics_history['rewards'][rew_name].append(rew_value)
                # 限制奖励历史数据长度
                if len(self.metrics_history['rewards'][rew_name]) > 200:
                    self.metrics_history['rewards'][rew_name] = self.metrics_history['rewards'][rew_name][-200:]

            if metrics:
                print(f"[Parsed] {list(metrics.keys())}")
                self._emit_sync('metrics_update', metrics)

        except Exception as e:
            print(f"[Parse Error] {e}")
