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

    def start(self, render=False):
        """启动训练"""
        if self.running:
            return False, "训练已在运行中"

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
            "--task", "rc"
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
            self.start_time = datetime.now()

            # 启动输出读取线程
            thread = threading.Thread(target=self._read_output, daemon=True)
            thread.start()

            print(f"[TrainManager] 训练已启动 (render={render})")
            return True, f"训练已启动 (渲染: {'开启' if render else '关闭'})"
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
        """获取训练状态"""
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
            'elapsed_seconds': round(elapsed, 1)
        }

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
