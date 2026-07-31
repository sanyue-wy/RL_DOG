#!/usr/bin/env python3
"""
12-DOF 四足机器人训练监控系统 - 后端
"""

import sys
import webbrowser
import threading
from pathlib import Path

import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from train_manager import TrainManager

# 创建 Socket.IO 服务器
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

# 创建 FastAPI 应用
app = FastAPI(title="12-DOF 训练监控系统")

# 添加 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 训练管理器
train_manager = TrainManager(socketio=sio)

# 静态文件目录
static_dir = Path(__file__).resolve().parent.parent / "static"
print(f"[App] Static directory: {static_dir}")
print(f"[App] Static dir exists: {static_dir.exists()}")


@app.get("/")
async def root():
    """主页"""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'))
    return {"message": "12-DOF 训练监控系统 API", "static_dir": str(static_dir)}


@app.get("/api/status")
async def get_status():
    """获取训练状态"""
    return train_manager.get_status()


@app.get("/api/metrics")
async def get_metrics():
    """获取指标历史"""
    return train_manager.get_metrics_history()


# 挂载静态文件
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    # 挂载 assets 目录到 /assets 路径
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


# 创建 Socket.IO ASGI 应用
socket_app = socketio.ASGIApp(sio, app)


@sio.event
async def connect(sid, environ):
    """客户端连接"""
    print(f"[Socket.IO] Client connected: {sid}")
    status = train_manager.get_status()
    print(f"[Socket.IO] Sending status: {status}")
    await sio.emit('train_status', status, to=sid)


@sio.event
async def disconnect(sid):
    """客户端断开"""
    print(f"[Socket.IO] Client disconnected: {sid}")


@sio.on('start_training')
async def handle_start_training(sid, data):
    """处理启动训练请求"""
    print(f"[Socket.IO] start_training received: {data}")
    render = data.get('render', False)
    success, message = train_manager.start(render=render)
    response = {
        'status': 'started' if success else 'error',
        'message': message,
        'render': render,
        **train_manager.get_status()
    }
    print(f"[Socket.IO] Sending response: {response}")
    await sio.emit('train_status', response)


@sio.on('stop_training')
async def handle_stop_training(sid):
    """处理停止训练请求"""
    print(f"[Socket.IO] stop_training received")
    success, message = train_manager.stop()
    await sio.emit('train_status', {
        'status': 'stopped' if success else 'error',
        'message': message,
        **train_manager.get_status()
    })


@sio.on('pause_training')
async def handle_pause_training(sid):
    """处理暂停训练请求"""
    print(f"[Socket.IO] pause_training received")
    success, message = train_manager.toggle_pause()
    await sio.emit('train_status', {
        'status': 'paused' if train_manager.paused else 'resumed',
        'message': message,
        **train_manager.get_status()
    })


@sio.on('get_tasks')
async def handle_get_tasks(sid):
    """获取可用任务列表"""
    print(f"[Socket.IO] get_tasks received")
    tasks = train_manager.get_available_tasks()
    await sio.emit('tasks_list', {'tasks': tasks}, to=sid)


@sio.on('select_task')
async def handle_select_task(sid, data):
    """选择训练任务"""
    print(f"[Socket.IO] select_task received: {data}")
    task_name = data.get('task', 'rc')
    success, message = train_manager.set_current_task(task_name)
    await sio.emit('task_selected', {
        'success': success,
        'message': message,
        'current_task': train_manager.current_task
    })


@sio.on('get_policies')
async def handle_get_policies(sid):
    """获取可用策略列表"""
    print(f"[Socket.IO] get_policies received")
    try:
        policies = train_manager.get_available_policies()
        await sio.emit('policies_list', {'policies': policies}, to=sid)
    except Exception as e:
        print(f"[Socket.IO] get_policies error: {e}")
        await sio.emit('policies_list', {'policies': [], 'error': str(e)}, to=sid)


@sio.on('start_sim2real')
async def handle_start_sim2real(sid, data):
    """启动 Sim2Real 验证"""
    print(f"[Socket.IO] start_sim2real received: {data}")
    policy_path = data.get('policy_path', None)
    success, message = train_manager.start_sim2real(policy_path)
    await sio.emit('sim2real_status', {
        'status': 'started' if success else 'error',
        'message': message,
        'sim2real_running': train_manager.sim2real_running
    })


@sio.on('stop_sim2real')
async def handle_stop_sim2real(sid):
    """停止 Sim2Real 验证"""
    print(f"[Socket.IO] stop_sim2real received")
    success, message = train_manager.stop_sim2real()
    await sio.emit('sim2real_status', {
        'status': 'stopped' if success else 'error',
        'message': message,
        'sim2real_running': train_manager.sim2real_running
    })


def open_browser():
    """延迟打开浏览器"""
    import time
    time.sleep(1.5)
    webbrowser.open('http://localhost:8000')


if __name__ == '__main__':
    print("=" * 60)
    print("12-DOF 四足机器人训练监控系统 - 后端")
    print("=" * 60)
    print(f"访问地址: http://localhost:8000")
    print("=" * 60)

    # 自动打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
