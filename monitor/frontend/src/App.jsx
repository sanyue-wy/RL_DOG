import { useState, useEffect, useCallback } from 'react'
import { io } from 'socket.io-client'
import { TrainingControl } from './components/TrainingControl'
import { StatusBar } from './components/StatusBar'
import { CoreCharts } from './components/CoreCharts'
import { LossCharts } from './components/LossCharts'
import { RewardBreakdown } from './components/RewardBreakdown'
import { RewardGrid } from './components/RewardGrid'
import { TrainingLog } from './components/TrainingLog'
import './index.css'

function App() {
  const [socket, setSocket] = useState(null)
  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState({
    running: false,
    paused: false,
    render_mode: false,
    current_iteration: 0,
    total_iterations: 100000,
    start_time: null,
    elapsed_seconds: 0,
  })
  const [metrics, setMetrics] = useState({
    mean_reward: [],
    mean_episode_length: [],
    action_noise_std: [],
    value_loss: [],
    surrogate_loss: [],
    estimation_loss: [],
    swap_loss: [],
    iterations: [],
  })
  const [rewards, setRewards] = useState({})
  const [logs, setLogs] = useState([])
  const [currentMetrics, setCurrentMetrics] = useState({})

  // 初始化 Socket.IO 连接
  useEffect(() => {
    const newSocket = io(window.location.origin, {
      path: '/socket.io',
      transports: ['websocket', 'polling']
    })

    newSocket.on('connect', () => {
      console.log('Connected to server')
      setConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server')
      setConnected(false)
    })

    newSocket.on('train_status', (data) => {
      console.log('Train status:', data)
      setStatus(prev => ({ ...prev, ...data }))
    })

    newSocket.on('iteration_update', (data) => {
      setStatus(prev => ({
        ...prev,
        current_iteration: data.current,
        total_iterations: data.total
      }))
      // 记录迭代次数
      setMetrics(prev => ({
        ...prev,
        iterations: [...prev.iterations.slice(-199), data.current]
      }))
    })

    newSocket.on('metrics_update', (data) => {
      setCurrentMetrics(prev => ({ ...prev, ...data }))

      // 更新历史数据
      setMetrics(prev => {
        const newMetrics = { ...prev }
        Object.keys(data).forEach(key => {
          if (key in newMetrics && Array.isArray(newMetrics[key])) {
            newMetrics[key] = [...newMetrics[key].slice(-199), data[key]]
          }
        })
        return newMetrics
      })

      // 更新奖励数据
      setRewards(prev => {
        const newRewards = { ...prev }
        Object.keys(data).forEach(key => {
          if (key.startsWith('rew_')) {
            const rewName = key.replace('rew_', '')
            if (!newRewards[rewName]) {
              newRewards[rewName] = []
            }
            newRewards[rewName] = [...newRewards[rewName].slice(-199), data[key]]
          }
        })
        return newRewards
      })
    })

    newSocket.on('train_output', (data) => {
      setLogs(prev => [...prev.slice(-99), data.data])
    })

    setSocket(newSocket)

    return () => {
      newSocket.close()
    }
  }, [])

  // 启动训练
  const handleStart = useCallback((render) => {
    if (socket) {
      socket.emit('start_training', { render })
    }
  }, [socket])

  // 停止训练
  const handleStop = useCallback(() => {
    if (socket) {
      socket.emit('stop_training')
    }
  }, [socket])

  // 暂停/恢复训练
  const handlePause = useCallback(() => {
    if (socket) {
      socket.emit('pause_training')
    }
  }, [socket])

  return (
    <div className="min-h-screen bg-background">
      {/* 头部 */}
      <header className="p-4 pb-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">
              12-DOF 四足机器人训练监控系统
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              基于 Isaac Gym + HIM-PPO 的强化学习训练
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-muted-foreground">
              {connected ? '已连接' : '未连接'}
            </span>
          </div>
        </div>
      </header>

      {/* 主要内容 - 左右两栏 */}
      <div className="flex gap-4 p-4">
        {/* 左侧：训练控制 (固定宽度，不滚动) */}
        <div className="w-72 flex-shrink-0">
          <TrainingControl
            status={status}
            onStart={handleStart}
            onStop={handleStop}
            onPause={handlePause}
          />
        </div>

        {/* 右侧：数据展示区 (可滚动) */}
        <div className="flex-1 space-y-4 min-w-0">
          {/* 区域1: 顶部状态栏 */}
          <StatusBar status={status} currentMetrics={currentMetrics} />

          {/* 区域2+3: 核心曲线 + 损失曲线 (并排) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <CoreCharts metrics={metrics} />
            </div>
            <div className="lg:col-span-1">
              <LossCharts metrics={metrics} />
            </div>
          </div>

          {/* 区域4: 最新奖励条形图 */}
          <RewardBreakdown rewards={rewards} currentMetrics={currentMetrics} />

          {/* 区域5: 奖励细节网格 */}
          <RewardGrid rewards={rewards} />

          {/* 训练日志 (位置不变，在底部) */}
          <TrainingLog logs={logs} />
        </div>
      </div>
    </div>
  )
}

export default App
