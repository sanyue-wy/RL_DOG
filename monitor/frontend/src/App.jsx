import { useState, useEffect, useCallback } from 'react'
import { io } from 'socket.io-client'
import { TrainingControl } from './components/TrainingControl'
import { MetricsPanel } from './components/MetricsPanel'
import { LossPanel } from './components/LossPanel'
import { RewardChart } from './components/RewardChart'
import { RewardBreakdown } from './components/RewardBreakdown'
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
    total_iterations: 100000
  })
  const [metrics, setMetrics] = useState({
    mean_reward: [],
    mean_episode_length: [],
    value_loss: [],
    surrogate_loss: [],
    estimation_loss: [],
    swap_loss: []
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
    })

    newSocket.on('metrics_update', (data) => {
      setCurrentMetrics(prev => ({ ...prev, ...data }))

      // 更新历史数据
      setMetrics(prev => {
        const newMetrics = { ...prev }
        Object.keys(data).forEach(key => {
          if (key in newMetrics && Array.isArray(newMetrics[key])) {
            newMetrics[key] = [...newMetrics[key].slice(-99), data[key]]
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
            newRewards[rewName] = [...newRewards[rewName].slice(-99), data[key]]
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

  // 准备图表数据
  const chartData = metrics.mean_reward.map((_, index) => ({
    iteration: index * 100,
    reward: metrics.mean_reward[index] || 0,
    episode_length: metrics.mean_episode_length[index] || 0
  }))

  const lossData = metrics.value_loss.map((_, index) => ({
    iteration: index * 100,
    value_loss: metrics.value_loss[index] || 0,
    surrogate_loss: metrics.surrogate_loss[index] || 0,
    estimation_loss: metrics.estimation_loss[index] || 0,
    swap_loss: metrics.swap_loss[index] || 0
  }))

  return (
    <div className="min-h-screen bg-background p-4 md:p-6 lg:p-8">
      {/* 头部 */}
      <header className="mb-6">
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

      {/* 主要内容 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 左侧：控制和指标 */}
        <div className="lg:col-span-1 space-y-6">
          {/* 训练控制 */}
          <TrainingControl
            status={status}
            onStart={handleStart}
            onStop={handleStop}
            onPause={handlePause}
          />

          {/* 核心指标 */}
          <MetricsPanel
            currentMetrics={currentMetrics}
            status={status}
          />

          {/* 损失函数 */}
          <LossPanel currentMetrics={currentMetrics} />
        </div>

        {/* 右侧：图表和日志 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 奖励曲线 */}
          <RewardChart data={chartData} />

          {/* 奖励项分解 */}
          <RewardBreakdown rewards={rewards} currentMetrics={currentMetrics} />

          {/* 训练日志 */}
          <TrainingLog logs={logs} />
        </div>
      </div>
    </div>
  )
}

export default App
