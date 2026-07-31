import { Play, Pause, Square, Eye, EyeOff, Gamepad2, Settings, Zap, RefreshCw } from 'lucide-react'
import { useState, useMemo } from 'react'

export function TrainingControl({
  status,
  onStart,
  onStop,
  onPause,
  // 新增 Props
  tasks = {},
  currentTask = 'rc',
  onTaskChange,
  onStartSim2Real,
  onStopSim2Real,
  sim2realRunning = false,
  // 策略选择 Props
  policies = [],
  selectedPolicy = '',
  isLoadingPolicies = false,
  onPolicyChange,
  onGetPolicies
}) {
  const { running, paused, render_mode, current_iteration, total_iterations, current_task_name } = status
  const [selectedRender, setSelectedRender] = useState(false)

  // 按实验和运行分组策略
  const groupedPolicies = useMemo(() => {
    const groups = {}
    policies.forEach(policy => {
      const expKey = policy.experiment
      if (!groups[expKey]) {
        groups[expKey] = {}
      }
      const runKey = policy.run
      if (!groups[expKey][runKey]) {
        groups[expKey][runKey] = []
      }
      groups[expKey][runKey].push(policy)
    })
    return groups
  }, [policies])

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-4 text-card-foreground">训练控制</h2>

      {/* 任务类型选择 */}
      <div className="mb-4">
        <label className="text-sm font-medium text-muted-foreground mb-2 block">
          选择策略（任务类型）
        </label>
        <select
          value={currentTask}
          onChange={(e) => onTaskChange && onTaskChange(e.target.value)}
          disabled={running}
          className="w-full px-3 py-2 rounded-md border border-input bg-background text-sm disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-primary"
        >
          {Object.entries(tasks).map(([key, task]) => (
            <option key={key} value={key}>
              {task.name} - {task.description}
            </option>
          ))}
        </select>
      </div>

      {/* 策略文件选择 */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-muted-foreground">
            选择验证策略
          </label>
          <button
            onClick={onGetPolicies}
            disabled={isLoadingPolicies}
            className="text-xs text-blue-600 hover:text-blue-800 disabled:opacity-50 flex items-center gap-1"
          >
            <RefreshCw className={`w-3 h-3 ${isLoadingPolicies ? 'animate-spin' : ''}`} />
            {isLoadingPolicies ? '刷新中...' : '刷新列表'}
          </button>
        </div>

        <select
          value={selectedPolicy}
          onChange={(e) => onPolicyChange && onPolicyChange(e.target.value)}
          className="w-full px-3 py-2 rounded-md border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
        >
          <option value="">
            {policies.length === 0
              ? '-- 没有找到策略文件 --'
              : '-- 请选择策略文件 --'}
          </option>

          {/* 按实验分组（不嵌套optgroup） */}
          {Object.entries(groupedPolicies).map(([experiment, runs]) => (
            Object.entries(runs).map(([run, runPolicies]) => (
              <optgroup key={`${experiment}-${run}`} label={`📁 ${experiment} / ${run}`}>
                {runPolicies.map((policy, idx) => (
                  <option key={idx} value={policy.path}>
                    {policy.filename} (iter {policy.iteration})
                  </option>
                ))}
              </optgroup>
            ))
          ))}
        </select>

        {selectedPolicy && (
          <p className="mt-1 text-xs text-muted-foreground">
            ✓ 已选择: {selectedPolicy.split('/').slice(-3).join('/')}
          </p>
        )}

        <p className="mt-1 text-xs text-muted-foreground">
          💡 共{policies.length}个策略文件可选
        </p>
      </div>

      {/* 一键训练按钮 */}
      <div className="mb-4">
        <button
          onClick={() => onStart && onStart(selectedRender)}
          disabled={running}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-md bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-lg hover:shadow-xl"
        >
          <Zap className="w-5 h-5" />
          <span>一键训练</span>
        </button>

        {/* 渲染模式选择 */}
        <div className="flex items-center gap-2 mt-2">
          <input
            type="checkbox"
            id="render-mode"
            checked={selectedRender}
            onChange={(e) => setSelectedRender(e.target.checked)}
            disabled={running}
            className="rounded border-input"
          />
          <label htmlFor="render-mode" className="text-sm text-muted-foreground cursor-pointer">
            开启渲染模式
          </label>
        </div>
      </div>

      {/* Sim2Real 按钮 */}
      <div className="mb-4">
        <button
          onClick={sim2realRunning ? onStopSim2Real : onStartSim2Real}
          disabled={!sim2realRunning && !selectedPolicy}
          className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md font-medium transition-all duration-200 ${
            sim2realRunning
              ? 'bg-orange-600 text-white hover:bg-orange-700'
              : selectedPolicy
                ? 'bg-purple-600 text-white hover:bg-purple-700'
                : 'bg-gray-400 cursor-not-allowed text-gray-200'
          }`}
        >
          <Gamepad2 className="w-4 h-4" />
          <span>{sim2realRunning ? '停止 Sim2Real' : '启动 Sim2Real'}</span>
        </button>

        {!selectedPolicy && !sim2realRunning && (
          <p className="mt-1 text-xs text-orange-600">
            ⚠️ 请先选择一个策略文件
          </p>
        )}
      </div>

      {/* 分隔线 */}
      <div className="border-t my-4" />

      {/* 原有控制按钮 */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <button
          onClick={onPause}
          disabled={!running}
          className="flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Pause className="w-4 h-4" />
          <span>{paused ? '恢复' : '暂停'}</span>
        </button>

        <button
          onClick={onStop}
          disabled={!running}
          className="flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-destructive text-destructive-foreground hover:bg-destructive/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Square className="w-4 h-4" />
          <span>停止</span>
        </button>
      </div>

      {/* 状态显示 */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">状态</span>
          <span className={`flex items-center gap-1.5 ${running ? 'text-green-600' : 'text-muted-foreground'}`}>
            <span className={`w-2 h-2 rounded-full ${running ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            {running ? (paused ? '已暂停' : '训练中') : '已停止'}
          </span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">当前策略</span>
          <span className="font-medium text-foreground truncate ml-2" title={current_task_name}>
            {current_task_name || 'Unknown'}
          </span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">渲染</span>
          <span className={render_mode ? 'text-green-600' : 'text-muted-foreground'}>
            {render_mode ? '开启' : '关闭'}
          </span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">迭代</span>
          <span className="font-mono">
            {current_iteration.toLocaleString()} / {total_iterations.toLocaleString()}
          </span>
        </div>

        {/* 进度条 */}
        <div className="w-full bg-secondary rounded-full h-2 mt-2">
          <div
            className="bg-primary h-2 rounded-full transition-all duration-300"
            style={{ width: `${total_iterations > 0 ? (current_iteration / total_iterations) * 100 : 0}%` }}
          />
        </div>
      </div>
    </div>
  )
}
