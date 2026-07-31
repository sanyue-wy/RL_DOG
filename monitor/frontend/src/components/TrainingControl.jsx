import { Play, Pause, Square, Eye, EyeOff } from 'lucide-react'

export function TrainingControl({ status, onStart, onStop, onPause }) {
  const { running, paused, render_mode, current_iteration, total_iterations } = status

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-4 text-card-foreground">训练控制</h2>

      {/* 控制按钮 */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <button
          onClick={() => onStart(true)}
          disabled={running}
          className="flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Eye className="w-4 h-4" />
          <span>开启渲染</span>
        </button>

        <button
          onClick={() => onStart(false)}
          disabled={running}
          className="flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <EyeOff className="w-4 h-4" />
          <span>关闭渲染</span>
        </button>

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
            style={{ width: `${(current_iteration / total_iterations) * 100}%` }}
          />
        </div>
      </div>
    </div>
  )
}
