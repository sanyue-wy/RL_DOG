import { memo } from 'react'

function formatTime(seconds) {
  if (!seconds || seconds <= 0) return '--'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h${m.toString().padStart(2, '0')}m`
  if (m > 0) return `${m}m${s.toString().padStart(2, '0')}s`
  return `${s}s`
}

export const StatusBar = memo(function StatusBar({ status, currentMetrics }) {
  const { current_iteration, total_iterations, elapsed_seconds } = status
  const progress = total_iterations > 0 ? (current_iteration / total_iterations) * 100 : 0

  const cards = [
    { label: '速度', value: currentMetrics.steps_per_sec ? currentMetrics.steps_per_sec.toFixed(0) : '--', unit: 'steps/s' },
    { label: '采集耗时', value: currentMetrics.collection_time?.toFixed(2) || '--', unit: 's' },
    { label: '学习耗时', value: currentMetrics.learning_time?.toFixed(2) || '--', unit: 's' },
    { label: '迭代耗时', value: currentMetrics.iteration_time?.toFixed(2) || '--', unit: 's' },
    { label: '总步数', value: currentMetrics.total_timesteps?.toLocaleString() || '--', unit: '' },
    { label: '运行时间', value: currentMetrics.total_time ? formatTime(currentMetrics.total_time) : formatTime(elapsed_seconds), unit: '' },
    { label: 'ETA', value: currentMetrics.eta ? formatTime(currentMetrics.eta) : '--', unit: '' },
  ]

  return (
    <div className="bg-card rounded-lg border p-4 mb-4">
      {/* 进度条 */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-sm mb-1">
          <span className="text-muted-foreground">训练进度</span>
          <span className="font-mono font-semibold">
            {current_iteration.toLocaleString()} / {total_iterations.toLocaleString()}
          </span>
        </div>
        <div className="w-full bg-secondary rounded-full h-3">
          <div
            className="h-3 rounded-full transition-all duration-500 bg-gradient-to-r from-green-500 to-emerald-400"
            style={{ width: `${Math.min(100, progress)}%` }}
          />
        </div>
        <div className="text-right text-xs text-muted-foreground mt-1">
          {progress.toFixed(2)}%
        </div>
      </div>

      {/* 数值卡片 */}
      <div className="grid grid-cols-7 gap-2">
        {cards.map((card) => (
          <div key={card.label} className="bg-muted/50 rounded-md p-2 text-center">
            <div className="text-xs text-muted-foreground mb-1">{card.label}</div>
            <div className="text-base font-bold font-mono">{card.value}</div>
            {card.unit && (
              <div className="text-xs text-muted-foreground">{card.unit}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
})
