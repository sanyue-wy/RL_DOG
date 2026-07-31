import { memo, useState } from 'react'

export const RewardBreakdown = memo(function RewardBreakdown({ rewards, currentMetrics }) {
  const [sortBy, setSortBy] = useState('value') // 'value' | 'name'

  // 获取所有奖励项
  const rewardItems = Object.keys(rewards).map(name => ({
    name,
    value: currentMetrics[`rew_${name}`] || 0,
  }))

  // 排序
  const sortedRewards = [...rewardItems].sort((a, b) => {
    if (sortBy === 'value') return b.value - a.value
    return a.name.localeCompare(b.name)
  })

  // 计算最大绝对值用于条形宽度归一化
  const maxAbsValue = Math.max(...sortedRewards.map(r => Math.abs(r.value)), 0.001)

  return (
    <div className="bg-card rounded-lg border p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-card-foreground">最新奖励分解</h2>
        <div className="flex gap-1">
          <button
            onClick={() => setSortBy('value')}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              sortBy === 'value'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            按数值
          </button>
          <button
            onClick={() => setSortBy('name')}
            className={`text-xs px-2 py-1 rounded transition-colors ${
              sortBy === 'name'
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            按名称
          </button>
        </div>
      </div>

      {sortedRewards.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-4">
          等待训练数据...
        </p>
      ) : (
        <div className="space-y-1.5 max-h-[300px] overflow-y-auto pr-2">
          {sortedRewards.map((reward) => {
            const isPositive = reward.value >= 0
            const barWidth = (Math.abs(reward.value) / maxAbsValue) * 100

            return (
              <div key={reward.name} className="flex items-center gap-2 text-sm">
                <div className="w-40 truncate text-xs text-muted-foreground text-right" title={reward.name}>
                  {reward.name}
                </div>
                <div className="flex-1 h-5 bg-secondary rounded-sm overflow-hidden relative">
                  <div
                    className={`h-full rounded-sm transition-all duration-300 ${
                      isPositive ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(100, barWidth)}%` }}
                  />
                </div>
                <div className={`w-16 text-right font-mono text-xs ${
                  isPositive ? 'text-green-600' : 'text-red-600'
                }`}>
                  {reward.value >= 0 ? '+' : ''}{reward.value.toFixed(3)}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
})
