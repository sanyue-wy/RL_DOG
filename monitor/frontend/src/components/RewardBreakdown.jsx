export function RewardBreakdown({ rewards, currentMetrics }) {
  // 获取所有奖励项
  const rewardItems = Object.keys(rewards).map(name => ({
    name,
    value: currentMetrics[`rew_${name}`] || 0,
    history: rewards[name] || []
  }))

  // 按值排序
  const sortedRewards = [...rewardItems].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-4 text-card-foreground">奖励项分解</h2>

      <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
        {sortedRewards.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-4">
            等待训练数据...
          </p>
        ) : (
          sortedRewards.map((reward) => {
            const isPositive = reward.value >= 0
            const absValue = Math.abs(reward.value)
            const maxWidth = Math.min(100, absValue * 100)

            return (
              <div key={reward.name} className="flex items-center gap-2 text-sm">
                <div className="w-32 truncate text-muted-foreground" title={reward.name}>
                  {reward.name}
                </div>
                <div className="flex-1 h-4 bg-secondary rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}
                    style={{ width: `${maxWidth}%` }}
                  />
                </div>
                <div className={`w-16 text-right font-mono ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                  {reward.value >= 0 ? '+' : ''}{reward.value.toFixed(3)}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
