export function LossPanel({ currentMetrics }) {
  const losses = [
    {
      label: 'Value Loss',
      value: currentMetrics.value_loss?.toFixed(4) || '--',
      color: 'bg-orange-500'
    },
    {
      label: 'Surrogate Loss',
      value: currentMetrics.surrogate_loss?.toFixed(4) || '--',
      color: 'bg-blue-500'
    },
    {
      label: 'Estimation Loss',
      value: currentMetrics.estimation_loss?.toFixed(4) || '--',
      color: 'bg-purple-500'
    },
    {
      label: 'Swap Loss',
      value: currentMetrics.swap_loss?.toFixed(4) || '--',
      color: 'bg-pink-500'
    }
  ]

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-4 text-card-foreground">损失函数</h2>

      <div className="space-y-3">
        {losses.map((loss) => (
          <div key={loss.label} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">{loss.label}</span>
              <span className="font-mono">{loss.value}</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-1.5">
              <div
                className={`h-1.5 rounded-full ${loss.color}`}
                style={{
                  width: `${Math.min(100, (parseFloat(loss.value) || 0) * 20)}%`
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
