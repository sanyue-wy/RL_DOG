import { TrendingUp, TrendingDown, Activity, Clock } from 'lucide-react'

export function MetricsPanel({ currentMetrics, status }) {
  const metrics = [
    {
      label: 'Mean Reward',
      value: currentMetrics.mean_reward?.toFixed(2) || '--',
      icon: Activity,
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950'
    },
    {
      label: 'Episode Length',
      value: currentMetrics.mean_episode_length?.toFixed(0) || '--',
      icon: Clock,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950'
    },
    {
      label: 'Action Noise',
      value: currentMetrics.action_noise_std?.toFixed(2) || '--',
      icon: TrendingDown,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950'
    }
  ]

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-4 text-card-foreground">核心指标</h2>

      <div className="space-y-3">
        {metrics.map((metric) => {
          const Icon = metric.icon
          return (
            <div
              key={metric.label}
              className="flex items-center gap-3 p-3 rounded-lg bg-muted/50"
            >
              <div className={`p-2 rounded-md ${metric.bgColor}`}>
                <Icon className={`w-4 h-4 ${metric.color}`} />
              </div>
              <div className="flex-1">
                <p className="text-xs text-muted-foreground">{metric.label}</p>
                <p className="text-lg font-semibold">{metric.value}</p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
