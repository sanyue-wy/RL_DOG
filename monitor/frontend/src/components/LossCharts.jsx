import { memo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const LOSS_CONFIG = [
  { key: 'value_loss', label: 'Value Loss', color: '#f97316' },
  { key: 'surrogate_loss', label: 'Surrogate Loss', color: '#3b82f6' },
  { key: 'estimation_loss', label: 'Estimation Loss', color: '#a855f7' },
  { key: 'swap_loss', label: 'Swap Loss', color: '#ec4899' },
]

function LossMiniChart({ data, config, height = 90 }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-muted-foreground">{config.label}</span>
        <span className="text-xs font-mono" style={{ color: config.color }}>
          {data.length > 0 ? data[data.length - 1][config.key]?.toFixed(4) : '--'}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
          <XAxis
            dataKey="iteration"
            tick={{ fontSize: 9 }}
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
          />
          <YAxis tick={{ fontSize: 9 }} width={45} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
              fontSize: 11
            }}
            formatter={(value) => [value.toFixed(4), config.label]}
            labelFormatter={(label) => `迭代: ${label}`}
          />
          <Line
            type="monotone"
            dataKey={config.key}
            stroke={config.color}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export const LossCharts = memo(function LossCharts({ metrics }) {
  const iterations = metrics.iterations || metrics.value_loss?.map((_, i) => i * 100) || []
  const data = iterations.map((iter, i) => ({
    iteration: iter,
    value_loss: metrics.value_loss?.[i] ?? 0,
    surrogate_loss: metrics.surrogate_loss?.[i] ?? 0,
    estimation_loss: metrics.estimation_loss?.[i] ?? 0,
    swap_loss: metrics.swap_loss?.[i] ?? 0,
  }))

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3 text-card-foreground">损失函数</h2>
      <div className="space-y-3">
        {LOSS_CONFIG.map((config) => (
          <LossMiniChart key={config.key} data={data} config={config} />
        ))}
      </div>
    </div>
  )
})
