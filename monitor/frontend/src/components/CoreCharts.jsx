import { memo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Brush
} from 'recharts'

function MiniLineChart({ data, dataKey, title, color, height = 150 }) {
  return (
    <div>
      <h3 className="text-sm font-medium text-muted-foreground mb-1">{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" className="opacity-20" />
          <XAxis
            dataKey="iteration"
            tick={{ fontSize: 10 }}
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
          />
          <YAxis tick={{ fontSize: 10 }} width={50} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
              fontSize: 12
            }}
            formatter={(value) => [value.toFixed(3), title]}
            labelFormatter={(label) => `迭代: ${label}`}
          />
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export const CoreCharts = memo(function CoreCharts({ metrics }) {
  // 构建图表数据
  const iterations = metrics.iterations || metrics.mean_reward?.map((_, i) => i * 100) || []
  const data = iterations.map((iter, i) => ({
    iteration: iter,
    mean_reward: metrics.mean_reward?.[i] ?? 0,
    action_noise_std: metrics.action_noise_std?.[i] ?? 0,
    mean_episode_length: metrics.mean_episode_length?.[i] ?? 0,
  }))

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3 text-card-foreground">核心训练曲线</h2>
      <div className="space-y-4">
        <MiniLineChart
          data={data}
          dataKey="mean_reward"
          title="平均奖励 (Mean Reward)"
          color="#22c55e"
          height={150}
        />
        <MiniLineChart
          data={data}
          dataKey="action_noise_std"
          title="动作噪声标准差 (Action Noise Std)"
          color="#a855f7"
          height={120}
        />
        <MiniLineChart
          data={data}
          dataKey="mean_episode_length"
          title="平均 Episode 长度"
          color="#3b82f6"
          height={120}
        />
      </div>
      {data.length > 10 && (
        <div className="mt-2">
          <Brush
            dataKey="iteration"
            height={20}
            stroke="hsl(var(--muted-foreground))"
            fill="hsl(var(--muted))"
          />
        </div>
      )}
    </div>
  )
})
