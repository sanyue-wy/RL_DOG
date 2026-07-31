import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

export function RewardChart({ data }) {
  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-4 text-card-foreground">训练曲线</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Mean Reward 图表 */}
        <div>
          <h3 className="text-sm font-medium text-muted-foreground mb-2">Mean Reward</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis
                dataKey="iteration"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
                formatter={(value) => [value.toFixed(2), 'Reward']}
                labelFormatter={(label) => `Iteration: ${label}`}
              />
              <Line
                type="monotone"
                dataKey="reward"
                stroke="#84cc16"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Episode Length 图表 */}
        <div>
          <h3 className="text-sm font-medium text-muted-foreground mb-2">Episode Length</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
              <XAxis
                dataKey="iteration"
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px'
                }}
                formatter={(value) => [value.toFixed(0), 'Length']}
                labelFormatter={(label) => `Iteration: ${label}`}
              />
              <Line
                type="monotone"
                dataKey="episode_length"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
