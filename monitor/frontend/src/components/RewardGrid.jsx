import { memo, useState, useCallback } from 'react'

// 奖励项语义分组
const REWARD_GROUPS = {
  '速度跟踪': ['tracking_lin_vel', 'tracking_ang_vel'],
  '姿态维持': ['orientation', 'orientation_pitch', 'ang_vel_xy'],
  '足端运动': ['foot_clearance', 'foot_drag', 'similar_legged'],
  '能耗平滑': ['torques', 'torque_limits', 'dof_acc', 'dof_vel', 'action_rate', 'joint_power', 'smoothness'],
  '身高碰撞': ['base_height', 'base_height_encourage', 'collision', 'low_height_thigh_horizontal'],
  '其他约束': ['stand_still', 'lin_vel_z', 'vel_y_zero_penalize', 'hip_abduction_deviation'],
}

// 为每个奖励项分配颜色
const REWARD_COLORS = {}
Object.entries(REWARD_GROUPS).forEach(([group, items]) => {
  const groupColors = {
    '速度跟踪': '#22c55e',
    '姿态维持': '#3b82f6',
    '足端运动': '#f59e0b',
    '能耗平滑': '#a855f7',
    '身高碰撞': '#ef4444',
    '其他约束': '#6b7280',
  }
  items.forEach(item => {
    REWARD_COLORS[item] = groupColors[group] || '#6b7280'
  })
})

// 极简 SVG 折线图
function MiniSparkline({ data, color, width = 100, height = 40 }) {
  if (!data || data.length < 2) {
    return (
      <svg width={width} height={height} className="opacity-30">
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="currentColor" strokeWidth="1" />
      </svg>
    )
  }

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const step = width / (data.length - 1)

  const points = data.map((v, i) => {
    const x = i * step
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={width} height={height}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  )
}

// 单个奖励卡片
const RewardCard = memo(function RewardCard({ name, data, color, onClick }) {
  const currentValue = data && data.length > 0 ? data[data.length - 1] : null

  return (
    <div
      className="bg-muted/30 rounded-md p-2 cursor-pointer hover:bg-muted/60 transition-colors border border-transparent hover:border-primary/30"
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium truncate" title={name}>
          {name}
        </span>
        <span className={`text-xs font-mono ml-1 ${
          currentValue >= 0 ? 'text-green-600' : 'text-red-600'
        }`}>
          {currentValue !== null ? currentValue.toFixed(2) : '--'}
        </span>
      </div>
      <MiniSparkline data={data} color={color} />
    </div>
  )
})

// 放大弹窗
function DetailModal({ name, data, color, onClose }) {
  if (!data || data.length === 0) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const width = 400
  const height = 200
  const step = width / (data.length - 1)

  const points = data.map((v, i) => {
    const x = i * step
    const y = height - ((v - min) / range) * (height - 20) - 10
    return `${x},${y}`
  }).join(' ')

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-card rounded-lg border p-6 max-w-lg w-full mx-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">{name}</h3>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            ✕
          </button>
        </div>
        <div className="mb-2 flex gap-4 text-sm text-muted-foreground">
          <span>当前: <span className="font-mono text-foreground">{data[data.length - 1]?.toFixed(4)}</span></span>
          <span>最小: <span className="font-mono text-foreground">{min.toFixed(4)}</span></span>
          <span>最大: <span className="font-mono text-foreground">{max.toFixed(4)}</span></span>
        </div>
        <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="bg-muted/30 rounded">
          {/* 网格线 */}
          {[0.25, 0.5, 0.75].map(pct => (
            <line
              key={pct}
              x1="0" y1={height * pct} x2={width} y2={height * pct}
              stroke="currentColor" strokeWidth="0.5" className="opacity-10"
            />
          ))}
          {/* 数据线 */}
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinejoin="round"
          />
          {/* Y轴标签 */}
          <text x="4" y="14" fontSize="10" className="fill-muted-foreground">{max.toFixed(2)}</text>
          <text x="4" y={height - 4} fontSize="10" className="fill-muted-foreground">{min.toFixed(2)}</text>
        </svg>
      </div>
    </div>
  )
}

export const RewardGrid = memo(function RewardGrid({ rewards }) {
  const [selectedReward, setSelectedReward] = useState(null)

  const handleClick = useCallback((name) => {
    setSelectedReward(name)
  }, [])

  const handleClose = useCallback(() => {
    setSelectedReward(null)
  }, [])

  // 获取所有奖励项并按分组排序
  const allRewardNames = Object.keys(rewards)
  const orderedNames = []
  Object.values(REWARD_GROUPS).forEach(items => {
    items.forEach(item => {
      if (allRewardNames.includes(item)) {
        orderedNames.push(item)
      }
    })
  })
  // 添加未分组的项
  allRewardNames.forEach(name => {
    if (!orderedNames.includes(name)) {
      orderedNames.push(name)
    }
  })

  if (orderedNames.length === 0) {
    return (
      <div className="bg-card rounded-lg border p-4">
        <h2 className="text-lg font-semibold mb-3 text-card-foreground">奖励细节</h2>
        <p className="text-sm text-muted-foreground text-center py-8">
          等待训练数据...
        </p>
      </div>
    )
  }

  return (
    <div className="bg-card rounded-lg border p-4">
      <h2 className="text-lg font-semibold mb-3 text-card-foreground">
        奖励细节
        <span className="text-sm font-normal text-muted-foreground ml-2">
          ({orderedNames.length} 项 · 点击查看详情)
        </span>
      </h2>

      <div className="grid grid-cols-6 gap-2">
        {orderedNames.map(name => (
          <RewardCard
            key={name}
            name={name}
            data={rewards[name]}
            color={REWARD_COLORS[name] || '#6b7280'}
            onClick={() => handleClick(name)}
          />
        ))}
      </div>

      {/* 分组标签 */}
      <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
        {Object.entries(REWARD_GROUPS).map(([group, items]) => {
          const color = REWARD_COLORS[items[0]] || '#6b7280'
          return (
            <span key={group} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: color }} />
              {group}
            </span>
          )
        })}
      </div>

      {/* 放大弹窗 */}
      {selectedReward && rewards[selectedReward] && (
        <DetailModal
          name={selectedReward}
          data={rewards[selectedReward]}
          color={REWARD_COLORS[selectedReward] || '#6b7280'}
          onClose={handleClose}
        />
      )}
    </div>
  )
})
