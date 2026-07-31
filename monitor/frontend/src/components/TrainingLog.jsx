import { useRef, useEffect } from 'react'
import { Terminal, Trash2 } from 'lucide-react'

export function TrainingLog({ logs }) {
  const logContainerRef = useRef(null)

  // 自动滚动到底部
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="bg-card rounded-lg border p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-card-foreground flex items-center gap-2">
          <Terminal className="w-5 h-5" />
          训练日志
        </h2>
        <span className="text-xs text-muted-foreground">
          {logs.length} 条记录
        </span>
      </div>

      <div
        ref={logContainerRef}
        className="bg-muted/50 rounded-md p-3 h-[200px] overflow-y-auto font-mono text-xs"
      >
        {logs.length === 0 ? (
          <p className="text-muted-foreground text-center py-4">
            等待训练输出...
          </p>
        ) : (
          logs.map((log, index) => (
            <div
              key={index}
              className={`py-0.5 ${
                log.includes('Error') ? 'text-red-500' :
                log.includes('Mean reward') ? 'text-green-600' :
                log.includes('Learning iteration') ? 'text-blue-600 font-semibold' :
                'text-foreground'
              }`}
            >
              {log}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
