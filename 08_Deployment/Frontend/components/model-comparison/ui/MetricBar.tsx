"use client"

import { motion } from "framer-motion"

export function MetricBar({
  label,
  value,
  maxValue,
  displayValue,
  isWinner,
  isLoser,
  delay = 0,
}: {
  label: string
  value: number
  maxValue: number
  displayValue: string
  isWinner?: boolean
  isLoser?: boolean
  delay?: number
}) {
  const pct = (value / maxValue) * 100

  return (
    <div className="flex items-center gap-2 sm:gap-4">
      <span className="w-20 sm:w-28 font-mono text-[8px] sm:text-[10px] tracking-wider uppercase text-muted-foreground truncate">
        {label}
      </span>
      <div className="flex-1 h-3 sm:h-4 bg-foreground/10 border border-foreground/20 overflow-hidden">
        <motion.div
          className={`h-full ${isLoser ? "bg-red-500/50" : "bg-accent"}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, delay, ease: "easeOut" }}
        />
      </div>
      <span className="w-14 sm:w-20 font-mono text-[9px] sm:text-xs tracking-wider text-foreground text-right font-bold">
        {displayValue}
      </span>
      {isWinner && <span className="text-yellow-400 text-xs">⭐</span>}
      {isLoser && <span className="text-red-500 text-xs">✕</span>}
    </div>
  )
}
