"use client"

import { useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"

interface Prediction {
  species: string
  confidence: number
}

interface PredictionsChartProps {
  predictions: Prediction[]
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 80) return "#22c55e"   // green — strong
  if (confidence >= 50) return "#eab308"   // yellow — moderate
  if (confidence >= 20) return "var(--accent-hex)"   // orange — low
  return "#ef4444"                          // red — very low
}

export function PredictionsChart({ predictions }: PredictionsChartProps) {
  const [open, setOpen] = useState(true)
  const maxConf = Math.max(...predictions.map(p => p.confidence), 1)

  return (
    <div className="border-2 border-foreground border-l-4 border-l-accent bg-gradient-to-br from-accent/5 to-background">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full border-b-2 border-foreground px-4 py-2 flex items-center justify-between cursor-pointer hover:bg-muted transition-none bg-transparent"
      >
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          TOP PREDICTIONS
        </span>
        {open ? <ChevronUp size={14} className="text-muted-foreground" /> : <ChevronDown size={14} className="text-muted-foreground" />}
      </button>

      {open && (
        <div className="p-4 space-y-3">
          {predictions.map((p, i) => {
            const color = getConfidenceColor(p.confidence)
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="font-mono text-[10px] tracking-wider uppercase text-foreground w-[140px] sm:w-[180px] truncate shrink-0 text-right">
                  {p.species}
                </span>
                <div className="flex-1 h-5 bg-foreground/10 overflow-hidden">
                  <div
                    className="h-full transition-all duration-700 ease-out"
                    style={{ width: `${(p.confidence / maxConf) * 100}%`, backgroundColor: color }}
                  />
                </div>
                <span className="font-mono text-[10px] tracking-wider w-[50px] shrink-0 font-bold" style={{ color }}>
                  {p.confidence.toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
