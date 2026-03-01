"use client"

import { useRef, useEffect, useMemo } from "react"

interface ConfidenceHistogramProps {
  confidences: number[]
  topConfidence: number
}

export function ConfidenceHistogram({ confidences, topConfidence }: ConfidenceHistogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const binLabels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

  // Count confidences into bins
  const { counts, topBin, maxCount, total, avg, median } = useMemo(() => {
    const c = new Array(5).fill(0)
    confidences.forEach((v) => {
      const idx = Math.min(Math.floor(v / 20), 4)
      c[idx]++
    })
    const tb = Math.min(Math.floor(topConfidence / 20), 4)
    const mx = Math.max(...c, 1)
    const tot = confidences.length
    const av = tot > 0 ? confidences.reduce((a, b) => a + b, 0) / tot : 0
    const sorted = [...confidences].sort((a, b) => a - b)
    const med = tot > 0 ? sorted[Math.floor(tot / 2)] : 0
    return { counts: c, topBin: tb, maxCount: mx, total: tot, avg: av, median: med }
  }, [confidences, topConfidence])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const W = rect.width
    const H = rect.height
    const pad = { top: 12, bottom: 28, left: 30, right: 12 }
    const plotW = W - pad.left - pad.right
    const plotH = H - pad.top - pad.bottom
    const gap = 8
    const barW = (plotW - gap * 4) / 5

    ctx.clearRect(0, 0, W, H)

    // Y-axis gridlines & labels
    const ySteps = Math.min(maxCount, 5)
    for (let i = 0; i <= ySteps; i++) {
      const val = Math.round((maxCount / ySteps) * i)
      const y = pad.top + plotH - (val / maxCount) * plotH
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(W - pad.right, y)
      ctx.strokeStyle = "hsl(0 0% 18%)"
      ctx.lineWidth = 0.5
      ctx.stroke()

      ctx.fillStyle = "hsl(0 0% 45%)"
      ctx.font = "9px monospace"
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      ctx.fillText(`${val}`, pad.left - 6, y)
    }

    // Bars
    for (let i = 0; i < 5; i++) {
      const x = pad.left + i * (barW + gap)
      const h = counts[i] > 0 ? (counts[i] / maxCount) * plotH : 0
      const y = pad.top + plotH - h

      // Bar fill
      ctx.fillStyle = i === topBin ? "var(--accent-hex)" : "hsl(0 0% 25%)"
      ctx.fillRect(x, y, barW, h)

      // Bar border
      if (h > 0) {
        ctx.strokeStyle = i === topBin ? "#f97316" : "hsl(0 0% 40%)"
        ctx.lineWidth = 1
        ctx.strokeRect(x, y, barW, h)
      }

      // Count label on top of bar
      if (counts[i] > 0) {
        ctx.fillStyle = i === topBin ? "var(--accent-hex)" : "hsl(0 0% 60%)"
        ctx.font = "bold 10px monospace"
        ctx.textAlign = "center"
        ctx.textBaseline = "bottom"
        ctx.fillText(`${counts[i]}`, x + barW / 2, y - 3)
      }

      // X-axis label
      ctx.fillStyle = i === topBin ? "var(--accent-hex)" : "hsl(0 0% 50%)"
      ctx.font = "8px monospace"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText(binLabels[i], x + barW / 2, pad.top + plotH + 8)
    }

  }, [confidences, topConfidence, counts, maxCount, topBin, binLabels])

  return (
    <div className="border-2 border-foreground border-l-4 border-l-[#3b82f6] bg-gradient-to-br from-[#3b82f6]/5 to-background h-full flex flex-col">
      <div className="border-b-2 border-foreground px-4 py-2">
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          CONFIDENCE DISTRIBUTION
        </span>
      </div>
      <div className="p-4 flex-1 flex flex-col gap-4">
        {/* Chart */}
        <canvas
          ref={canvasRef}
          className="w-full flex-1 min-h-[180px]"
        />

        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2 border-t border-foreground/20 pt-3">
          <div className="text-center">
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-muted-foreground block">
              TOTAL
            </span>
            <span className="font-mono text-base font-bold text-foreground">
              {total}
            </span>
          </div>
          <div className="text-center">
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-muted-foreground block">
              AVG CONF
            </span>
            <span className="font-mono text-base font-bold text-accent">
              {avg.toFixed(1)}%
            </span>
          </div>
          <div className="text-center">
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-muted-foreground block">
              MEDIAN
            </span>
            <span className="font-mono text-base font-bold text-foreground">
              {median.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
