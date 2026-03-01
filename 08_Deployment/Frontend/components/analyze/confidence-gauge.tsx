"use client"

import { useRef, useEffect } from "react"

interface ConfidenceGaugeProps {
  confidence: number
}

export function ConfidenceGauge({ confidence }: ConfidenceGaugeProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const getLabel = (v: number) => {
    if (v >= 85) return "VERY HIGH"
    if (v >= 70) return "HIGH"
    if (v >= 40) return "MEDIUM"
    return "LOW"
  }

  const getColor = (v: number) => {
    if (v >= 70) return "#22c55e"
    if (v >= 40) return "#eab308"
    return "#ef4444"
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const size = 200
    canvas.width = size * dpr
    canvas.height = (size * 0.65) * dpr
    ctx.scale(dpr, dpr)

    const W = size
    const H = size * 0.65
    const cx = W / 2
    const cy = H - 10
    const r = 80
    const lineW = 12

    ctx.clearRect(0, 0, W, H)

    // Background arc
    ctx.beginPath()
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI)
    ctx.strokeStyle = "hsl(0 0% 20%)"
    ctx.lineWidth = lineW
    ctx.stroke()

    // Colored arc (animated via confidence prop)
    const endAngle = Math.PI + (confidence / 100) * Math.PI
    const color = getColor(confidence)

    ctx.beginPath()
    ctx.arc(cx, cy, r, Math.PI, endAngle)
    ctx.strokeStyle = color
    ctx.lineWidth = lineW
    ctx.lineCap = "butt"
    ctx.stroke()

    // Tick marks at 0%, 25%, 50%, 75%, 100%
    for (let i = 0; i <= 4; i++) {
      const angle = Math.PI + (i / 4) * Math.PI
      const innerR = r - lineW / 2 - 4
      const outerR = r - lineW / 2 - 12
      const x1 = cx + Math.cos(angle) * innerR
      const y1 = cy + Math.sin(angle) * innerR
      const x2 = cx + Math.cos(angle) * outerR
      const y2 = cy + Math.sin(angle) * outerR
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.strokeStyle = "hsl(0 0% 40%)"
      ctx.lineWidth = 1.5
      ctx.stroke()
    }

    // Center percentage text
    ctx.fillStyle = color
    ctx.font = "bold 22px monospace"
    ctx.textAlign = "center"
    ctx.textBaseline = "bottom"
    ctx.fillText(`${confidence.toFixed(1)}%`, cx, cy - 8)

    // Label
    ctx.fillStyle = "hsl(0 0% 60%)"
    ctx.font = "bold 9px monospace"
    ctx.fillText(getLabel(confidence), cx, cy + 4)

    // 0% and 100% labels
    ctx.fillStyle = "hsl(0 0% 40%)"
    ctx.font = "8px monospace"
    ctx.textAlign = "left"
    ctx.fillText("0%", cx - r - lineW, cy + 4)
    ctx.textAlign = "right"
    ctx.fillText("100%", cx + r + lineW, cy + 4)

  }, [confidence])

  return (
    <div className="flex items-center justify-center">
      <canvas
        ref={canvasRef}
        className="w-[200px] h-[130px]"
        style={{ imageRendering: "auto" }}
      />
    </div>
  )
}
