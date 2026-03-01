"use client"

import { useRef, useEffect } from "react"
import { Info } from "lucide-react"

interface TopResultCardProps {
  species: string
  scientificName: string
  confidence: number
  onSpeciesClick?: (species: string) => void
}

function getConfidenceLabel(c: number) {
  if (c >= 90) return "EXCELLENT"
  if (c >= 75) return "VERY HIGH"
  if (c >= 60) return "HIGH"
  if (c >= 40) return "MEDIUM"
  return "LOW"
}

export function TopResultCard({ species, scientificName, confidence, onSpeciesClick }: TopResultCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const size = 140
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`
    ctx.scale(dpr, dpr)

    const cx = size / 2
    const cy = size / 2
    const maxR = 60
    let angle = 0

    // Blip position based on confidence (higher = closer to center)
    const blipDist = maxR * (1 - confidence / 100) * 0.8 + maxR * 0.1
    const blipAngle = (confidence * 3.7 + 45) * (Math.PI / 180) // deterministic angle

    function draw() {
      ctx.clearRect(0, 0, size, size)

      // Concentric rings
      for (let i = 1; i <= 4; i++) {
        const r = (maxR / 4) * i
        ctx.beginPath()
        ctx.arc(cx, cy, r, 0, Math.PI * 2)
        ctx.strokeStyle = `rgba(234, 88, 12, ${i === 4 ? 0.3 : 0.12})`
        ctx.lineWidth = i === 4 ? 1.5 : 0.5
        ctx.stroke()
      }

      // Crosshairs
      ctx.beginPath()
      ctx.moveTo(cx - maxR, cy)
      ctx.lineTo(cx + maxR, cy)
      ctx.moveTo(cx, cy - maxR)
      ctx.lineTo(cx, cy + maxR)
      ctx.strokeStyle = "rgba(234, 88, 12, 0.08)"
      ctx.lineWidth = 0.5
      ctx.stroke()

      // Sweep line
      const sweepX = cx + Math.cos(angle) * maxR
      const sweepY = cy + Math.sin(angle) * maxR
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(sweepX, sweepY)
      ctx.strokeStyle = "rgba(234, 88, 12, 0.6)"
      ctx.lineWidth = 1.5
      ctx.stroke()

      // Sweep trail (fading arc)
      const trailLen = 0.8
      const grad = ctx.createConicGradient(angle - trailLen, cx, cy)
      grad.addColorStop(0, "rgba(234, 88, 12, 0)")
      grad.addColorStop(trailLen / (Math.PI * 2), "rgba(234, 88, 12, 0.15)")
      grad.addColorStop(trailLen / (Math.PI * 2) + 0.001, "rgba(234, 88, 12, 0)")
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.arc(cx, cy, maxR, angle - trailLen, angle)
      ctx.closePath()
      ctx.fillStyle = grad
      ctx.fill()

      // Detection blip — pulsing
      const bx = cx + Math.cos(blipAngle) * blipDist
      const by = cy + Math.sin(blipAngle) * blipDist
      const pulse = 1 + Math.sin(Date.now() * 0.004) * 0.3

      // Glow
      const glow = ctx.createRadialGradient(bx, by, 0, bx, by, 10 * pulse)
      glow.addColorStop(0, "rgba(234, 88, 12, 0.5)")
      glow.addColorStop(1, "rgba(234, 88, 12, 0)")
      ctx.fillStyle = glow
      ctx.fillRect(bx - 12, by - 12, 24, 24)

      // Dot
      ctx.beginPath()
      ctx.arc(bx, by, 3 * pulse, 0, Math.PI * 2)
      ctx.fillStyle = "var(--accent-hex)"
      ctx.fill()

      // Center dot
      ctx.beginPath()
      ctx.arc(cx, cy, 2, 0, Math.PI * 2)
      ctx.fillStyle = "rgba(234, 88, 12, 0.4)"
      ctx.fill()

      angle += 0.02
      animRef.current = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animRef.current)
  }, [confidence])

  return (
    <div className="border-2 border-foreground border-l-4 border-l-accent bg-gradient-to-br from-accent/8 to-background h-full">
      <div className="border-b-2 border-foreground px-4 py-2">
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          TOP DETECTION
        </span>
      </div>
      <div className="p-6 sm:p-8 flex flex-col items-center text-center gap-4">
        {/* Species name */}
        <div>
          <button
            type="button"
            onClick={() => onSpeciesClick?.(species)}
            className="font-mono text-lg sm:text-xl font-bold tracking-[0.1em] uppercase text-foreground hover:text-accent cursor-pointer bg-transparent border-none p-0 flex items-center gap-2 justify-center"
            style={{ textShadow: "0 0 20px rgba(234,88,12,0.3)" }}
          >
            {species}
            <Info size={14} className="text-muted-foreground" />
          </button>
          <p className="font-mono text-[10px] tracking-[0.15em] text-accent/60 italic mt-2 px-2 py-0.5 border border-accent/20 bg-accent/5 inline-block">
            {scientificName}
          </p>
        </div>

        {/* Sonar radar visualization */}
        <div className="relative">
          <canvas ref={canvasRef} />
          <span className="absolute bottom-[-8px] left-1/2 -translate-x-1/2 font-mono text-[8px] tracking-[0.25em] uppercase text-muted-foreground">
            {getConfidenceLabel(confidence)}
          </span>
        </div>

        {/* Confidence bar */}
        <div className="w-full max-w-[280px] space-y-2">
          <div className="w-full h-3 border-2 border-foreground bg-background overflow-hidden">
            <div
              className="h-full bg-accent transition-all duration-1000 ease-out"
              style={{ width: `${confidence}%` }}
            />
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="font-mono text-2xl font-bold text-accent">
              {confidence.toFixed(1)}%
            </span>
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground">
              CONFIDENCE
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
