"use client"

import { useRef, useEffect, useState } from "react"

export type ColormapName = "MAGMA" | "VIRIDIS" | "GRAYSCALE" | "PLASMA" | "INFERNO"

const COLORMAPS: Record<ColormapName, (v: number) => [number, number, number]> = {
  MAGMA: (v) => {
    if (v < 0.25) return [10 + v * 200, 10 + v * 40, 30 + v * 200]
    if (v < 0.5) return [60 + v * 300, 10 + v * 60, 80 + v * 200]
    if (v < 0.75) return [200 + v * 70, 50 + v * 100, 60 + v * 40]
    return [234, 88 + (v - 0.75) * 400, 12 + v * 200]
  },
  VIRIDIS: (v) => {
    if (v < 0.25) return [68 - v * 100, 1 + v * 100, 84 + v * 200]
    if (v < 0.5) return [30 + v * 40, 80 + v * 200, 130 - v * 40]
    if (v < 0.75) return [50 + v * 200, 160 + v * 100, 50 + v * 20]
    return [180 + v * 80, 220 + v * 30, 40 - v * 20]
  },
  GRAYSCALE: (v) => {
    const g = Math.floor(v * 255)
    return [g, g, g]
  },
  PLASMA: (v) => {
    if (v < 0.25) return [12 + v * 300, 0, 120 + v * 400]
    if (v < 0.5) return [150 + v * 150, 10 + v * 40, 180 - v * 100]
    if (v < 0.75) return [220 + v * 30, 60 + v * 200, 80 - v * 80]
    return [240, 200 + v * 60, 30 + v * 100]
  },
  INFERNO: (v) => {
    if (v < 0.2) return [10 + v * 100, 2 + v * 30, 20 + v * 100]
    if (v < 0.5) return [60 + v * 300, 10 + v * 40, 40 - v * 20]
    if (v < 0.75) return [200 + v * 50, 50 + v * 120, 10]
    return [234, 150 + (v - 0.75) * 400, 20 + v * 200]
  },
}

interface SpectrogramViewProps {
  audioBuffer: AudioBuffer | null
}

export function SpectrogramView({ audioBuffer }: SpectrogramViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [colormap, setColormap] = useState<ColormapName>("MAGMA")

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !audioBuffer) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const W = rect.width
    const H = rect.height
    const data = audioBuffer.getChannelData(0)

    const fftSize = 256
    const cols = Math.min(Math.floor(data.length / fftSize), Math.floor(W))
    const rows = fftSize / 2

    const colorFn = COLORMAPS[colormap]

    for (let col = 0; col < cols; col++) {
      const start = col * fftSize
      for (let row = 0; row < rows; row++) {
        let sum = 0
        const binSize = 4
        for (let k = 0; k < binSize; k++) {
          const idx = start + row * (fftSize / rows) + k
          if (idx < data.length) {
            sum += Math.abs(data[idx])
          }
        }
        const mag = Math.min(sum / binSize * 3, 1)
        const [r, g, b] = colorFn(mag)
        ctx.fillStyle = `rgb(${Math.floor(Math.min(r, 255))},${Math.floor(Math.min(g, 255))},${Math.floor(Math.min(b, 255))})`

        const x = (col / cols) * W
        const y = H - ((row + 1) / rows) * H
        const cellW = W / cols + 1
        const cellH = H / rows + 1
        ctx.fillRect(x, y, cellW, cellH)
      }
    }

    // Axis labels
    ctx.fillStyle = "hsl(0 0% 60%)"
    ctx.font = "9px monospace"
    const sr = audioBuffer.sampleRate
    const maxFreq = sr / 2

    ctx.fillText(`${(maxFreq / 1000).toFixed(0)}kHz`, 2, 12)
    ctx.fillText(`${(maxFreq / 2000).toFixed(0)}kHz`, 2, H / 2)
    ctx.fillText("0Hz", 2, H - 4)

    const dur = audioBuffer.duration
    ctx.fillText("0s", 30, H - 4)
    ctx.fillText(`${(dur / 2).toFixed(0)}s`, W / 2, H - 4)
    ctx.fillText(`${dur.toFixed(0)}s`, W - 20, H - 4)

  }, [audioBuffer, colormap])

  return (
    <div className="border-2 border-foreground border-l-4 border-l-[#a855f7] bg-gradient-to-br from-[#a855f7]/5 to-background">
      <div className="border-b-2 border-foreground px-4 py-2 flex items-center justify-between">
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          SPECTROGRAM
        </span>
        {/* Colormap toggle */}
        <div className="flex gap-1">
          {(Object.keys(COLORMAPS) as ColormapName[]).map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setColormap(name)}
              className={`px-3 py-1 font-mono text-[10px] tracking-wider uppercase border cursor-pointer transition-none ${colormap === name
                ? "border-accent bg-accent text-white font-bold"
                : "border-foreground/30 text-muted-foreground hover:border-foreground hover:text-foreground"
                }`}
            >
              {name}
            </button>
          ))}
        </div>
      </div>
      <div className="p-4">
        {audioBuffer ? (
          <canvas
            ref={canvasRef}
            className="w-full h-[200px] sm:h-[260px]"
            style={{ imageRendering: "pixelated" }}
          />
        ) : (
          <div className="w-full h-[140px] flex items-center justify-center">
            <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground/40">
              NO DATA
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
