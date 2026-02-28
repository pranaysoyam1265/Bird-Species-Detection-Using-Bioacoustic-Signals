"use client"

import { useEffect, useRef } from "react"
import { useTheme } from "next-themes"

// 4x4 Bayer threshold matrix
const BAYER_4X4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
]

function bayer(x: number, y: number): number {
  return BAYER_4X4[y % 4][x % 4] / 16
}

export function DitherCard() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { resolvedTheme } = useTheme()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const W = 480
    const H = 320
    canvas.width = W
    canvas.height = H

    const isDark = resolvedTheme === "dark"
    const bg = isDark ? 20 : 235
    const fg = isDark ? 240 : 10

    // --- Draw bird silhouette into an offscreen canvas ---
    const off = document.createElement("canvas")
    off.width = W
    off.height = H
    const oc = off.getContext("2d")!

    // Background gradient (gives the dithered noise texture)
    const grad = oc.createRadialGradient(W / 2, H / 2, 10, W / 2, H / 2, W * 0.65)
    grad.addColorStop(0, isDark ? "#555" : "#bbb")
    grad.addColorStop(1, isDark ? "#111" : "#f0f0f0")
    oc.fillStyle = grad
    oc.fillRect(0, 0, W, H)

    // Bird silhouette (centered, wings-up shape)
    const cx = W / 2
    const cy = H / 2 - 10
    const sc = 1.15
    oc.save()
    oc.translate(cx, cy)
    oc.scale(sc, sc)

    // Main body
    oc.beginPath()
    oc.ellipse(0, 20, 38, 55, 0, 0, Math.PI * 2)
    oc.fillStyle = isDark ? "#ddd" : "#fff"
    oc.fill()

    // Left wing (sweeping up-left)
    oc.beginPath()
    oc.moveTo(-10, -10)
    oc.bezierCurveTo(-60, -60, -130, -40, -155, 10)
    oc.bezierCurveTo(-130, 30, -70, 20, -10, 30)
    oc.closePath()
    oc.fill()

    // Right wing (sweeping up-right)
    oc.beginPath()
    oc.moveTo(10, -10)
    oc.bezierCurveTo(60, -60, 130, -40, 155, 10)
    oc.bezierCurveTo(130, 30, 70, 20, 10, 30)
    oc.closePath()
    oc.fill()

    // Head
    oc.beginPath()
    oc.ellipse(0, -42, 26, 28, 0, 0, Math.PI * 2)
    oc.fill()

    // Beak
    oc.beginPath()
    oc.moveTo(8, -48)
    oc.lineTo(28, -44)
    oc.lineTo(8, -38)
    oc.closePath()
    oc.fillStyle = isDark ? "#aaa" : "#ccc"
    oc.fill()

    // Tail feathers (bottom)
    oc.beginPath()
    oc.moveTo(-18, 65)
    oc.lineTo(-30, 110)
    oc.lineTo(-10, 78)
    oc.lineTo(0, 115)
    oc.lineTo(10, 78)
    oc.lineTo(30, 110)
    oc.lineTo(18, 65)
    oc.closePath()
    oc.fillStyle = isDark ? "#ddd" : "#fff"
    oc.fill()

    oc.restore()

    // --- Bayer dither the offscreen canvas into the main canvas ---
    const src = oc.getImageData(0, 0, W, H)
    const out = ctx.createImageData(W, H)

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4
        const lum = (src.data[i] * 0.299 + src.data[i + 1] * 0.587 + src.data[i + 2] * 0.114) / 255
        const threshold = bayer(x, y)
        const val = lum > threshold ? fg : bg
        out.data[i] = val
        out.data[i + 1] = val
        out.data[i + 2] = val
        out.data[i + 3] = 255
      }
    }

    ctx.putImageData(out, 0, 0)
  }, [resolvedTheme])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between border-b-2 border-foreground px-4 py-2">
        <span className="text-[10px] tracking-widest text-muted-foreground uppercase">
          birdsense.detect
        </span>
        <span className="text-[10px] tracking-widest text-muted-foreground">dither_4x4</span>
      </div>
      <div className="flex-1 flex items-center justify-center p-4 bg-background overflow-hidden">
        <canvas
          ref={canvasRef}
          className="w-full h-auto"
          style={{ imageRendering: "pixelated" }}
          aria-label="Bayer dithered bird visualization for BirdSense"
          role="img"
        />
      </div>
    </div>
  )
}
