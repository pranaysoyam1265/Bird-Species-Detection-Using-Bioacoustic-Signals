"use client"

import { useEffect, useState, useRef } from "react"
import { motion, useInView } from "framer-motion"
// Canvas-based spectrogram visualization

const ease = [0.22, 1, 0.36, 1] as const

/* ── scramble text reveal ── */
function ScrambleText({ text, className }: { text: string; className?: string }) {
  const [display, setDisplay] = useState(text)
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: "-50px" })
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:"

  useEffect(() => {
    if (!inView) return
    let iteration = 0
    const interval = setInterval(() => {
      setDisplay(
        text
          .split("")
          .map((char, i) => {
            if (char === " ") return " "
            if (i < iteration) return text[i]
            return chars[Math.floor(Math.random() * chars.length)]
          })
          .join("")
      )
      iteration += 0.5
      if (iteration >= text.length) {
        setDisplay(text)
        clearInterval(interval)
      }
    }, 30)
    return () => clearInterval(interval)
  }, [inView, text])

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  )
}

/* ── blinking cursor ── */
function BlinkDot() {
  return <span className="inline-block h-2 w-2 bg-accent animate-blink" />
}

/* ── live analysis counter ── */
function AnalysisCounter() {
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    const base = 2847561 + Math.floor(Math.random() * 10000)
    setSeconds(base)
    const interval = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(interval)
  }, [])

  const format = (n: number) => {
    return n.toLocaleString()
  }

  return (
    <span className="font-mono text-accent" style={{ fontVariantNumeric: "tabular-nums" }}>
      {format(seconds)} samples analyzed
    </span>
  )
}

/* ── stat block ── */
const STATS = [
  { label: "SPECIES_DETECTED", value: "87" },
  { label: "TEST_ACCURACY", value: "96.06%" },
  { label: "AUDIO_SAMPLES", value: "2.8M+" },
  { label: "TOP_5_ACCURACY", value: "98.74%" },
]

function StatBlock({ label, value, index }: { label: string; value: string; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-30px" }}
      transition={{ delay: 0.15 + index * 0.08, duration: 0.5, ease }}
      className="flex flex-col gap-1 border-2 border-foreground px-4 py-3"
    >
      <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
        {label}
      </span>
      <span className="text-xl lg:text-2xl font-mono font-bold tracking-tight">
        <ScrambleText text={value} />
      </span>
    </motion.div>
  )
}

/* ── Magma-like colormap (7 stops) ── */
const MAGMA: [number, number, number][] = [
  [0, 0, 4],       // 0.0 – near black
  [30, 12, 70],     // ~0.17 – deep indigo
  [120, 28, 109],   // ~0.33 – purple
  [187, 55, 84],    // ~0.50 – magenta-red
  [234, 88, 12],    // ~0.67 – orange (#ea580c)
  [253, 190, 51],   // ~0.83 – amber
  [252, 253, 191],  // 1.0 – pale yellow
]

function magma(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t))
  const n = MAGMA.length - 1
  const idx = v * n
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, n)
  const f = idx - lo
  return [
    Math.round(MAGMA[lo][0] + (MAGMA[hi][0] - MAGMA[lo][0]) * f),
    Math.round(MAGMA[lo][1] + (MAGMA[hi][1] - MAGMA[lo][1]) * f),
    Math.round(MAGMA[lo][2] + (MAGMA[hi][2] - MAGMA[lo][2]) * f),
  ]
}

/* ── Seeded PRNG for deterministic noise ── */
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/*
 * A "syllable" is one continuous frequency trace.
 * freqCurve(t) returns the normalised frequency (0–1) at time t (0–1 within syllable).
 * This lets us draw swept arcs: ╭──╮ shapes.
 */
interface Syllable {
  tStart: number // absolute normalised time 0–1
  tEnd: number
  freqCurve: (t: number) => number // returns freq 0–1 at local t
  amplitude: number
  harmonicCount: number
  harmonicSpacing: number // in normalised freq units
}

/* ── Realistic mel-spectrogram ── */
function SpectrogramCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return

    const rect = parent.getBoundingClientRect()
    const W = Math.floor(rect.width) || 600
    const H = Math.floor(rect.height) || 300
    canvas.width = W
    canvas.height = H

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const rng = mulberry32(42)

    // ── Pre-compute 2D intensity buffer ──
    const buf = new Float32Array(W * H)

    // 1) Noise floor — low freq ambient + faint broadband
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) {
        const freqN = 1 - y / H // 0 at bottom, 1 at top
        if (freqN < 0.12) {
          // Strong low-freq ambient noise
          buf[y * W + x] = (1 - freqN / 0.12) * 0.18 + rng() * 0.06
        } else {
          // Very faint broadband
          buf[y * W + x] = rng() * 0.012
        }
      }
    }

    // 2) Helper: arc frequency curve (╭──╮ shape)
    //    peak = top of arc, base = starting/ending freq
    const makeArc = (base: number, peak: number) =>
      (t: number) => base + (peak - base) * Math.sin(t * Math.PI)

    // Flat trace
    const makeFlat = (freq: number) => () => freq

    // Descending sweep
    const makeDescend = (hi: number, lo: number) =>
      (t: number) => hi + (lo - hi) * t

    // Ascending sweep
    const makeAscend = (lo: number, hi: number) =>
      (t: number) => lo + (hi - lo) * t

    // Warble / sine modulated
    const makeWarble = (center: number, depth: number, cycles: number) =>
      (t: number) => center + depth * Math.sin(t * Math.PI * 2 * cycles)

    // 3) Define syllables — the individual traces that make bird song
    const syllables: Syllable[] = [
      // ─── Species A: song with 3 repeated rising-falling arcs (╭──╮ ╭──╮ ╭──╮) ───
      { tStart: 0.02, tEnd: 0.07, freqCurve: makeArc(0.35, 0.62), amplitude: 0.95, harmonicCount: 2, harmonicSpacing: 0.15 },
      { tStart: 0.08, tEnd: 0.13, freqCurve: makeArc(0.35, 0.65), amplitude: 0.90, harmonicCount: 2, harmonicSpacing: 0.15 },
      { tStart: 0.14, tEnd: 0.19, freqCurve: makeArc(0.35, 0.60), amplitude: 0.85, harmonicCount: 2, harmonicSpacing: 0.15 },

      // ─── Species B: descending chirps (short, steep) ───
      { tStart: 0.23, tEnd: 0.26, freqCurve: makeDescend(0.75, 0.45), amplitude: 0.88, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.27, tEnd: 0.30, freqCurve: makeDescend(0.78, 0.48), amplitude: 0.85, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.31, tEnd: 0.34, freqCurve: makeDescend(0.73, 0.42), amplitude: 0.80, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.35, tEnd: 0.38, freqCurve: makeDescend(0.70, 0.40), amplitude: 0.75, harmonicCount: 1, harmonicSpacing: 0 },

      // ─── Species A response: 2 arcs, slightly higher ───
      { tStart: 0.42, tEnd: 0.48, freqCurve: makeArc(0.40, 0.68), amplitude: 0.85, harmonicCount: 2, harmonicSpacing: 0.14 },
      { tStart: 0.49, tEnd: 0.55, freqCurve: makeArc(0.40, 0.70), amplitude: 0.82, harmonicCount: 2, harmonicSpacing: 0.14 },

      // ─── Species C: rapid trill — many short flat notes ───
      { tStart: 0.58, tEnd: 0.595, freqCurve: makeFlat(0.72), amplitude: 0.70, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.60, tEnd: 0.615, freqCurve: makeFlat(0.73), amplitude: 0.72, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.62, tEnd: 0.635, freqCurve: makeFlat(0.71), amplitude: 0.68, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.64, tEnd: 0.655, freqCurve: makeFlat(0.74), amplitude: 0.72, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.66, tEnd: 0.675, freqCurve: makeFlat(0.72), amplitude: 0.70, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.68, tEnd: 0.695, freqCurve: makeFlat(0.73), amplitude: 0.68, harmonicCount: 1, harmonicSpacing: 0 },
      { tStart: 0.70, tEnd: 0.715, freqCurve: makeFlat(0.71), amplitude: 0.65, harmonicCount: 1, harmonicSpacing: 0 },

      // ─── Species D: low warble with harmonics ───
      { tStart: 0.74, tEnd: 0.82, freqCurve: makeWarble(0.30, 0.06, 3), amplitude: 0.80, harmonicCount: 3, harmonicSpacing: 0.12 },

      // ─── Species A: final song — 3 arcs ───
      { tStart: 0.85, tEnd: 0.89, freqCurve: makeArc(0.33, 0.60), amplitude: 0.90, harmonicCount: 2, harmonicSpacing: 0.15 },
      { tStart: 0.90, tEnd: 0.94, freqCurve: makeArc(0.33, 0.63), amplitude: 0.88, harmonicCount: 2, harmonicSpacing: 0.15 },
      { tStart: 0.95, tEnd: 0.99, freqCurve: makeArc(0.33, 0.58), amplitude: 0.82, harmonicCount: 2, harmonicSpacing: 0.15 },
    ]

    // 4) Render each syllable as a thin frequency contour trace
    const traceWidth = 3 // half-width in pixels for the trace thickness

    for (const syl of syllables) {
      const xStart = Math.floor(syl.tStart * W)
      const xEnd = Math.floor(syl.tEnd * W)
      if (xEnd <= xStart) continue

      for (let sx = xStart; sx < xEnd; sx++) {
        const localT = (sx - xStart) / (xEnd - xStart)

        // Amplitude envelope: quick attack, sustained, quick decay
        const attackEnd = 0.08
        const decayStart = 0.85
        let env = 1
        if (localT < attackEnd) env = localT / attackEnd
        else if (localT > decayStart) env = (1 - localT) / (1 - decayStart)

        // Paint fundamental + harmonics
        for (let h = 0; h < syl.harmonicCount; h++) {
          const freq = syl.freqCurve(localT) + h * syl.harmonicSpacing
          if (freq < 0 || freq > 1) continue

          const harmonicAmp = syl.amplitude * env / (1 + h * 0.55)
          const yCenter = Math.floor((1 - freq) * H)

          // Narrow Gaussian trace (±traceWidth pixels)
          for (let dy = -traceWidth; dy <= traceWidth; dy++) {
            const yPixel = yCenter + dy
            if (yPixel < 0 || yPixel >= H) continue

            const normalizedDist = Math.abs(dy) / traceWidth
            const gaussian = Math.exp(-normalizedDist * normalizedDist * 3)
            const val = harmonicAmp * gaussian

            buf[yPixel * W + sx] = Math.min(1, buf[yPixel * W + sx] + val)
          }
        }
      }
    }

    // 5) Render to canvas with Magma colormap
    const imageData = ctx.createImageData(W, H)
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const intensity = buf[y * W + x]
        // Sharper gamma for high contrast traces
        const mapped = Math.pow(intensity, 0.5)
        const [r, g, b] = magma(mapped)
        const idx = (y * W + x) * 4
        imageData.data[idx] = r
        imageData.data[idx + 1] = g
        imageData.data[idx + 2] = b
        imageData.data[idx + 3] = 255
      }
    }
    ctx.putImageData(imageData, 0, 0)

    // 6) Subtle grid overlay
    ctx.globalAlpha = 0.06
    ctx.strokeStyle = "#ffffff"
    ctx.lineWidth = 1

    // Horizontal freq lines
    for (let i = 1; i < 5; i++) {
      const yLine = Math.floor((i / 5) * H)
      ctx.beginPath()
      ctx.moveTo(0, yLine)
      ctx.lineTo(W, yLine)
      ctx.stroke()
    }
    // Vertical time lines
    for (let i = 1; i < 5; i++) {
      const xLine = Math.floor((i / 5) * W)
      ctx.beginPath()
      ctx.moveTo(xLine, 0)
      ctx.lineTo(xLine, H)
      ctx.stroke()
    }

    // 7) Scan line effect
    ctx.globalAlpha = 0.025
    ctx.fillStyle = "#000000"
    for (let y = 0; y < H; y += 2) {
      ctx.fillRect(0, y, W, 1)
    }

    ctx.globalAlpha = 1
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ imageRendering: "auto" }}
      aria-label="Mel-spectrogram visualization of bird vocalizations showing frequency sweeps, harmonics, and call patterns"
      role="img"
    />
  )
}

/* ── main about section ── */
export function AboutSection() {
  return (
    <section className="w-full px-6 py-20 lg:px-12">
      {/* Section label */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5, ease }}
        className="flex items-center gap-4 mb-8"
      >
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          {"// SECTION: ABOUT_BIRDSENSE"}
        </span>
        <div className="flex-1 border-t border-border" />
        <BlinkDot />
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          005
        </span>
      </motion.div>

      {/* Two-column layout */}
      <div className="flex flex-col lg:flex-row gap-0 border-2 border-foreground">
        {/* Left: Image */}
        <motion.div
          initial={{ opacity: 0, x: -30, filter: "blur(6px)" }}
          whileInView={{ opacity: 1, x: 0, filter: "blur(0px)" }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.7, ease }}
          className="relative w-full lg:w-1/2 min-h-[300px] lg:min-h-[500px] border-b-2 lg:border-b-0 lg:border-r-2 border-foreground overflow-hidden bg-foreground"
        >
          {/* Image label overlay */}
          <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-2 bg-foreground/80 backdrop-blur-sm">
            <span className="text-[10px] tracking-[0.2em] uppercase text-background/60 font-mono">
              RENDER: spectrogram_analysis.mel
            </span>
            <span className="text-[10px] tracking-[0.2em] uppercase text-accent font-mono">
              LIVE
            </span>
          </div>

          {/* Live animated spectrogram canvas */}
          <SpectrogramCanvas />

          {/* Bottom image coordinates */}
          <div className="absolute bottom-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-2 bg-foreground/80 backdrop-blur-sm">
            <span className="text-[10px] tracking-[0.2em] uppercase text-background/40 font-mono">
              {"FREQ: 0-22kHz / MEL"}
            </span>
            <span className="text-[10px] tracking-[0.2em] uppercase text-background/40 font-mono">
              {"RES: 128 mel bands"}
            </span>
          </div>
        </motion.div>

        {/* Right: Content */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.7, delay: 0.1, ease }}
          className="flex flex-col w-full lg:w-1/2"
        >
          {/* Header bar */}
          <div className="flex items-center justify-between px-5 py-3 border-b-2 border-foreground">
            <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
              README.md
            </span>
            <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
              v1.0.0
            </span>
          </div>

          {/* Content body */}
          <div className="flex-1 flex flex-col justify-between px-5 py-6 lg:py-8">
            <div className="flex flex-col gap-6">
              <motion.h2
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-30px" }}
                transition={{ duration: 0.5, delay: 0.2, ease }}
                className="text-2xl lg:text-3xl font-mono font-bold tracking-tight uppercase text-balance"
              >
                AI built for
                <br />
                <span className="text-accent">bioacoustic intelligence</span>
              </motion.h2>

              <motion.div
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-30px" }}
                transition={{ delay: 0.3, duration: 0.5, ease }}
                className="flex flex-col gap-4"
              >
                <p className="text-xs lg:text-sm font-mono text-muted-foreground leading-relaxed">
                  BirdSense processes raw audio recordings through a deep learning
                  pipeline trained on 2.8M+ samples. Mel-spectrograms are extracted,
                  fed into a convolutional classifier, and matched against 87 North
                  American bird species with confidence scoring.
                </p>
                <p className="text-xs lg:text-sm font-mono text-muted-foreground leading-relaxed">
                  Built by researchers studying avian bioacoustics and conservation.
                  Every detection is explainable, every confidence score is
                  calibrated, and every spectrogram is inspectable.
                </p>
              </motion.div>

              {/* Analysis counter line */}
              <motion.div
                initial={{ opacity: 0, scaleX: 0.8 }}
                whileInView={{ opacity: 1, scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4, duration: 0.5, ease }}
                style={{ transformOrigin: "left" }}
                className="flex items-center gap-3 py-3 border-t-2 border-b-2 border-foreground"
              >
                <span className="h-1.5 w-1.5 bg-accent" />
                <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
                  PROCESSED:
                </span>
                <AnalysisCounter />
              </motion.div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-0 mt-6">
              {STATS.map((stat, i) => (
                <StatBlock key={stat.label} {...stat} index={i} />
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
