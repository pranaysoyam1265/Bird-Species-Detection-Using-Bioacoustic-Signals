"use client"

import { useEffect, useState } from "react"

interface AnalysisProgressProps {
  totalChunks: number
  chunkDuration: number
  onComplete: () => void
}

// Animated equalizer bars
function EqualizerBars() {
  return (
    <div className="flex items-end gap-[3px] h-6">
      {[1, 2, 3, 4, 5].map((i) => (
        <div
          key={i}
          className="w-1.5 bg-accent rounded-sm"
          style={{
            animation: `eq-bar ${0.4 + i * 0.1}s ease-in-out infinite alternate`,
            animationDelay: `${i * 0.07}s`,
          }}
        />
      ))}
      <style>{`
        @keyframes eq-bar {
          from { height: 4px; opacity: 0.4; }
          to   { height: 24px; opacity: 1; }
        }
      `}</style>
    </div>
  )
}

export function AnalysisProgress({ totalChunks, chunkDuration, onComplete }: AnalysisProgressProps) {
  const [currentChunk, setCurrentChunk] = useState(0)
  const [speciesFound, setSpeciesFound] = useState(0)
  const [startTime] = useState(Date.now())
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentChunk((prev) => {
        const next = prev + 1
        if (next >= totalChunks) {
          clearInterval(interval)
          setTimeout(onComplete, 300)
          return totalChunks
        }
        if (next === 2 || next === 5 || next === 8) {
          setSpeciesFound((s) => s + 1)
        }
        return next
      })
      setElapsed(Date.now() - startTime)
    }, 180)
    return () => clearInterval(interval)
  }, [totalChunks, onComplete, startTime])

  const progress = totalChunks > 0 ? Math.round((currentChunk / totalChunks) * 100) : 0
  const currentStart = currentChunk * chunkDuration
  const currentEnd = Math.min((currentChunk + 1) * chunkDuration, totalChunks * chunkDuration)
  const remaining = totalChunks > 0 && currentChunk > 0
    ? Math.round(((totalChunks - currentChunk) / currentChunk) * (elapsed / 1000))
    : 0

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }

  return (
    <div className="border-2 border-accent bg-background p-6 space-y-4 animate-pulse-border relative overflow-hidden">
      {/* Pulsing orange border glow */}
      <div className="absolute inset-0 border-2 border-accent opacity-30 animate-ping pointer-events-none rounded-none" />

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <EqualizerBars />
          <span className="font-mono text-xs tracking-[0.2em] uppercase text-foreground font-bold">
            ANALYZING AUDIO
          </span>
        </div>
        <span className="font-mono text-sm tracking-wider text-accent font-bold">
          {progress}%
        </span>
      </div>

      {/* Chunk progress label */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
          CHUNK {Math.min(currentChunk + 1, totalChunks)} / {totalChunks}
        </span>
        <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
          ~{remaining}s REMAINING
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-full h-3 border-2 border-foreground bg-background overflow-hidden">
        <div
          className="h-full bg-accent transition-all duration-150 relative"
          style={{ width: `${progress}%` }}
        >
          {/* Shimmer on bar */}
          <div className="absolute inset-0 bg-white/20 animate-shimmer" />
        </div>
      </div>

      {/* Details */}
      <div className="space-y-1.5">
        <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground flex items-center gap-2">
          <span className="text-accent">⟩</span>
          Segment: {fmt(currentStart)} – {fmt(currentEnd)}
        </p>
        <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground flex items-center gap-2">
          <span className="text-[#22c55e]">⟩</span>
          Species detected: <span className="text-[#22c55e] font-bold">{speciesFound}</span>
        </p>
      </div>

      <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground animate-blink">
        PROCESSING AUDIO SEGMENTS...
      </p>
    </div>
  )
}
