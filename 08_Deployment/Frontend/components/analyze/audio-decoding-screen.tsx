"use client"

import { useState, useEffect, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"

/* ─── Types ─── */
interface AudioDecodingScreenProps {
  file: File
}

/* ─── Helpers ─── */
function getFileFormat(file: File): string {
  const ext = file.name.split(".").pop()?.toUpperCase()
  if (ext) return ext
  if (file.type.includes("wav")) return "WAV"
  if (file.type.includes("mp3") || file.type.includes("mpeg")) return "MP3"
  if (file.type.includes("ogg")) return "OGG"
  if (file.type.includes("flac")) return "FLAC"
  return "AUDIO"
}

function getCodec(file: File): string {
  const ext = file.name.split(".").pop()?.toLowerCase()
  switch (ext) {
    case "wav": return "PCM"
    case "mp3": return "MPEG-1 L3"
    case "ogg": return "VORBIS"
    case "flac": return "FLAC"
    case "m4a": case "aac": return "AAC"
    default: return "AUTO"
  }
}

function estimateDuration(file: File): string {
  const ext = file.name.split(".").pop()?.toLowerCase()
  const sizeBytes = file.size
  let bitsPerSecond = 1411200
  if (ext === "mp3") bitsPerSecond = 320000
  if (ext === "ogg") bitsPerSecond = 256000
  if (ext === "flac") bitsPerSecond = 900000
  const seconds = Math.round((sizeBytes * 8) / bitsPerSecond)
  if (seconds < 60) return `~${seconds}s`
  const min = Math.floor(seconds / 60)
  const sec = seconds % 60
  return `~${min}m ${sec}s`
}

/* ─── Build log steps ─── */
interface LogStep {
  text: string
  duration: number
}

function buildSteps(file: File): LogStep[] {
  const sizeMB = (file.size / (1024 * 1024)).toFixed(2)
  const ext = getFileFormat(file)
  const codec = getCodec(file)
  const dur = estimateDuration(file)

  return [
    { text: `Loading file: ${file.name}`, duration: 600 },
    { text: `Format: ${ext} · ${sizeMB} MB · ${codec} · ${dur}`, duration: 500 },
    { text: "Validating audio headers...", duration: 800 },
    { text: "Decoding audio buffer...", duration: 1200 },
    { text: "Extracting waveform peaks...", duration: 900 },
    { text: "Computing spectrogram frames...", duration: 1000 },
    { text: "Preparing visualization data...", duration: 700 },
  ]
}

/* ─── Typing animation ─── */
function TypingText({ text, onDone }: { text: string; onDone: () => void }) {
  const [chars, setChars] = useState(0)
  const doneRef = useRef(false)

  useEffect(() => {
    doneRef.current = false
    setChars(0)
    const iv = setInterval(() => {
      setChars((c) => {
        if (c >= text.length) {
          clearInterval(iv)
          return c
        }
        return c + 1
      })
    }, 20)
    return () => clearInterval(iv)
  }, [text])

  useEffect(() => {
    if (chars > 0 && chars >= text.length && !doneRef.current) {
      doneRef.current = true
      onDone()
    }
  }, [chars, text.length, onDone])

  return (
    <>
      {text.slice(0, chars)}
      {chars < text.length && <span className="text-accent animate-pulse">▌</span>}
    </>
  )
}

/* ─── ASCII Progress bar ─── */
function AsciiProgressBar({ percent, label }: { percent: number; label: string }) {
  const totalBlocks = 30
  const filled = Math.round((percent / 100) * totalBlocks)
  const bar = "█".repeat(filled) + "░".repeat(totalBlocks - filled)

  return (
    <div className="flex items-center gap-3 font-mono text-[11px]">
      <span className="text-muted-foreground/50 select-none">$</span>
      <span className="text-foreground/40">[</span>
      <span className="text-accent tracking-[-0.05em]">{bar}</span>
      <span className="text-foreground/40">]</span>
      <span className="text-accent font-bold w-10 text-right">{percent}%</span>
      <span className="text-muted-foreground uppercase tracking-[0.15em] text-[9px]">{label}</span>
    </div>
  )
}

/* ─── Component ─── */
export function AudioDecodingScreen({ file }: AudioDecodingScreenProps) {
  const steps = useRef(buildSteps(file))
  const [completedLines, setCompletedLines] = useState<string[]>([])
  const [activeIndex, setActiveIndex] = useState(0)
  const [typing, setTyping] = useState(true)
  const logRef = useRef<HTMLDivElement>(null)

  const totalSteps = steps.current.length
  const progress = Math.min(Math.round(((activeIndex) / totalSteps) * 100), 100)
  const currentLabel = activeIndex < totalSteps
    ? steps.current[activeIndex].text.split("...")[0].split(":")[0]
    : "COMPLETE"

  function handleTypingDone() {
    setTyping(false)
    const step = steps.current[activeIndex]
    setTimeout(() => {
      setCompletedLines((prev) => [...prev, step.text])
      if (activeIndex < totalSteps - 1) {
        setActiveIndex((i) => i + 1)
        setTyping(true)
      }
    }, step.duration)
  }

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [completedLines, activeIndex])

  return (
    <div className="border-2 border-accent bg-background overflow-hidden relative">
      {/* Scan-line overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-10 opacity-[0.02]"
        style={{
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, hsl(var(--accent) / 0.08) 2px, hsl(var(--accent) / 0.08) 4px)",
        }}
      />

      {/* ── Title bar ── */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-foreground/10 bg-muted/10">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
          <div className="w-2.5 h-2.5 rounded-full bg-accent/50" />
        </div>
        <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground ml-1">
          birdsense — audio decoder
        </span>
      </div>

      {/* ── Terminal body ── */}
      <div ref={logRef} className="px-4 py-3 space-y-1 min-h-[140px] max-h-[260px] overflow-y-auto relative" style={{ scrollBehavior: "smooth" }}>
        {/* Completed lines */}
        <AnimatePresence>
          {completedLines.map((line, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-start gap-2 font-mono text-[11px] leading-relaxed"
            >
              <span className="text-muted-foreground/50 select-none shrink-0">$</span>
              <span className="text-muted-foreground">{line}</span>
              <span className="text-accent ml-auto shrink-0">✓</span>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Active typing line */}
        {activeIndex < totalSteps && (
          <div className="flex items-start gap-2 font-mono text-[11px] leading-relaxed">
            <span className="text-muted-foreground/50 select-none shrink-0">$</span>
            <span className="text-foreground">
              {typing ? (
                <TypingText
                  text={steps.current[activeIndex].text}
                  onDone={handleTypingDone}
                />
              ) : (
                <>
                  {steps.current[activeIndex].text}
                  <span className="ml-2 text-accent/70 animate-pulse text-[9px]">processing...</span>
                </>
              )}
            </span>
          </div>
        )}

        {/* All done */}
        {activeIndex >= totalSteps - 1 && completedLines.length >= totalSteps && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-start gap-2 font-mono text-[11px] leading-relaxed pt-1"
          >
            <span className="text-muted-foreground/50 select-none shrink-0">$</span>
            <span className="text-accent font-bold">Audio ready for analysis.</span>
          </motion.div>
        )}
      </div>

      {/* ── Progress bar ── */}
      <div className="px-4 py-2.5 border-t border-foreground/10 bg-muted/5">
        <AsciiProgressBar percent={progress} label={currentLabel} />
      </div>
    </div>
  )
}
