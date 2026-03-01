"use client"

import { useState } from "react"
import { Clock, Radio, Star, Activity, ChevronDown } from "lucide-react"

interface RecordingInfoProps {
  duration: number
  sampleRate: number
}

export function RecordingInfo({ duration, sampleRate }: RecordingInfoProps) {
  const [open, setOpen] = useState(false)

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`
  }

  const quality = sampleRate >= 44100 ? "EXCELLENT" : sampleRate >= 22050 ? "GOOD" : "FAIR"
  const qualityColor = quality === "EXCELLENT" ? "#22c55e" : quality === "GOOD" ? "var(--accent-hex)" : "#eab308"

  // Mock SNR
  const snr = (18.5 + Math.random() * 4).toFixed(1)

  const stats = [
    {
      label: "DURATION",
      value: fmt(duration),
      icon: <Clock size={12} />,
      accent: "#3b82f6",
    },
    {
      label: "SAMPLE RATE",
      value: `${(sampleRate / 1000).toFixed(2)}kHz`,
      icon: <Radio size={12} />,
      accent: "#22c55e",
    },
    {
      label: "QUALITY",
      value: quality,
      icon: <Star size={12} />,
      accent: qualityColor,
    },
    {
      label: "SNR",
      value: `${snr} dB`,
      icon: <Activity size={12} />,
      accent: "#a855f7",
    },
  ]

  return (
    <div className="border-2 border-foreground bg-background">
      {/* Clickable header — always visible */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 cursor-pointer hover:bg-muted/30 transition-none"
      >
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
            RECORDING INFO
          </span>
          {/* Compact inline summary when collapsed */}
          {!open && (
            <span className="font-mono text-[10px] tracking-wider text-muted-foreground hidden sm:inline-flex items-center gap-2">
              <span className="text-foreground font-bold">{fmt(duration)}</span>
              <span className="text-foreground/30">•</span>
              <span className="text-foreground font-bold">{(sampleRate / 1000).toFixed(1)}kHz</span>
              <span className="text-foreground/30">•</span>
              <span style={{ color: qualityColor }} className="font-bold">{quality}</span>
            </span>
          )}
        </div>
        <ChevronDown
          size={14}
          className={`text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {/* Expandable stats grid */}
      {open && (
        <div className="grid grid-cols-2 sm:grid-cols-4 border-t-2 border-foreground">
          {stats.map((stat, i) => (
            <div
              key={i}
              className={`p-3 sm:p-4 relative group hover:bg-muted/30 transition-colors ${i < stats.length - 1 ? "border-b-2 sm:border-b-0 sm:border-r-2 border-foreground" : ""
                }`}
            >
              {/* Colored top micro-bar */}
              <div className="absolute top-0 left-0 right-0 h-[3px]" style={{ backgroundColor: stat.accent, opacity: 0.7 }} />
              <span
                className="font-mono text-xs tracking-[0.25em] uppercase font-bold flex items-center gap-1.5 mb-1"
                style={{ color: stat.accent }}
              >
                {stat.icon}
                {stat.label}
              </span>
              <span className="font-mono text-sm font-bold text-foreground">
                {stat.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
