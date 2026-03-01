"use client"

import { useState } from "react"
import { Play, Square, ChevronDown, ChevronUp } from "lucide-react"

export interface DetectionSegment {
  startTime: number
  endTime: number
  confidence: number
  species: string
}

interface DetectionTimelineProps {
  segments: DetectionSegment[]
  totalDuration: number
  onPlaySegment?: (start: number, end: number) => void
  playingIndex?: number
}

const SPECIES_COLORS = [
  "var(--accent-hex)",
  "#3b82f6",
  "#22c55e",
  "#a855f7",
  "#f59e0b",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
]

function speciesColor(species: string, allSpecies: string[]): string {
  const idx = allSpecies.indexOf(species)
  return SPECIES_COLORS[idx % SPECIES_COLORS.length]
}

export function DetectionTimeline({ segments, totalDuration, onPlaySegment, playingIndex }: DetectionTimelineProps) {
  const [open, setOpen] = useState(false)

  const uniqueSpecies = Array.from(new Set(segments.map(s => s.species)))

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }

  return (
    <div className="border-2 border-foreground border-l-4 border-l-[#f59e0b] bg-gradient-to-br from-[#f59e0b]/5 to-background">
      {/* Collapsible header */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full border-b-2 border-foreground px-4 py-2.5 flex items-center justify-between cursor-pointer hover:bg-muted transition-none bg-transparent"
      >
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
            DETECTION TIMELINE
          </span>
          <span className="font-mono text-[10px] text-accent font-bold">{segments.length} segments</span>
          {/* Mini inline preview when collapsed */}
          {!open && segments.length > 0 && (
            <div className="hidden sm:flex items-center gap-1 ml-2">
              {uniqueSpecies.slice(0, 3).map((sp) => (
                <span key={sp} className="flex items-center gap-1 font-mono text-[9px] text-muted-foreground">
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: speciesColor(sp, uniqueSpecies) }} />
                  {sp.split(" ").pop()}
                </span>
              ))}
              {uniqueSpecies.length > 3 && (
                <span className="font-mono text-[9px] text-muted-foreground">+{uniqueSpecies.length - 3}</span>
              )}
            </div>
          )}
        </div>
        {open ? <ChevronUp size={14} className="text-muted-foreground" /> : <ChevronDown size={14} className="text-muted-foreground" />}
      </button>

      {open && (
        <div className="p-4 space-y-4">
          {/* ── Timeline bar ── */}
          <div>
            {/* Species color legend */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mb-2">
              {uniqueSpecies.map((sp) => (
                <span key={sp} className="flex items-center gap-1.5 font-mono text-[9px] tracking-wider uppercase text-muted-foreground">
                  <span className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: speciesColor(sp, uniqueSpecies) }} />
                  {sp}
                </span>
              ))}
            </div>

            {/* Track */}
            <div className="relative w-full h-10 bg-foreground/5 border border-foreground/15 overflow-hidden">
              {/* Tick marks */}
              {Array.from({ length: 11 }, (_, i) => (
                <div
                  key={i}
                  className="absolute top-0 bottom-0 w-px bg-foreground/[0.06]"
                  style={{ left: `${i * 10}%` }}
                />
              ))}

              {/* Segments */}
              {segments.map((seg, i) => {
                const left = (seg.startTime / totalDuration) * 100
                const width = ((seg.endTime - seg.startTime) / totalDuration) * 100
                const isPlaying = playingIndex === i
                const color = speciesColor(seg.species, uniqueSpecies)
                return (
                  <div
                    key={i}
                    onClick={() => onPlaySegment?.(seg.startTime, seg.endTime)}
                    className={`absolute top-1 bottom-1 cursor-pointer ${isPlaying ? "ring-1 ring-white animate-pulse" : "hover:brightness-125"}`}
                    style={{
                      left: `${left}%`,
                      width: `${Math.max(width, 2)}%`,
                      backgroundColor: `${color}${isPlaying ? "ff" : "bb"}`,
                      borderRadius: "2px",
                    }}
                    title={`${seg.species} · ${seg.confidence.toFixed(1)}% · ${fmt(seg.startTime)}–${fmt(seg.endTime)}`}
                  />
                )
              })}
            </div>

            {/* Time axis */}
            <div className="flex justify-between mt-1">
              {Array.from({ length: 6 }, (_, i) => {
                const t = (totalDuration / 5) * i
                return (
                  <span key={i} className="font-mono text-[9px] text-muted-foreground/60">
                    {fmt(t)}
                  </span>
                )
              })}
            </div>
          </div>

          {/* ── Segment list (compact) ── */}
          <div className="space-y-1">
            {segments.map((seg, i) => {
              const isPlaying = playingIndex === i
              const color = speciesColor(seg.species, uniqueSpecies)
              return (
                <button
                  key={i}
                  type="button"
                  onClick={() => onPlaySegment?.(seg.startTime, seg.endTime)}
                  className={`w-full flex items-center gap-3 px-3 py-2 border font-mono text-left cursor-pointer transition-none ${isPlaying
                      ? "bg-accent/10 border-accent"
                      : "border-foreground/10 hover:border-foreground/30 hover:bg-muted/30"
                    }`}
                >
                  {/* Play/Stop icon */}
                  <div className="w-6 h-6 border flex items-center justify-center shrink-0"
                    style={{ borderColor: isPlaying ? color : `${color}60` }}
                  >
                    {isPlaying ? <Square size={10} style={{ color }} /> : <Play size={10} style={{ color }} />}
                  </div>

                  {/* Species dot + name */}
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                  <span className="text-[10px] tracking-wider uppercase font-bold truncate flex-1">
                    {seg.species}
                  </span>

                  {/* Time range */}
                  <span className="text-[9px] tracking-wider text-muted-foreground shrink-0">
                    {fmt(seg.startTime)}–{fmt(seg.endTime)}
                  </span>

                  {/* Confidence */}
                  <span className="text-[10px] font-bold shrink-0 w-10 text-right" style={{ color }}>
                    {seg.confidence.toFixed(0)}%
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
