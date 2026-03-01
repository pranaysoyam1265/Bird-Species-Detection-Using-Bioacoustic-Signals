"use client"

import { useMemo, useState } from "react"
import { Check, X, Play, Pause, AlertTriangle, BarChart3, ChevronDown } from "lucide-react"
import { SPECIES_META } from "@/lib/species-meta"

// ── Helpers ──

function getAccuracyColor(v: number): string {
  if (v >= 80) return "rgb(34, 197, 94)"
  if (v >= 60) return "rgb(59, 130, 246)"
  if (v >= 40) return "rgb(234, 179, 8)"
  if (v >= 20) return "rgb(249, 115, 22)"
  return "rgb(239, 68, 68)"
}

// ── Types ──

interface Prediction {
  species: string
  confidence: number
}

interface Segment {
  species: string
  startTime: number
  endTime: number
  confidence: number
}

export interface SearchVerdictProps {
  targetSpecies: string
  predictions: Prediction[]
  segments: Segment[]
  totalDuration: number
  /** Sensitivity threshold (%) */
  sensitivity: number
  onPlaySegment: (start: number, end: number) => void
  playingIndex?: number
}

// ── Confidence Over Time Chart ──

function ConfidenceTimeline({
  segments,
  species,
  totalDuration,
  sensitivity,
}: {
  segments: Segment[]
  species: string
  totalDuration: number
  sensitivity: number
}) {
  const targetSegments = segments
    .filter((s) => s.species.toLowerCase() === species.toLowerCase())
    .sort((a, b) => a.startTime - b.startTime)

  // Build data points: one per segment, plus anchor points at 0 and totalDuration
  const points = useMemo(() => {
    if (targetSegments.length === 0) return []
    const pts: { time: number; conf: number }[] = []
    pts.push({ time: 0, conf: 0 })
    if (targetSegments[0].startTime > 0) {
      pts.push({ time: targetSegments[0].startTime - 0.1, conf: 0 })
    }
    targetSegments.forEach((seg) => {
      pts.push({ time: seg.startTime, conf: seg.confidence })
      pts.push({ time: seg.endTime, conf: seg.confidence })
      pts.push({ time: seg.endTime + 0.1, conf: 0 })
    })
    pts.push({ time: totalDuration, conf: 0 })
    return pts
  }, [targetSegments, totalDuration])

  if (points.length === 0) return null

  const chartW = 100
  const chartH = 60

  const pathD = points
    .map((p, i) => {
      const x = (p.time / totalDuration) * chartW
      const y = chartH - (p.conf / 100) * chartH
      return `${i === 0 ? "M" : "L"} ${x} ${y}`
    })
    .join(" ")

  const fillD = pathD + ` L ${chartW} ${chartH} L 0 ${chartH} Z`
  const threshY = chartH - (sensitivity / 100) * chartH

  return (
    <div className="border-2 border-foreground bg-background">
      <div className="px-4 py-2 border-b-2 border-foreground flex items-center gap-2">
        <BarChart3 size={12} className="text-accent" />
        <span className="font-mono text-[13px] tracking-[0.25em] uppercase text-accent font-bold">
          CONFIDENCE OVER TIME
        </span>
      </div>
      <div className="p-4">
        <svg
          viewBox={`0 0 ${chartW} ${chartH}`}
          className="w-full h-16"
          preserveAspectRatio="none"
        >
          <path d={fillD} fill="rgba(234,88,12,0.12)" />
          <path d={pathD} fill="none" stroke="#ea580c" strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
          <line
            x1="0" y1={threshY} x2={chartW} y2={threshY}
            stroke="rgba(255,255,255,0.3)" strokeWidth="0.5" strokeDasharray="2,2" vectorEffect="non-scaling-stroke"
          />
          {targetSegments.map((seg, i) => {
            const x = ((seg.startTime + seg.endTime) / 2 / totalDuration) * chartW
            const y = chartH - (seg.confidence / 100) * chartH
            return (
              <circle
                key={i}
                cx={x} cy={y} r="1.5"
                fill="#ea580c"
                vectorEffect="non-scaling-stroke"
              />
            )
          })}
        </svg>
        <div className="flex justify-between mt-1">
          <span className="font-mono text-[11px] text-muted-foreground/40">0:00</span>
          <span className="font-mono text-[11px] text-muted-foreground/40">
            {Math.floor(totalDuration / 60)}:{String(Math.floor(totalDuration % 60)).padStart(2, "0")}
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Troubleshooting Tips (shown only when NOT FOUND) ──

function TroubleshootingTips({
  targetSpecies,
  confidence,
  sensitivity,
}: {
  targetSpecies: string
  confidence: number
  sensitivity: number
}) {
  const meta = SPECIES_META[targetSpecies]
  const isNearMiss = confidence > 0 && confidence < sensitivity

  // Build context-aware tips
  const tips: { icon: string; text: string }[] = []

  if (isNearMiss) {
    tips.push({ icon: "⚙️", text: `Lower sensitivity below ${confidence.toFixed(0)}% — your detection (${confidence.toFixed(1)}%) is just under the ${sensitivity}% threshold` })
  }

  tips.push({ icon: "🎙️", text: "Record in a quieter environment with less background noise" })

  if (meta) {
    tips.push({ icon: "🌍", text: `Try recording in ${meta.habitat.toLowerCase()} — typical habitat for ${meta.name}` })
    tips.push({ icon: "🔊", text: `Listen for ${meta.callType.toLowerCase()} calls in the ${meta.freqLow}–${meta.freqHigh} kHz range` })
  }

  tips.push({ icon: "⏱️", text: "Use a longer recording (30s+) for better detection accuracy" })
  tips.push({ icon: "☑️", text: "Enable noise reduction in settings before analyzing" })

  return (
    <div className="border-2 border-foreground bg-background">
      <div className="px-4 py-2 border-b-2 border-foreground flex items-center gap-2">
        <AlertTriangle size={12} className="text-accent" />
        <span className="font-mono text-[13px] tracking-[0.25em] uppercase text-accent font-bold">
          WHAT TO TRY NEXT
        </span>
      </div>
      <div className="p-4 space-y-2.5">
        {tips.map((tip, i) => (
          <div key={i} className="flex items-start gap-2.5">
            <span className="text-sm leading-none mt-0.5 shrink-0">{tip.icon}</span>
            <p className="font-mono text-[13px] tracking-wider text-muted-foreground leading-relaxed">
              {tip.text}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main Export ──

export function SearchVerdict({
  targetSpecies,
  predictions,
  segments,
  totalDuration,
  sensitivity,
  onPlaySegment,
  playingIndex = -1,
}: SearchVerdictProps) {
  const match = predictions.find(
    (p) => p.species.toLowerCase() === targetSpecies.toLowerCase()
  )
  const confidence = match?.confidence ?? 0
  const found = confidence >= sensitivity
  const verdictColor = found ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)"
  const confColor = getAccuracyColor(confidence)
  const meta = SPECIES_META[targetSpecies]

  const targetSegments = segments.filter(
    (s) => s.species.toLowerCase() === targetSpecies.toLowerCase()
  )

  const [segmentsOpen, setSegmentsOpen] = useState(false)

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, "0")}`
  }

  return (
    <div className="space-y-4">
      {/* ━━━ VERDICT CARD ━━━ */}
      <div
        className="border-2 bg-background relative overflow-hidden"
        style={{ borderColor: verdictColor }}
      >
        {/* Glow strip */}
        <div className="absolute top-0 left-0 right-0 h-1" style={{ backgroundColor: verdictColor, opacity: 0.8 }} />

        <div className="p-5 sm:p-6">
          {/* Verdict header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 border-2 flex items-center justify-center"
                style={{ borderColor: verdictColor, backgroundColor: `${verdictColor}15` }}
              >
                {found
                  ? <Check size={20} style={{ color: verdictColor }} strokeWidth={3} />
                  : <X size={20} style={{ color: verdictColor }} strokeWidth={3} />}
              </div>
              <div>
                <h2 className="font-mono text-lg sm:text-xl tracking-[0.15em] uppercase font-bold" style={{ color: verdictColor }}>
                  {found ? "DETECTED" : "NOT FOUND"}
                </h2>
                <p className="font-mono text-[13px] tracking-wider text-muted-foreground uppercase">
                  {targetSpecies}
                  {meta && <span className="italic ml-2 text-muted-foreground/60 normal-case">({meta.scientificName})</span>}
                </p>
              </div>
            </div>
            {/* Big confidence number */}
            <div className="text-right">
              <span className="font-mono text-3xl sm:text-4xl font-bold" style={{ color: confColor }}>
                {confidence.toFixed(1)}
              </span>
              <span className="font-mono text-xs text-muted-foreground/60 ml-0.5">%</span>
            </div>
          </div>

          {/* Confidence gauge with threshold marker */}
          <div className="space-y-1.5">
            <div className="relative h-3 border border-foreground/20 bg-foreground/5 overflow-hidden">
              <div
                className="absolute top-0 left-0 bottom-0 transition-all duration-500"
                style={{ width: `${Math.min(confidence, 100)}%`, backgroundColor: confColor, opacity: 0.7 }}
              />
              <div
                className="absolute top-0 bottom-0 w-[2px] bg-foreground/60"
                style={{ left: `${sensitivity}%` }}
              />
            </div>
            <div className="flex justify-between text-muted-foreground/40 font-mono text-[11px]">
              <span>0%</span>
              <span className="text-foreground/40 text-[11px]">▲ {sensitivity}% threshold</span>
              <span>100%</span>
            </div>
          </div>

          {/* Stats strip */}
          <div className="flex gap-4 mt-4 pt-3 border-t border-foreground/10">
            {[
              { label: "SEGMENTS", value: String(targetSegments.length) },
              { label: "TOTAL TIME", value: `${targetSegments.reduce((a, s) => a + (s.endTime - s.startTime), 0).toFixed(1)}s` },
              { label: "COVERAGE", value: `${((targetSegments.reduce((a, s) => a + (s.endTime - s.startTime), 0) / totalDuration) * 100).toFixed(1)}%` },
            ].map((stat) => (
              <div key={stat.label} className="flex items-center gap-2">
                <span className="font-mono text-[11px] tracking-[0.15em] uppercase text-muted-foreground/50">
                  {stat.label}
                </span>
                <span className="font-mono text-xs font-bold text-foreground">{stat.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ━━━ DETECTION SEGMENTS (collapsible) ━━━ */}
      {targetSegments.length > 0 && (
        <div className="border-2 border-foreground bg-background">
          <button
            type="button"
            onClick={() => setSegmentsOpen(!segmentsOpen)}
            className="w-full px-4 py-2.5 border-b-2 border-foreground flex items-center justify-between cursor-pointer hover:bg-muted/30 transition-none bg-transparent"
          >
            <div className="flex items-center gap-2">
              <Play size={12} className="text-accent" />
              <span className="font-mono text-[13px] tracking-[0.25em] uppercase text-foreground font-bold">
                DETECTION SEGMENTS
              </span>
              <span className="font-mono text-[10px] text-accent font-bold ml-1">
                {targetSegments.length}
              </span>
            </div>
            <ChevronDown
              size={14}
              className={`text-muted-foreground transition-transform duration-200 ${segmentsOpen ? "rotate-180" : ""}`}
            />
          </button>

          {segmentsOpen && (
            <div className="divide-y divide-foreground/10">
              {targetSegments.map((seg, i) => {
                const isPlaying = playingIndex === i
                return (
                  <button
                    key={i}
                    type="button"
                    onClick={() => onPlaySegment(seg.startTime, seg.endTime)}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 font-mono text-left cursor-pointer transition-none ${isPlaying
                      ? "bg-accent/10"
                      : "hover:bg-muted/20"
                      }`}
                  >
                    {/* Play/Pause icon */}
                    <div className={`w-7 h-7 border flex items-center justify-center shrink-0 ${isPlaying ? "border-accent bg-accent/10" : "border-foreground/20"
                      }`}>
                      {isPlaying
                        ? <Pause size={11} className="text-accent" />
                        : <Play size={11} className="text-muted-foreground" />
                      }
                    </div>

                    {/* Segment label */}
                    <span className="text-[11px] tracking-wider uppercase text-muted-foreground">
                      Segment {i + 1}
                    </span>

                    {/* Time range */}
                    <span className="text-[11px] tracking-wider text-foreground font-bold ml-auto">
                      {fmt(seg.startTime)} – {fmt(seg.endTime)}
                    </span>

                    {/* Confidence */}
                    <span
                      className="text-[11px] font-bold w-12 text-right"
                      style={{ color: getAccuracyColor(seg.confidence) }}
                    >
                      {seg.confidence.toFixed(0)}%
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      )}


      {/* ━━━ CONFIDENCE OVER TIME ━━━ */}
      <ConfidenceTimeline
        segments={segments}
        species={targetSpecies}
        totalDuration={totalDuration}
        sensitivity={sensitivity}
      />

      {/* ━━━ LOW CONFIDENCE HINT ━━━ */}
      {!found && confidence > 0 && (
        <div className="border border-amber-500/40 bg-amber-500/5 px-5 py-3 flex items-start gap-3">
          <AlertTriangle size={14} className="text-amber-500 shrink-0 mt-0.5" />
          <div>
            <p className="font-mono text-[13px] tracking-wider text-amber-500/80 uppercase font-bold">
              LOW CONFIDENCE DETECTION
            </p>
            <p className="font-mono text-[13px] tracking-wider text-muted-foreground/60 mt-1 leading-relaxed">
              {targetSpecies} was detected at {confidence.toFixed(1)}% which is below
              your {sensitivity}% threshold. Try lowering sensitivity or use a
              cleaner recording.
            </p>
          </div>
        </div>
      )}

      {/* ━━━ TROUBLESHOOTING (only when NOT FOUND) ━━━ */}
      {!found && (
        <TroubleshootingTips
          targetSpecies={targetSpecies}
          confidence={confidence}
          sensitivity={sensitivity}
        />
      )}
    </div>
  )
}
