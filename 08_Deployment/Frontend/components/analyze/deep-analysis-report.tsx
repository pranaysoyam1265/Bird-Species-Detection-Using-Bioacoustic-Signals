"use client"

import { useState } from "react"
import { BarChart3, Eye, EyeOff } from "lucide-react"
import { TopResultCard } from "@/components/analyze/top-result-card"
import { PredictionsChart } from "@/components/analyze/predictions-chart"
import { DetectionTimeline, type DetectionSegment } from "@/components/analyze/detection-timeline"
import { SpectrogramView } from "@/components/analyze/spectrogram-view"

interface DeepAnalysisReportProps {
  results: {
    topSpecies: string
    topScientific: string
    topConfidence: number
    predictions: Array<{ species: string; confidence: number }>
    segments: DetectionSegment[]
  }
  duration: number
  audioBuffer: AudioBuffer | null
  onSpeciesClick: (s: string) => void
  onPlaySegment: (start: number, end: number) => void
  playingIndex: number
}

export function DeepAnalysisReport({
  results,
  duration,
  audioBuffer,
  onSpeciesClick,
  onPlaySegment,
  playingIndex,
}: DeepAnalysisReportProps) {
  const [revealed, setRevealed] = useState(false)

  return (
    <div className="space-y-4">
      {/* ── Reveal Button (before reveal) ── */}
      {!revealed && (
        <div className="relative">
          {/* Preview — light blur so borders/shapes are visible */}
          <div
            className="overflow-hidden pointer-events-none select-none"
            style={{ maxHeight: "280px" }}
          >
            <div
              style={{
                filter: "blur(3px)",
                opacity: 0.65,
              }}
            >
              <div className="space-y-4 p-4">
                <TopResultCard
                  species={results.topSpecies}
                  scientificName={results.topScientific}
                  confidence={results.topConfidence}
                  onSpeciesClick={() => { }}
                />
                <PredictionsChart predictions={results.predictions} />
              </div>
            </div>
          </div>

          {/* Gradual fade: transparent at top → solid at bottom */}
          <div className="absolute inset-0 pointer-events-none" style={{
            background: "linear-gradient(to bottom, transparent 0%, hsl(var(--background) / 0.15) 30%, hsl(var(--background) / 0.5) 55%, hsl(var(--background) / 0.85) 75%, hsl(var(--background)) 100%)",
          }} />

          {/* Button centered at the bottom */}
          <div className="absolute bottom-5 left-0 right-0 flex justify-center z-10">
            <button
              type="button"
              onClick={() => setRevealed(true)}
              className="px-8 py-3.5 border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.2em] uppercase font-bold shadow-[4px_4px_0px_0px_rgba(234,88,12,0.3)] hover:bg-background hover:text-accent active:shadow-none active:translate-x-[4px] active:translate-y-[4px] transition-none cursor-pointer flex items-center gap-3"
            >
              <Eye size={15} />
              VIEW DETAILED BREAKDOWN
              <BarChart3 size={15} />
            </button>
          </div>
        </div>
      )}

      {/* ── Revealed content ── */}
      {revealed && (
        <div className="space-y-4 lg:space-y-6 animate-in fade-in duration-500">
          {/* Section label */}
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <BarChart3 size={14} className="text-accent" />
              <span className="font-mono text-[11px] tracking-[0.25em] uppercase text-foreground font-bold">
                DETAILED BREAKDOWN
              </span>
              <span className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
                — {results.predictions.length} species · {results.segments.length} segments
              </span>
            </div>
            <button
              type="button"
              onClick={() => setRevealed(false)}
              className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              title="Hide breakdown"
            >
              <EyeOff size={15} />
            </button>
          </div>

          {/* Top Result */}
          <div className="group transition-shadow duration-300 hover:shadow-[0_0_24px_2px_rgba(234,88,12,0.18)]">
            <TopResultCard
              species={results.topSpecies}
              scientificName={results.topScientific}
              confidence={results.topConfidence}
              onSpeciesClick={onSpeciesClick}
            />
          </div>

          {/* Predictions Chart */}
          <div className="group transition-shadow duration-300 hover:shadow-[0_0_24px_2px_rgba(234,88,12,0.12)]">
            <PredictionsChart predictions={results.predictions} />
          </div>

          {/* Detection Timeline */}
          <div className="group transition-shadow duration-300 hover:shadow-[0_0_24px_2px_rgba(245,158,11,0.15)]">
            <DetectionTimeline
              segments={results.segments}
              totalDuration={Math.max(duration, 30)}
              onPlaySegment={onPlaySegment}
              playingIndex={playingIndex}
            />
          </div>

          {/* Spectrogram */}
          <div className="group transition-shadow duration-300 hover:shadow-[0_0_24px_2px_rgba(168,85,247,0.15)]">
            <SpectrogramView audioBuffer={audioBuffer} />
          </div>
        </div>
      )}
    </div>
  )
}
