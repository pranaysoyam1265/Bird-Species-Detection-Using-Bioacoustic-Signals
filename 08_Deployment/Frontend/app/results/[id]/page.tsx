"use client"

import { useState, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import { SpectrogramView } from "@/components/analyze/spectrogram-view"
import {
  ArrowLeft,
  Play,
  Pause,
  Download,
  Trash2,
  Clock,
  Calendar,
  FileAudio,
  Activity,
  BarChart3,
  Eye,
} from "lucide-react"
import {
  type DetectionRecord,
  getDetectionById,
  deleteDetections,
} from "@/lib/detection-store"
import { useToast } from "@/hooks/use-toast"
import { fmtDuration, getConfColor } from "@/lib/format"

/* ─── Page ───────────────────────────────────────────────── */
export default function ResultDetailPage() {
  const params = useParams()
  const router = useRouter()
  const { toast } = useToast()
  const id = params.id as string

  const [record, setRecord] = useState<DetectionRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    const r = getDetectionById(id)
    setRecord(r ?? null)
    setLoading(false)

    // Decode base64 audio → AudioBuffer for spectrogram
    if (r?.audioUrl) {
      fetch(r.audioUrl)
        .then((res) => res.arrayBuffer())
        .then((buf) => {
          const ctx = new AudioContext()
          return ctx.decodeAudioData(buf)
        })
        .then((ab) => setAudioBuffer(ab))
        .catch(() => { /* audio decode failed, spectrogram will show NO DATA */ })
    }
  }, [id])

  // Audio controls
  const togglePlay = () => {
    if (!record?.audioUrl) return
    if (playing) {
      audioRef.current?.pause()
      setPlaying(false)
    } else {
      if (!audioRef.current) {
        audioRef.current = new Audio(record.audioUrl)
        audioRef.current.addEventListener("ended", () => setPlaying(false))
      }
      audioRef.current.play()
      setPlaying(true)
    }
  }

  useEffect(() => {
    return () => { audioRef.current?.pause() }
  }, [])

  const handleDelete = () => {
    if (!record) return
    deleteDetections([record.id])
    toast({ title: "Record deleted", description: "Redirecting to history..." })
    router.push("/results")
  }

  const handleExport = () => {
    if (!record) return
    const json = JSON.stringify(record, null, 2)
    const blob = new Blob([json], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `birdsense-${record.filename}.json`
    a.click()
    URL.revokeObjectURL(url)
    toast({ title: "Export complete", description: `Saved as birdsense-${record.filename}.json` })
  }

  if (loading) {
    return (
      <div className="min-h-screen dot-grid-bg flex items-center justify-center scanline-overlay">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent animate-spin" />
          <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
            LOADING RECORD...
          </span>
        </div>
      </div>
    )
  }

  if (!record) {
    return (
      <div className="min-h-screen dot-grid-bg flex flex-col scanline-overlay">
        <div
          className="pointer-events-none fixed inset-0 z-0"
          style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
        />
        <Navbar />
        <div className="flex-1 flex flex-col items-center justify-center px-4 text-center space-y-4">
          <span className="font-mono text-lg tracking-[0.2em] uppercase text-muted-foreground">
            RECORD NOT FOUND
          </span>
          <p className="font-mono text-xs text-muted-foreground/60">
            This detection may have been deleted.
          </p>
          <button
            type="button"
            onClick={() => router.push("/results")}
            className="mt-4 px-6 py-2 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
          >
            ← BACK TO HISTORY
          </button>
        </div>
      </div>
    )
  }

  const confColor = getConfColor(record.topConfidence)
  const maxPredConf = Math.max(...record.predictions.map((p) => p.confidence), 1)

  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      {/* Radial vignette */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />
      <Navbar />

      {/* ── Page Header ── */}
      <div className="px-4 lg:px-6 pt-4 lg:pt-6 flex items-center gap-3">
        <NavSidebar />
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <Eye size={16} className="text-accent" />
            <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
              DETECTION DETAIL
            </h1>
          </div>
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            {record.filename}
          </p>
        </div>
      </div>

      <main className="flex-1 p-4 lg:p-6 space-y-6 max-w-5xl mx-auto w-full">
        {/* ────── ACTION BAR ────── */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => router.push("/results")}
            className="w-9 h-9 border-2 border-foreground flex items-center justify-center text-foreground hover:bg-muted cursor-pointer transition-none shrink-0"
          >
            <ArrowLeft size={16} />
          </button>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 flex-1 min-w-0">
            <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground flex items-center gap-1">
              <Calendar size={10} /> {record.date}
            </span>
            <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground flex items-center gap-1">
              <Clock size={10} /> {record.time}
            </span>
            <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground flex items-center gap-1">
              <FileAudio size={10} /> {fmtDuration(record.duration)}
            </span>
          </div>
          <div className="flex gap-2 shrink-0">
            <button
              type="button"
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
            >
              <Download size={12} /> EXPORT
            </button>
            <button
              type="button"
              onClick={handleDelete}
              className="flex items-center gap-1.5 px-3 py-2 border-2 border-red-500 text-red-500 font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-red-500/10 cursor-pointer transition-none"
            >
              <Trash2 size={12} /> DELETE
            </button>
          </div>
        </div>

        {/* ────── TOP RESULT CARD ────── */}
        <div className="border-2 border-foreground bg-background p-6">
          <div className="flex items-start gap-6">
            {/* Confidence gauge */}
            <div className="shrink-0 flex flex-col items-center gap-2">
              <div
                className="w-20 h-20 rounded-full border-4 flex items-center justify-center"
                style={{ borderColor: confColor }}
              >
                <span className="font-mono text-lg font-bold" style={{ color: confColor }}>
                  {record.topConfidence}%
                </span>
              </div>
              <span className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
                CONFIDENCE
              </span>
            </div>

            {/* Species info */}
            <div className="flex-1 min-w-0 space-y-2">
              <div>
                <h2 className="font-mono text-xl tracking-[0.15em] uppercase text-foreground font-bold">
                  {record.topSpecies}
                </h2>
                <p className="font-mono text-xs tracking-wider text-muted-foreground italic">
                  {record.topScientific}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent font-bold">
                  TOP DETECTION
                </span>
                <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground">
                  {record.segments.length} SEGMENTS · {record.predictions.length} SPECIES DETECTED
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ────── AUDIO PLAYER ────── */}
        {record.audioUrl && (
          <div className="border-2 border-foreground bg-background p-4">
            <div className="flex items-center gap-3">
              <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold shrink-0">
                <Activity size={12} className="inline mr-1.5" />
                AUDIO PREVIEW
              </span>
              <div className="flex-1" />
              <button
                type="button"
                onClick={togglePlay}
                className={`flex items-center gap-2 px-4 py-2 border-2 font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none ${playing
                  ? "border-accent bg-accent text-white"
                  : "border-foreground hover:bg-muted"
                  }`}
              >
                {playing ? <><Pause size={14} /> PAUSE</> : <><Play size={14} /> PLAY</>}
              </button>
            </div>
            {/* Waveform placeholder bar */}
            <div className="mt-3 h-12 bg-foreground/5 border border-foreground/10 flex items-end gap-px px-2">
              {Array.from({ length: 60 }).map((_, i) => {
                const h = Math.sin(i * 0.3) * 30 + Math.random() * 15 + 10
                return (
                  <div
                    key={i}
                    className="flex-1 rounded-t-sm"
                    style={{
                      height: `${h}%`,
                      backgroundColor: playing ? "var(--accent-hex)" : "currentColor",
                      opacity: playing ? 0.8 : 0.15,
                    }}
                  />
                )
              })}
            </div>
          </div>
        )}

        {/* ────── SPECTROGRAM ────── */}
        {record.audioUrl && (
          <SpectrogramView audioBuffer={audioBuffer} />
        )}

        {/* ────── PREDICTIONS BAR CHART ────── */}
        <div className="border-2 border-foreground bg-background p-4 space-y-3">
          <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
            <BarChart3 size={12} className="inline mr-1.5" />
            ALL PREDICTIONS
          </span>
          <div className="space-y-2">
            {record.predictions.map((pred, i) => {
              const barW = (pred.confidence / maxPredConf) * 100
              const color = getConfColor(pred.confidence)
              return (
                <div key={i} className="flex items-center gap-3">
                  <span className="font-mono text-[10px] tracking-wider text-foreground w-[160px] truncate shrink-0 text-right">
                    {pred.species}
                  </span>
                  <div className="flex-1 h-5 bg-foreground/5 overflow-hidden">
                    <div
                      className="h-full flex items-center px-2"
                      style={{ width: `${barW}%`, backgroundColor: color }}
                    >
                      <span className="font-mono text-[9px] font-bold text-white drop-shadow-sm">
                        {pred.confidence.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* ────── DETECTION SEGMENTS ────── */}
        {record.segments.length > 0 && (
          <div className="border-2 border-foreground bg-background p-4 space-y-3">
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
              <Clock size={12} className="inline mr-1.5" />
              DETECTION SEGMENTS ({record.segments.length})
            </span>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-foreground/20">
                    <th className="text-left px-3 py-2 font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">#</th>
                    <th className="text-left px-3 py-2 font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">TIME RANGE</th>
                    <th className="text-left px-3 py-2 font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">SPECIES</th>
                    <th className="text-left px-3 py-2 font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">CONF</th>
                  </tr>
                </thead>
                <tbody>
                  {record.segments.map((seg, i) => {
                    const segColor = getConfColor(seg.confidence)
                    return (
                      <tr key={i} className="border-b border-foreground/5 hover:bg-muted/30">
                        <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground">{i + 1}</td>
                        <td className="px-3 py-2 font-mono text-[11px] text-foreground">
                          {fmtDuration(Math.round(seg.startTime))} — {fmtDuration(Math.round(seg.endTime))}
                        </td>
                        <td className="px-3 py-2 font-mono text-[11px] text-foreground font-bold">{seg.species}</td>
                        <td className="px-3 py-2">
                          <span className="font-mono text-[11px] font-bold" style={{ color: segColor }}>
                            {seg.confidence.toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ────── STATUS BAR ────── */}
        <div className="border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">
            SYS_STATUS:
          </span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            DETAIL VIEW · {record.filename}
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>
      </main>
    </div>
  )
}
