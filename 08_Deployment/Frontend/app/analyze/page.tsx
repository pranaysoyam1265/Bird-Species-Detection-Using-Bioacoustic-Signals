"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import Image from "next/image"
import { Navbar } from "@/components/navbar"
import { UploadZone } from "@/components/analyze/upload-zone"
import { AudioPlayer } from "@/components/analyze/audio-player"
import { RecordingInfo } from "@/components/analyze/recording-info"
import { TopResultCard } from "@/components/analyze/top-result-card"
import { PredictionsChart } from "@/components/analyze/predictions-chart"
import { SpectrogramView } from "@/components/analyze/spectrogram-view"
import { DetectionTimeline, type DetectionSegment } from "@/components/analyze/detection-timeline"
import { AnalyzeSidebar, type SidebarSettings } from "@/components/analyze/analyze-sidebar"
import { AnalysisProgress } from "@/components/analyze/analysis-progress"
import { AudioDecodingScreen } from "@/components/analyze/audio-decoding-screen"
import { SpeciesInfoModal } from "@/components/analyze/species-info-modal"
import { FileQueue, type QueueItem } from "@/components/analyze/file-queue"
import { MicRecorder } from "@/components/analyze/mic-recorder"
import { SpeciesProfileCard } from "@/components/analyze/species-profile-card"
import { SearchVerdict } from "@/components/analyze/search-verdict"
import { DeepAnalysisReport } from "@/components/analyze/deep-analysis-report"
import { NavSidebar } from "@/components/nav-sidebar"
import { detectAudio, type DetectionRecord, saveDetection, buildRecord, fileToDataUri } from "@/lib/detection-store"
import { SPECIES_META } from "@/lib/species-meta"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"
import { Menu, X, Zap, Loader2, Upload, Mic, PanelLeftClose, PanelLeftOpen, Search } from "lucide-react"

// ── Types ──
type PageState = "idle" | "loading" | "uploaded" | "analyzing" | "results" | "error"
type InputMode = "upload" | "record"

export default function AnalyzePage() {
  const { toast } = useToast()
  const { user, loading: authLoading } = useAuth()
  const router = useRouter()
  const [pageState, setPageState] = useState<PageState>("idle")

  /* ── Auth guard: redirect to login if not authenticated ── */
  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login")
    }
  }, [authLoading, user, router])
  const [inputMode, setInputMode] = useState<InputMode>("upload")
  const [file, setFile] = useState<File | null>(null)
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null)
  const [results, setResults] = useState<DetectionRecord | null>(null)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState("")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [modalSpecies, setModalSpecies] = useState<string | null>(null)
  const [playingSegmentIndex, setPlayingSegmentIndex] = useState<number>(-1)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const segmentTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // File queue
  const [queue, setQueue] = useState<QueueItem[]>([])

  const [settings, setSettings] = useState<SidebarSettings>({
    noiseReduction: false,
    chunkDuration: 5,
    topK: 5,
    minConfidence: 10,
    searchSpecies: "",
    sensitivity: 5,
  })

  const loadingStartRef = useRef(0)

  // ── File handling ──
  const handleFileSelect = useCallback((f: File) => {
    setFile(f)
    setUploadError("")
    setResults(null)
    setAudioBuffer(null)
    setPageState("loading")
    loadingStartRef.current = Date.now()
  }, [])

  const handleMultiFileSelect = useCallback((files: File[]) => {
    const newItems: QueueItem[] = files.map((f) => ({
      file: f,
      status: "pending" as const,
      id: `${f.name}-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    }))
    setQueue((prev) => [...prev, ...newItems])
    // Auto-select first if nothing loaded
    if (!file && files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [file, handleFileSelect])

  const handleRemoveFile = useCallback(() => {
    setFile(null)
    setAudioBuffer(null)
    setResults(null)
    setPageState("idle")
    setPlayingSegmentIndex(-1)
  }, [])

  const handleAudioDecoded = useCallback((buffer: AudioBuffer) => {
    setAudioBuffer(buffer)
    const elapsed = Date.now() - loadingStartRef.current
    const minDelay = 10000
    const remaining = Math.max(minDelay - elapsed, 0)
    setTimeout(() => setPageState("uploaded"), remaining)
  }, [])

  // ── Analysis ── (real ML inference via /api/detect → FastAPI → EfficientNet-B2)
  const handleAnalyze = useCallback(async () => {
    if (!file) return
    setPageState("analyzing")
    setPlayingSegmentIndex(-1)
    setAnalyzeError(null)
    try {
      const record = await detectAudio(file, {
        topK: settings.topK,
        confidenceThreshold: settings.minConfidence,
        noiseReduction: settings.noiseReduction,
      })
      setResults(record)
      setPageState("results")
    } catch (err) {
      console.error("[Analyze] Detection failed:", err)
      setAnalyzeError(err instanceof Error ? err.message : "Detection failed")
      setPageState("error")
      toast({ title: "Detection failed", description: err instanceof Error ? err.message : "An unknown error occurred" })
    }
  }, [file, settings, toast])

  const handleAnalyzeNew = useCallback(() => {
    setFile(null)
    setAudioBuffer(null)
    setResults(null)
    setUploadError("")
    setPageState("idle")
    setPlayingSegmentIndex(-1)
  }, [])

  // ── Export / Copy / Save ──
  const handleExport = useCallback(() => {
    if (!results) return
    const data = JSON.stringify(results, null, 2)
    const blob = new Blob([data], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "birdsense-results.json"
    a.click()
    URL.revokeObjectURL(url)
  }, [results])

  const handleCopyResults = useCallback(async () => {
    if (!results || !file) return
    const lines = [
      "=== BIRDSENSE DETECTION RESULTS ===",
      `File: ${file.name}`,
      `Date: ${new Date().toLocaleString()}`,
      "",
      "TOP DETECTION:",
      `${results.topSpecies} (${results.topScientific})`,
      `Confidence: ${results.topConfidence.toFixed(1)}%`,
      "",
      "ALL PREDICTIONS:",
      ...results.predictions.map((p, i) => `${i + 1}. ${p.species} - ${p.confidence.toFixed(1)}%`),
    ]
    await navigator.clipboard.writeText(lines.join("\n"))
  }, [results, file])

  const [saved, setSaved] = useState(false)

  const handleSave = useCallback(async () => {
    // detectAudio() already saves the record to DB automatically
    if (!results) return
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
    toast({ title: "Saved to history", description: `${results.topSpecies} detection saved` })
  }, [results, toast])

  // ── Segment playback ──
  const handlePlaySegment = useCallback((start: number, end: number) => {
    const audio = document.querySelector<HTMLAudioElement>("audio")
    if (!audio) return

    // Find which segment index this is
    const idx = results?.segments.findIndex(s => s.startTime === start && s.endTime === end) ?? -1

    if (playingSegmentIndex === idx) {
      // Stop playing
      audio.pause()
      setPlayingSegmentIndex(-1)
      if (segmentTimerRef.current) clearInterval(segmentTimerRef.current)
      return
    }

    audio.currentTime = start
    audio.play()
    setPlayingSegmentIndex(idx)

    if (segmentTimerRef.current) clearInterval(segmentTimerRef.current)
    segmentTimerRef.current = setInterval(() => {
      if (audio.currentTime >= end || audio.paused) {
        audio.pause()
        setPlayingSegmentIndex(-1)
        if (segmentTimerRef.current) clearInterval(segmentTimerRef.current)
      }
    }, 100)
  }, [playingSegmentIndex, results])

  // ── Queue ──
  const handleQueueRemove = useCallback((id: string) => {
    setQueue((prev) => prev.filter((i) => i.id !== id))
  }, [])

  const handleQueueClear = useCallback(() => setQueue([]), [])

  const handleQueueProcess = useCallback(() => {
    // Process first pending item
    const pending = queue.find((i) => i.status === "pending")
    if (pending) {
      handleFileSelect(pending.file)
      setQueue((prev) =>
        prev.map((i) => i.id === pending.id ? { ...i, status: "processing" as const } : i)
      )
    }
  }, [queue, handleFileSelect])

  const handleQueueSelect = useCallback((item: QueueItem) => {
    handleFileSelect(item.file)
  }, [handleFileSelect])

  // ── Recording ──
  const handleRecordingComplete = useCallback((f: File) => {
    handleFileSelect(f)
    setInputMode("upload")
  }, [handleFileSelect])

  const duration = audioBuffer?.duration ?? 0
  const sampleRate = audioBuffer?.sampleRate ?? 0

  /* Show loading while auth is resolving or redirecting */
  if (authLoading || !user) {
    return (
      <div className="min-h-screen dot-grid-bg flex items-center justify-center scanline-overlay">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent animate-spin" />
          <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
            AUTHENTICATING...
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      {/* Radial vignette */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />
      <Navbar />

      {/* ── Page Header ── */}
      <div className="px-4 lg:px-6 pt-4 lg:pt-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <NavSidebar />
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Mic size={16} className="text-accent" />
              <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
                AUDIO ANALYSIS
              </h1>
            </div>
            <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
              BIOACOUSTIC SPECIES DETECTION
            </p>
          </div>
        </div>
      </div>

      <main className="flex-1 flex flex-col lg:flex-row p-4 lg:p-6 gap-4 lg:gap-6">
        {/* SIDEBAR — settings toggle + AnalyzeSidebar */}
        <div className="space-y-3 shrink-0">
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex items-center gap-2 px-4 py-2 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase cursor-pointer hover:bg-muted hover:text-accent transition-none shrink-0"
          >
            {sidebarOpen ? <PanelLeftClose size={14} /> : <PanelLeftOpen size={14} />}
            {sidebarOpen ? "HIDE SETTINGS" : "SHOW SETTINGS"}
          </button>
          {sidebarOpen && (
            <AnalyzeSidebar
              settings={settings}
              onSettingsChange={setSettings}
              hasResults={pageState === "results"}
              onAnalyzeNew={handleAnalyzeNew}
              onExport={handleExport}
              onSave={handleSave}
              onCopyResults={handleCopyResults}
            />
          )}
        </div>

        {/* MAIN CONTENT */}
        <div className="flex-1 space-y-4 lg:space-y-6 min-w-0">

          {/* Top row: Upload/Record toggle + Status bar */}
          <div className="flex items-center justify-between gap-4">
            {/* Input mode toggle (Upload / Record) */}
            {pageState === "idle" ? (
              <div className="flex gap-0 border-2 border-foreground w-fit">
                <button
                  type="button"
                  onClick={() => setInputMode("upload")}
                  className={`flex items-center gap-2 px-5 py-2.5 font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none ${inputMode === "upload"
                    ? "bg-foreground text-background font-bold"
                    : "bg-background text-foreground hover:bg-muted"
                    }`}
                >
                  <Upload size={14} /> UPLOAD
                </button>
                <button
                  type="button"
                  onClick={() => setInputMode("record")}
                  className={`flex items-center gap-2 px-5 py-2.5 font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none border-l-2 border-foreground ${inputMode === "record"
                    ? "bg-foreground text-background font-bold"
                    : "bg-background text-foreground hover:bg-muted"
                    }`}
                >
                  <Mic size={14} /> RECORD
                </button>
              </div>
            ) : <div />}

            {/* Status bar */}
            <div className="border border-foreground/30 bg-muted/30 px-3 py-2 font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground flex items-center gap-2 shrink-0">
              <span className="text-accent/60 select-none">SYS_STATUS:</span>
              <span className="inline-flex items-center gap-1 text-foreground font-bold">
                {pageState === "idle" && "AWAITING INPUT"}
                {pageState === "loading" && "DECODING..."}
                {pageState === "uploaded" && "READY"}
                {pageState === "analyzing" && "PROCESSING..."}
                {pageState === "results" && "COMPLETE"}
                {pageState === "error" && "ERROR"}
                {analyzeError && <span className="text-red-500 ml-2 normal-case text-[9px]">{analyzeError}</span>}
                <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
              </span>
            </div>
          </div>

          {/* ── Main content area (always visible) ── */}
          <>
            {/* Species focus prompt — subtle nudge when no target selected */}
            {pageState === "idle" && !settings.searchSpecies && (
              <div className="border border-accent/30 bg-accent/5 px-4 py-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <Search size={14} className="text-accent shrink-0" />
                  <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground/80">
                    Select a <strong className="text-accent">target species</strong> in the sidebar to search for it in your audio
                  </span>
                </div>
                {!sidebarOpen && (
                  <button
                    type="button"
                    onClick={() => setSidebarOpen(true)}
                    className="shrink-0 px-3 py-1.5 border border-accent text-accent font-mono text-[9px] tracking-[0.15em] uppercase font-bold hover:bg-accent hover:text-white cursor-pointer transition-none bg-transparent"
                  >
                    OPEN SETTINGS
                  </button>
                )}
              </div>
            )}

            {/* Upload Zone or Mic Recorder */}
            {pageState === "idle" && inputMode === "upload" && (
              <UploadZone
                onFileSelect={handleFileSelect}
                currentFile={file}
                error={uploadError}
              />
            )}

            {pageState === "idle" && inputMode === "record" && (
              <MicRecorder onRecordingComplete={handleRecordingComplete} />
            )}

            {/* File Queue */}
            {queue.length > 0 && (
              <FileQueue
                items={queue}
                onRemove={handleQueueRemove}
                onClear={handleQueueClear}
                onProcess={handleQueueProcess}
                onSelectItem={handleQueueSelect}
              />
            )}

            {/* Loading state while audio is being decoded */}
            {pageState === "loading" && file && (
              <AudioDecodingScreen file={file} />
            )}

            {/* Audio Player — render during loading (hidden) so decoding can happen */}
            {file && pageState !== "idle" && (
              <>
                <div className={pageState === "loading" ? "hidden" : ""}>
                  <AudioPlayer
                    file={file}
                    onRemove={handleRemoveFile}
                    onAudioDecoded={handleAudioDecoded}
                  />
                </div>

                {audioBuffer && pageState !== "loading" && (
                  <RecordingInfo duration={duration} sampleRate={sampleRate} />
                )}
              </>
            )}

            {/* Analyze Button (when uploaded and not yet analyzing) */}
            {pageState === "uploaded" && (
              <>
                {!settings.searchSpecies ? (
                  <div className="w-full py-4 border-2 border-foreground/30 bg-muted/30 flex flex-col items-center gap-3">
                    <div className="flex items-center gap-2">
                      <Search size={16} className="text-accent" />
                      <span className="font-mono text-sm tracking-[0.2em] uppercase text-muted-foreground">
                        SELECT A TARGET SPECIES TO ANALYZE
                      </span>
                    </div>
                    <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground/60">
                      Choose a bird from the Species Focus panel in the sidebar
                    </p>
                    {!sidebarOpen && (
                      <button
                        type="button"
                        onClick={() => setSidebarOpen(true)}
                        className="px-4 py-2 border border-accent text-accent font-mono text-[10px] tracking-[0.15em] uppercase font-bold hover:bg-accent hover:text-white cursor-pointer transition-none bg-transparent"
                      >
                        OPEN SETTINGS
                      </button>
                    )}
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    className="w-full py-4 border-2 border-accent bg-accent text-white font-mono text-sm tracking-[0.2em] uppercase font-bold shadow-[4px_4px_0px_0px_rgba(234,88,12,0.3)] hover:bg-background hover:text-accent active:shadow-none active:translate-x-[4px] active:translate-y-[4px] transition-none cursor-pointer flex items-center justify-center gap-3 overflow-hidden relative group"
                  >
                    <span className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-white/10 skew-x-12 pointer-events-none" />
                    <Zap size={16} className="animate-zap" />
                    {`>>> ANALYZE & SEARCH ${settings.searchSpecies.toUpperCase()} <<<`}
                  </button>
                )}
              </>
            )}

            {/* Real inference in progress */}
            {pageState === "analyzing" && (
              <div className="border-2 border-accent bg-background p-6 space-y-4 relative overflow-hidden">
                <div className="absolute inset-0 border-2 border-accent opacity-30 animate-ping pointer-events-none rounded-none" />
                <div className="flex items-center gap-3">
                  <Loader2 size={18} className="animate-spin text-accent" />
                  <span className="font-mono text-xs tracking-[0.2em] uppercase text-foreground font-bold">
                    ANALYZING AUDIO
                  </span>
                </div>
                <div className="w-full h-3 border-2 border-foreground bg-background overflow-hidden">
                  <div className="h-full bg-accent animate-pulse w-full relative">
                    <div className="absolute inset-0 bg-white/20 animate-shimmer" />
                  </div>
                </div>
                <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
                  Sending audio to ML model (EfficientNet-B2)...
                </p>
                <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground animate-blink">
                  PROCESSING AUDIO SEGMENTS...
                </p>
              </div>
            )}

            {/* Results (when analysis complete) */}
            {pageState === "results" && results && (
              <div className="space-y-4 lg:space-y-6">

                {/* ══ FOCUS VERDICT ═══ Only when a species is selected ══ */}
                {settings.searchSpecies && (
                  <SearchVerdict
                    targetSpecies={settings.searchSpecies}
                    predictions={results.predictions}
                    segments={results.segments}
                    totalDuration={Math.max(duration, 30)}
                    sensitivity={settings.sensitivity}
                    onPlaySegment={handlePlaySegment}
                    playingIndex={playingSegmentIndex}
                  />
                )}

                {/* Species Profile — visible when target bird is detected */}
                {(() => {
                  const targetMatch = results.predictions.find(
                    (p) => p.species.toLowerCase() === settings.searchSpecies.toLowerCase()
                  )
                  if (!targetMatch || targetMatch.confidence < settings.sensitivity) return null
                  const meta = SPECIES_META[settings.searchSpecies]
                  return (
                    <div className="group transition-shadow duration-300 hover:shadow-[0_0_24px_2px_rgba(34,197,94,0.15)]">
                      <SpeciesProfileCard
                        species={settings.searchSpecies}
                        scientificName={meta?.scientificName ?? results.topScientific}
                      />
                    </div>
                  )
                })()}

                {/* ══ DEEP ANALYSIS — Terminal Typewriter Report ══ */}
                <DeepAnalysisReport
                  results={results}
                  duration={duration}
                  audioBuffer={audioBuffer}
                  onSpeciesClick={setModalSpecies}
                  onPlaySegment={handlePlaySegment}
                  playingIndex={playingSegmentIndex}
                />
              </div>
            )}

            {/* Empty state — idle bird hero */}
            {pageState === "idle" && (
              <div className="flex flex-col items-center justify-center py-12 px-4 gap-6 text-center">
                {/* Floating bird silhouette */}
                <div className="animate-float relative">
                  <div className="w-32 h-32 flex items-center justify-center relative">
                    {/* Glow ring */}
                    <div className="absolute inset-0 rounded-full bg-accent/10 blur-xl" />
                    <Image
                      src="/bird-svgrepo-com (2).svg"
                      alt="Bird silhouette"
                      width={112}
                      height={112}
                      className="relative w-28 h-28"
                      style={{ filter: "invert(42%) sepia(93%) saturate(1352%) hue-rotate(5deg) brightness(119%) contrast(119%)" }}
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="font-mono text-sm tracking-[0.2em] uppercase text-foreground font-bold">
                    DROP AN AUDIO FILE TO BEGIN
                  </p>
                  <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
                    or use the upload zone above
                  </p>
                </div>

                {/* Quick start steps */}
                <div className="border border-foreground/10 bg-background/50 p-4 space-y-2 max-w-sm w-full">
                  <span className="font-mono text-[9px] tracking-[0.25em] uppercase text-muted-foreground/60 block mb-3">
                    QUICK START
                  </span>
                  {[
                    "Upload a field recording or record from your microphone",
                    "Configure detection settings in the sidebar",
                    "Click ANALYZE to identify bird species",
                    "Click species names for detailed info & external links",
                  ].map((tip, i) => (
                    <p key={i} className="flex items-start gap-2 font-mono text-[10px] text-muted-foreground text-left">
                      <span className="text-accent shrink-0 font-bold">{String(i + 1).padStart(2, "0")}.</span>
                      {tip}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </>



        </div>
      </main>

      {/* Species Info Modal */}
      {modalSpecies && (
        <SpeciesInfoModal
          species={modalSpecies}
          onClose={() => setModalSpecies(null)}
        />
      )}
    </div>
  )
}
