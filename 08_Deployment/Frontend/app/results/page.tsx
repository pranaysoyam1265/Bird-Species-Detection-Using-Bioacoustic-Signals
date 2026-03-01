"use client"

import { useState, useEffect, useMemo, useCallback, useRef } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import {
  Search,
  Filter,
  Download,
  Trash2,
  ChevronUp,
  ChevronDown,
  X,
  History,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  BarChart3,
  Eye,
  FileText,
  Zap,
} from "lucide-react"
import {
  type DetectionRecord,
  getDetections,
  deleteDetections,
} from "@/lib/detection-store"
import { useToast } from "@/hooks/use-toast"
import { fmtDuration, getConfColor } from "@/lib/format"

type SortField = "date" | "species" | "confidence" | "duration"
type SortOrder = "asc" | "desc"

/* ─── Helpers ────────────────────────────────────────────── */
function timeAgo(dateStr: string, timeStr: string): string {
  const then = new Date(`${dateStr}T${timeStr}`)
  const now = new Date()
  const diffMs = now.getTime() - then.getTime()
  if (diffMs < 0) return "just now"
  const mins = Math.floor(diffMs / 60000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 7) return `${days}d ago`
  return `${Math.floor(days / 7)}w ago`
}

const PAGE_SIZE = 10
const SORT_KEY = "birdsense-results-sort"

/* ─── Page ───────────────────────────────────────────────── */
export default function ResultsPage() {
  const router = useRouter()
  const { toast } = useToast()
  const [history, setHistory] = useState<DetectionRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortField, setSortField] = useState<SortField>("date")
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc")
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [filterOpen, setFilterOpen] = useState(false)
  const [minConfidence, setMinConfidence] = useState(0)
  const [speciesFilter, setSpeciesFilter] = useState<Set<string>>(new Set())
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [confirmDelete, setConfirmDelete] = useState<{ ids: string[]; label: string } | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // ── Load from localStorage on mount + restore sort prefs ──
  useEffect(() => {
    setHistory(getDetections())
    setLoading(false)
    try {
      const saved = localStorage.getItem(SORT_KEY)
      if (saved) {
        const { field, order } = JSON.parse(saved)
        if (field) setSortField(field)
        if (order) setSortOrder(order)
      }
    } catch { /* ignore */ }
  }, [])

  // ── Unique species list for filter ──
  const allSpecies = useMemo(
    () => [...new Set(history.map((e) => e.topSpecies))].sort(),
    [history]
  )

  // ── Stats ──
  const totalCount = history.length
  const avgConf =
    totalCount > 0
      ? (history.reduce((a, e) => a + e.topConfidence, 0) / totalCount).toFixed(1)
      : "0.0"

  // ── Filter + Sort ──
  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase()
    let items = history.filter(
      (e) =>
        e.topConfidence >= minConfidence &&
        (speciesFilter.size === 0 || speciesFilter.has(e.topSpecies)) &&
        (e.filename.toLowerCase().includes(q) ||
          e.topSpecies.toLowerCase().includes(q) ||
          e.topScientific.toLowerCase().includes(q))
    )

    items.sort((a, b) => {
      let cmp = 0
      switch (sortField) {
        case "date":
          cmp = `${a.date}T${a.time}`.localeCompare(`${b.date}T${b.time}`)
          break
        case "species":
          cmp = a.topSpecies.localeCompare(b.topSpecies)
          break
        case "confidence":
          cmp = a.topConfidence - b.topConfidence
          break
        case "duration":
          cmp = a.duration - b.duration
          break
      }
      return sortOrder === "asc" ? cmp : -cmp
    })
    return items
  }, [history, searchQuery, sortField, sortOrder, minConfidence, speciesFilter])

  // ── Pagination ──
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safeCurrentPage = Math.min(currentPage, totalPages)
  const paginatedItems = filtered.slice((safeCurrentPage - 1) * PAGE_SIZE, safeCurrentPage * PAGE_SIZE)

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [searchQuery, minConfidence, speciesFilter])

  // ── Selection ──
  const allSelected = filtered.length > 0 && filtered.every((e) => selectedIds.has(e.id))
  const someSelected = selectedIds.size > 0

  const toggleAll = useCallback(() => {
    if (allSelected) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(filtered.map((e) => e.id)))
    }
  }, [allSelected, filtered])

  const toggleOne = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  // ── Sort handler (persists to localStorage) ──
  const handleSort = useCallback(
    (field: SortField) => {
      let newOrder: SortOrder = "desc"
      if (sortField === field) {
        newOrder = sortOrder === "asc" ? "desc" : "asc"
        setSortOrder(newOrder)
      } else {
        setSortField(field)
        setSortOrder("desc")
      }
      try { localStorage.setItem(SORT_KEY, JSON.stringify({ field, order: newOrder })) } catch { /* */ }
    },
    [sortField, sortOrder]
  )

  // ── Delete (with confirmation) ──
  const requestDeleteSelected = useCallback(() => {
    setConfirmDelete({ ids: [...selectedIds], label: `${selectedIds.size} record(s)` })
  }, [selectedIds])

  const requestDeleteSingle = useCallback((id: string) => {
    setConfirmDelete({ ids: [id], label: "this record" })
  }, [])

  const executeDelete = useCallback(() => {
    if (!confirmDelete) return
    deleteDetections(confirmDelete.ids)
    setHistory(getDetections())
    setSelectedIds((prev) => {
      const n = new Set(prev)
      confirmDelete.ids.forEach((id) => n.delete(id))
      return n
    })
    toast({ title: `${confirmDelete.ids.length} record(s) deleted`, description: "Items removed from history" })
    setConfirmDelete(null)
  }, [confirmDelete, toast])

  // ── Audio preview ──
  const togglePlay = useCallback((entry: DetectionRecord) => {
    if (!entry.audioUrl) return
    if (playingId === entry.id) {
      audioRef.current?.pause()
      setPlayingId(null)
      return
    }
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = entry.audioUrl
      audioRef.current.play()
    } else {
      const a = new Audio(entry.audioUrl)
      audioRef.current = a
      a.play()
    }
    setPlayingId(entry.id)
  }, [playingId])

  // Clean up audio on unmount
  useEffect(() => {
    const a = audioRef.current
    return () => { a?.pause() }
  }, [])

  // Handle audio ended
  useEffect(() => {
    const handler = () => setPlayingId(null)
    const a = audioRef.current
    a?.addEventListener("ended", handler)
    return () => { a?.removeEventListener("ended", handler) }
  }, [playingId])

  // ── Toggle species filter ──
  const toggleSpecies = useCallback((species: string) => {
    setSpeciesFilter((prev) => {
      const next = new Set(prev)
      if (next.has(species)) next.delete(species)
      else next.add(species)
      return next
    })
  }, [])

  // ── Export (JSON + CSV) ──
  const downloadBlob = useCallback((blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  const exportSelected = useCallback(() => {
    const data = history.filter((e) => selectedIds.has(e.id))
    const json = JSON.stringify(data, null, 2)
    downloadBlob(new Blob([json], { type: "application/json" }), "birdsense-history-export.json")
    toast({ title: "Selected records exported", description: `${data.length} item(s) saved as JSON` })
  }, [history, selectedIds, toast, downloadBlob])

  const exportAll = useCallback(() => {
    const json = JSON.stringify(history, null, 2)
    downloadBlob(new Blob([json], { type: "application/json" }), "birdsense-history-all.json")
    toast({ title: "All records exported", description: `${history.length} item(s) saved as JSON` })
  }, [history, toast, downloadBlob])

  const exportCsv = useCallback(() => {
    const header = "filename,date,time,species,scientific_name,confidence,duration_sec,segments\n"
    const rows = history.map((e) =>
      `"${e.filename}",${e.date},${e.time},"${e.topSpecies}","${e.topScientific}",${e.topConfidence},${e.duration},${e.segments.length}`
    ).join("\n")
    downloadBlob(new Blob([header + rows], { type: "text/csv" }), "birdsense-history.csv")
    toast({ title: "CSV exported", description: `${history.length} records saved` })
  }, [history, toast, downloadBlob])

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "TEXTAREA") return
      if (e.key === "Escape") {
        audioRef.current?.pause()
        setPlayingId(null)
        setConfirmDelete(null)
      }
      if (e.key === "Delete" && selectedIds.size > 0) {
        requestDeleteSelected()
      }
      if (e.key === "ArrowLeft" && !e.metaKey && !e.ctrlKey) {
        setCurrentPage((p) => Math.max(1, p - 1))
      }
      if (e.key === "ArrowRight" && !e.metaKey && !e.ctrlKey) {
        setCurrentPage((p) => Math.min(totalPages, p + 1))
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [selectedIds, requestDeleteSelected, totalPages])

  // ── Sort indicator component ──
  const SortIndicator = ({ field }: { field: SortField }) => {
    if (sortField !== field) return <ChevronDown size={10} className="text-foreground/20 ml-1" />
    return sortOrder === "asc" ? (
      <ChevronUp size={10} className="text-accent ml-1" />
    ) : (
      <ChevronDown size={10} className="text-accent ml-1" />
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
              <History size={16} className="text-accent" />
              <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
                DETECTION HISTORY
              </h1>
            </div>
            <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
              {totalCount} RECORDS • {selectedIds.size} SELECTED
            </p>
          </div>
        </div>
      </div>

      <main className="flex-1 px-4 lg:px-6 py-6 lg:py-8 relative z-10 max-w-7xl mx-auto w-full space-y-5">
        {/* ────── QUICK STATS ────── */}
        <div className="flex gap-3 justify-end">
          <div className="border-2 border-foreground px-4 py-2 min-w-[70px] text-center">
            <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground block">TOTAL</span>
            <span className="font-mono text-xl font-bold text-foreground">{totalCount}</span>
          </div>
          <div className="border-2 border-foreground px-4 py-2 min-w-[70px] text-center">
            <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground block">AVG</span>
            <span className="font-mono text-xl font-bold text-accent">{avgConf}%</span>
          </div>
        </div>

        {/* ────── CONTROLS BAR ────── */}
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Search */}
          <div className="flex-1 flex items-center border-2 border-foreground bg-background">
            <Search size={14} className="text-muted-foreground ml-3 shrink-0" />
            <input
              type="text"
              placeholder="SEARCH FILES OR SPECIES..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 bg-transparent font-mono text-xs tracking-wider uppercase text-foreground placeholder:text-muted-foreground/40 px-3 py-2.5 outline-none focus:text-accent"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="mr-2 text-muted-foreground hover:text-foreground cursor-pointer"
              >
                <X size={12} />
              </button>
            )}
          </div>

          {/* Filter + Export buttons */}
          <div className="flex gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setFilterOpen(!filterOpen)}
              className={`flex items-center gap-2 px-4 py-2 border-2 font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none ${filterOpen || minConfidence > 0 || speciesFilter.size > 0
                ? "border-accent bg-accent text-white"
                : "border-foreground bg-background text-foreground hover:bg-muted"
                }`}
            >
              <Filter size={12} />
              FILTER
              {minConfidence > 0 && <span className="text-[9px]">({minConfidence}%+)</span>}
            </button>
            <button
              type="button"
              onClick={exportAll}
              className="flex items-center gap-2 px-4 py-2 border-2 border-foreground bg-background text-foreground font-mono text-xs tracking-[0.15em] uppercase cursor-pointer hover:bg-muted transition-none"
            >
              <Download size={12} />
              JSON
            </button>
            <button
              type="button"
              onClick={exportCsv}
              className="flex items-center gap-2 px-4 py-2 border-2 border-foreground bg-background text-foreground font-mono text-xs tracking-[0.15em] uppercase cursor-pointer hover:bg-muted transition-none"
            >
              <FileText size={12} />
              CSV
            </button>
          </div>
        </div>

        {/* ────── FILTER PANEL ────── */}
        {filterOpen && (
          <div className="border-2 border-accent bg-background p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
                FILTERS
              </span>
              <button
                type="button"
                onClick={() => { setMinConfidence(0); setSpeciesFilter(new Set()); setFilterOpen(false) }}
                className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground cursor-pointer"
              >
                CLEAR FILTERS
              </button>
            </div>
            <div className="space-y-1.5">
              <label className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground flex items-center justify-between">
                MIN CONFIDENCE
                <span className="text-foreground font-bold">{minConfidence}%</span>
              </label>
              <input
                type="range"
                min={0}
                max={90}
                step={5}
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                className="w-full h-1.5 bg-foreground/10 appearance-none cursor-pointer accent-accent"
              />
              <div className="flex justify-between font-mono text-[8px] text-muted-foreground/50">
                <span>0%</span>
                <span>45%</span>
                <span>90%</span>
              </div>
            </div>

            {/* ── Species filter ── */}
            {allSpecies.length > 0 && (
              <div className="space-y-1.5">
                <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground flex items-center justify-between">
                  SPECIES
                  {speciesFilter.size > 0 && (
                    <button
                      type="button"
                      onClick={() => setSpeciesFilter(new Set())}
                      className="text-accent hover:text-foreground cursor-pointer"
                    >
                      CLEAR ({speciesFilter.size})
                    </button>
                  )}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {allSpecies.map((sp) => (
                    <button
                      key={sp}
                      type="button"
                      onClick={() => toggleSpecies(sp)}
                      className={`px-2.5 py-1 border font-mono text-[9px] tracking-[0.1em] uppercase cursor-pointer transition-none ${speciesFilter.has(sp)
                        ? "border-accent bg-accent text-white"
                        : "border-foreground/20 text-muted-foreground hover:border-foreground hover:text-foreground"
                        }`}
                    >
                      {sp}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ────── BULK ACTIONS BAR ────── */}
        {someSelected && (
          <div className="flex items-center gap-3 border-2 border-accent bg-accent/5 px-4 py-2.5">
            <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent font-bold">
              {selectedIds.size} SELECTED
            </span>
            <div className="flex-1" />
            <button
              type="button"
              onClick={exportSelected}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-foreground/30 font-mono text-[10px] tracking-[0.15em] uppercase text-foreground hover:border-accent hover:text-accent cursor-pointer transition-none"
            >
              <Download size={10} /> EXPORT
            </button>
            <button
              type="button"
              onClick={requestDeleteSelected}
              className="flex items-center gap-1.5 px-3 py-1.5 border border-red-500/30 font-mono text-[10px] tracking-[0.15em] uppercase text-red-500 hover:bg-red-500/10 cursor-pointer transition-none"
            >
              <Trash2 size={10} /> DELETE
            </button>
            <button
              type="button"
              onClick={() => setSelectedIds(new Set())}
              className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground cursor-pointer"
            >
              CLEAR
            </button>
          </div>
        )}

        {/* ────── CONFIRM DELETE DIALOG ────── */}
        {confirmDelete && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
            <div className="border-2 border-foreground bg-background p-6 shadow-[6px_6px_0px_0px_hsl(var(--foreground))] max-w-sm w-full mx-4 space-y-4">
              <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">CONFIRM DELETE</span>
              <p className="font-mono text-xs tracking-wider text-foreground">
                Delete {confirmDelete.label}? This cannot be undone.
              </p>
              <div className="flex gap-2 justify-end">
                <button
                  type="button"
                  onClick={() => setConfirmDelete(null)}
                  className="px-4 py-2 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
                >
                  CANCEL
                </button>
                <button
                  type="button"
                  onClick={executeDelete}
                  className="px-4 py-2 border-2 border-red-500 bg-red-500 text-white font-mono text-xs tracking-[0.15em] uppercase hover:bg-red-600 cursor-pointer transition-none"
                >
                  DELETE
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ────── LOADING SKELETON ────── */}
        {loading ? (
          <div className="border-2 border-foreground bg-background p-4 space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 animate-pulse">
                <div className="w-4 h-4 bg-foreground/10" />
                <div className="flex-1 space-y-1.5">
                  <div className="h-3 bg-foreground/10 w-3/4" />
                  <div className="h-2 bg-foreground/5 w-1/2" />
                </div>
                <div className="w-16 h-3 bg-foreground/10" />
                <div className="w-10 h-3 bg-foreground/10" />
              </div>
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="border-2 border-foreground/20 bg-background py-16 flex flex-col items-center justify-center gap-4 text-center">
            <AlertCircle size={32} className="text-muted-foreground/40" />
            <p className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
              {history.length === 0 ? "NO DETECTION HISTORY" : "NO RESULTS MATCH FILTERS"}
            </p>
            <p className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground/60">
              {history.length === 0
                ? "Run an analysis on the Analyze page to see results here"
                : "Try adjusting your search or filter settings"}
            </p>
            {history.length === 0 && (
              <Link
                href="/analyze"
                className="mt-2 flex items-center gap-2 px-5 py-2.5 border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.15em] uppercase font-bold shadow-[3px_3px_0px_0px_rgba(234,88,12,0.3)] hover:bg-background hover:text-accent cursor-pointer transition-none"
              >
                <Zap size={14} /> GO TO ANALYZE
              </Link>
            )}
          </div>
        ) : (
          <>
            {/* ── Desktop Table ── */}
            <div className="hidden md:block border-2 border-foreground bg-background overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b-2 border-foreground">
                    <th className="w-10 px-3 py-2.5 text-center">
                      <input
                        type="checkbox"
                        checked={allSelected}
                        onChange={toggleAll}
                        className="accent-accent cursor-pointer"
                      />
                    </th>
                    <th
                      className="text-left px-3 py-2.5 cursor-pointer select-none"
                      onClick={() => handleSort("date")}
                    >
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground flex items-center hover:text-foreground">
                        FILE <SortIndicator field="date" />
                      </span>
                    </th>
                    <th
                      className="text-left px-3 py-2.5 cursor-pointer select-none"
                      onClick={() => handleSort("date")}
                    >
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground flex items-center hover:text-foreground">
                        DATE/TIME <SortIndicator field="date" />
                      </span>
                    </th>
                    <th
                      className="text-left px-3 py-2.5 cursor-pointer select-none"
                      onClick={() => handleSort("species")}
                    >
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground flex items-center hover:text-foreground">
                        SPECIES <SortIndicator field="species" />
                      </span>
                    </th>
                    <th
                      className="text-left px-3 py-2.5 cursor-pointer select-none w-[140px]"
                      onClick={() => handleSort("confidence")}
                    >
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground flex items-center hover:text-foreground">
                        CONF <SortIndicator field="confidence" />
                      </span>
                    </th>
                    <th
                      className="text-left px-3 py-2.5 cursor-pointer select-none"
                      onClick={() => handleSort("duration")}
                    >
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground flex items-center hover:text-foreground">
                        DUR <SortIndicator field="duration" />
                      </span>
                    </th>
                    <th className="text-left px-3 py-2.5">
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">
                        SEGS
                      </span>
                    </th>
                    <th className="text-left px-3 py-2.5">
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">
                        ACTIONS
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedItems.map((entry) => {
                    const isSelected = selectedIds.has(entry.id)
                    const confColor = getConfColor(entry.topConfidence)
                    const isPlaying = playingId === entry.id
                    return (
                      <tr
                        key={entry.id}
                        className={`border-b border-foreground/10 hover:bg-muted/30 transition-none cursor-pointer ${isSelected ? "bg-accent/5" : ""} ${isPlaying ? "bg-accent/10" : ""}`}
                        onClick={() => router.push(`/results/${entry.id}`)}
                      >
                        <td className="px-3 py-2.5 text-center" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleOne(entry.id)}
                            className="accent-accent cursor-pointer"
                          />
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="font-mono text-[11px] tracking-wider text-foreground truncate block max-w-[180px]" title={entry.filename}>
                            {entry.filename}
                          </span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="font-mono text-[11px] tracking-wider text-foreground block">{entry.date}</span>
                          <span className="font-mono text-[9px] tracking-wider text-muted-foreground">{entry.time}</span>
                          <span className="font-mono text-[8px] tracking-wider text-muted-foreground/50 block">{timeAgo(entry.date, entry.time)}</span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="font-mono text-[11px] tracking-wider text-foreground block font-bold">{entry.topSpecies}</span>
                          <span className="font-mono text-[9px] tracking-wider text-muted-foreground italic">{entry.topScientific}</span>
                        </td>
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-2.5 bg-foreground/5 overflow-hidden max-w-[80px]">
                              <div
                                className="h-full"
                                style={{ width: `${entry.topConfidence}%`, backgroundColor: confColor }}
                              />
                            </div>
                            <span className="font-mono text-[11px] font-bold shrink-0" style={{ color: confColor }}>
                              {entry.topConfidence}%
                            </span>
                          </div>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="font-mono text-[11px] tracking-wider text-foreground">{fmtDuration(entry.duration)}</span>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="font-mono text-[11px] font-bold text-accent">{entry.segments.length}</span>
                        </td>
                        <td className="px-3 py-2.5" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center gap-1">
                            {entry.audioUrl && (
                              <button
                                type="button"
                                onClick={() => togglePlay(entry)}
                                className={`w-7 h-7 border flex items-center justify-center cursor-pointer transition-none ${isPlaying ? "border-accent text-accent" : "border-foreground/20 text-muted-foreground hover:text-accent hover:border-accent/50"}`}
                                title={isPlaying ? "Pause preview" : "Play preview"}
                              >
                                {isPlaying ? <Pause size={10} /> : <Play size={10} />}
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => router.push(`/results/${entry.id}`)}
                              className="w-7 h-7 border border-foreground/20 flex items-center justify-center text-muted-foreground hover:text-accent hover:border-accent/50 cursor-pointer transition-none"
                              title="View details"
                            >
                              <Eye size={10} />
                            </button>
                            <button
                              type="button"
                              onClick={() => requestDeleteSingle(entry.id)}
                              className="w-7 h-7 border border-foreground/20 flex items-center justify-center text-muted-foreground hover:text-red-500 hover:border-red-500/50 cursor-pointer transition-none"
                              title="Delete"
                            >
                              <Trash2 size={10} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* ── Mobile Cards ── */}
            <div className="md:hidden space-y-3">
              {paginatedItems.map((entry) => {
                const isSelected = selectedIds.has(entry.id)
                const confColor = getConfColor(entry.topConfidence)
                const isPlaying = playingId === entry.id
                return (
                  <div
                    key={entry.id}
                    className={`border-2 border-foreground bg-background cursor-pointer ${isSelected ? "border-l-4 border-l-accent" : ""} ${isPlaying ? "bg-accent/5" : ""}`}
                    onClick={() => router.push(`/results/${entry.id}`)}
                  >
                    <div className="flex items-start gap-3 px-3 py-3">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleOne(entry.id)}
                        onClick={(e) => e.stopPropagation()}
                        className="accent-accent cursor-pointer mt-1 shrink-0"
                      />
                      <div className="flex-1 min-w-0 space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-xs tracking-wider text-foreground font-bold truncate">
                            {entry.topSpecies}
                          </span>
                          <span className="font-mono text-xs font-bold shrink-0" style={{ color: confColor }}>
                            {entry.topConfidence}%
                          </span>
                        </div>
                        <span className="font-mono text-[9px] tracking-wider text-muted-foreground italic block">
                          {entry.topScientific}
                        </span>
                        <div className="h-2 bg-foreground/5 overflow-hidden">
                          <div className="h-full" style={{ width: `${entry.topConfidence}%`, backgroundColor: confColor }} />
                        </div>
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5">
                          <span className="font-mono text-[9px] tracking-wider text-muted-foreground">
                            {entry.filename}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[9px] tracking-wider text-muted-foreground">
                          <span>{entry.date} {entry.time}</span>
                          <span className="text-muted-foreground/50">{timeAgo(entry.date, entry.time)}</span>
                          <span>DUR {fmtDuration(entry.duration)}</span>
                          <span className="text-accent">{entry.segments.length} SEGS</span>
                        </div>
                      </div>
                    </div>
                    {/* Card actions */}
                    <div className="border-t border-foreground/10 px-3 py-2 flex gap-2" onClick={(e) => e.stopPropagation()}>
                      {entry.audioUrl && (
                        <button
                          type="button"
                          onClick={() => togglePlay(entry)}
                          className={`flex items-center gap-1 px-2 py-1 border font-mono text-[9px] uppercase cursor-pointer transition-none ${isPlaying ? "border-accent text-accent" : "border-foreground/20 text-muted-foreground hover:text-accent"}`}
                        >
                          {isPlaying ? <><Pause size={9} /> PAUSE</> : <><Play size={9} /> PLAY</>}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => router.push(`/results/${entry.id}`)}
                        className="flex items-center gap-1 px-2 py-1 border border-foreground/20 font-mono text-[9px] uppercase text-muted-foreground hover:text-accent cursor-pointer transition-none"
                      >
                        <Eye size={9} /> VIEW
                      </button>
                      <div className="flex-1" />
                      <button
                        type="button"
                        onClick={() => requestDeleteSingle(entry.id)}
                        className="flex items-center gap-1 px-2 py-1 border border-red-500/20 font-mono text-[9px] uppercase text-red-500 hover:bg-red-500/10 cursor-pointer transition-none"
                      >
                        <Trash2 size={9} />
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>
          </>
        )}

        {/* ────── PAGINATION ────── */}
        <div className="flex items-center justify-between border-2 border-foreground bg-background px-4 py-2.5">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            SHOWING {((safeCurrentPage - 1) * PAGE_SIZE) + 1}–{Math.min(safeCurrentPage * PAGE_SIZE, filtered.length)} OF {filtered.length} RECORDS
          </span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={safeCurrentPage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              className={`w-8 h-8 border flex items-center justify-center font-mono text-xs cursor-pointer transition-none ${safeCurrentPage <= 1 ? "border-foreground/20 text-muted-foreground/30 cursor-not-allowed" : "border-foreground/40 text-foreground hover:bg-muted"}`}
            >
              <ChevronLeft size={14} />
            </button>
            {Array.from({ length: totalPages }, (_, i) => i + 1).slice(
              Math.max(0, safeCurrentPage - 3),
              Math.min(totalPages, safeCurrentPage + 2)
            ).map((page) => (
              <button
                key={page}
                type="button"
                onClick={() => setCurrentPage(page)}
                className={`w-8 h-8 border-2 flex items-center justify-center font-mono text-xs font-bold cursor-pointer transition-none ${page === safeCurrentPage
                  ? "border-accent bg-accent text-white"
                  : "border-foreground/20 text-foreground hover:bg-muted"
                  }`}
              >
                {page}
              </button>
            ))}
            <button
              type="button"
              disabled={safeCurrentPage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              className={`w-8 h-8 border flex items-center justify-center font-mono text-xs cursor-pointer transition-none ${safeCurrentPage >= totalPages ? "border-foreground/20 text-muted-foreground/30 cursor-not-allowed" : "border-foreground/40 text-foreground hover:bg-muted"}`}
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>

        {/* ────── STATUS BAR ────── */}
        <div className="border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">
            SYS_STATUS:
          </span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            HISTORY LOADED
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>
      </main>
    </div>
  )
}
