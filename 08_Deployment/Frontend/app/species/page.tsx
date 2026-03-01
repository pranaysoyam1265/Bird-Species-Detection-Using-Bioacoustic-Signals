"use client"

import { useState, useMemo, useCallback, useEffect, useRef, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import { motion } from "framer-motion"
import { getAccuracyColor } from "@/lib/format"
import { SPECIES_META, type SpeciesMeta } from "@/lib/species-meta"
import {
  Search,
  X,
  ChevronDown,
  Bird,
  ExternalLink,
  Zap,
  BarChart3,
  Star,
  GitCompareArrows,
  Activity,
} from "lucide-react"
import { getDetectionCountsBySpecies } from "@/lib/detection-store"

/* ═══════════════════════════════════════════════════════════════
   TYPES
   ═══════════════════════════════════════════════════════════════ */

interface Species {
  id: string
  commonName: string
  scientificName: string
  family: string
  habitat: string
  callType: string
  status: "common" | "uncommon" | "rare"
}

/* ═══════════════════════════════════════════════════════════════
   SPECIES DATA — built from the real 87-species ML model
   ═══════════════════════════════════════════════════════════════ */

const ALL_SPECIES: Species[] = Object.values(SPECIES_META)
  .map((meta, idx) => ({
    id: String(idx + 1),
    commonName: meta.name,
    scientificName: meta.scientificName,
    family: meta.family,
    habitat: meta.habitat,
    callType: meta.callType,
    status: meta.status,
  }))
  .sort((a, b) => a.commonName.localeCompare(b.commonName))

const FAMILIES = [...new Set(ALL_SPECIES.map((s) => s.family))].sort()

type SortOption = "name-asc" | "name-desc" | "family"
type ViewMode = "grid" | "list"

const SORT_LABELS: Record<SortOption, string> = {
  "name-asc": "NAME A→Z",
  "name-desc": "NAME Z→A",
  "family": "FAMILY",
}

/* ═══════════════════════════════════════════════════════════════
   EXTERNAL LINKS
   ═══════════════════════════════════════════════════════════════ */

const EXTERNAL_LINKS = {
  eBird: (name: string) => `https://ebird.org/species/${name.toLowerCase().replace(/ /g, "")}`,
  xenoCanto: (name: string) => `https://xeno-canto.org/explore?query=${encodeURIComponent(name)}`,
  wikipedia: (name: string) => `https://en.wikipedia.org/wiki/${name.replace(/ /g, "_")}`,
  allAboutBirds: (name: string) => `https://www.allaboutbirds.org/guide/${name.replace(/ /g, "_")}`,
}

/* ═══════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════ */

const TOTAL_FAMILIES = FAMILIES.length
const BATCH_SIZE = 20

/* ═══════════════════════════════════════════════════════════════
   PAGE COMPONENT
   ═══════════════════════════════════════════════════════════════ */

/* ── Loading skeleton ── */
function SpeciesSkeleton() {
  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      <Navbar />
      <div className="px-4 lg:px-6 pt-4 lg:pt-6"><div className="h-8 w-64 bg-muted/50 animate-pulse" /></div>
      <main className="flex-1 px-4 lg:px-6 py-6 max-w-7xl mx-auto w-full space-y-5">
        <div className="border-2 border-foreground/20 bg-background p-5 h-40 animate-pulse" />
        <div className="h-10 bg-muted/30 animate-pulse" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="border-2 border-foreground/10 bg-background p-4 h-48 animate-pulse" />
          ))}
        </div>
      </main>
    </div>
  )
}

/* Wrapper with Suspense for useSearchParams */
export default function SpeciesPage() {
  return (
    <Suspense fallback={<SpeciesSkeleton />}>
      <SpeciesPageContent />
    </Suspense>
  )
}

function SpeciesPageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()

  /* ── State (hydrate from URL) ── */
  const [searchQuery, setSearchQuery] = useState(searchParams.get("q") || "")
  const [sortBy, setSortBy] = useState<SortOption>((searchParams.get("sort") as SortOption) || "name-asc")
  const [selectedFamily, setSelectedFamily] = useState(searchParams.get("family") || "All")

  /* ── Load-more pagination ── */
  const [visibleCount, setVisibleCount] = useState(BATCH_SIZE)
  const [selectedSpecies, setSelectedSpecies] = useState<Species | null>(null)
  const [sortDropdownOpen, setSortDropdownOpen] = useState(false)
  const sortRef = useRef<HTMLDivElement>(null)

  /* ── Detection counts from history ── */
  const [detectionCounts, setDetectionCounts] = useState<Record<string, number>>({})
  useEffect(() => {
    setDetectionCounts(getDetectionCountsBySpecies())
  }, [])

  /* ── Favorites ── */
  const [favorites, setFavorites] = useState<Set<string>>(new Set())
  useEffect(() => {
    try {
      const stored = localStorage.getItem("birdsense-species-favorites")
      if (stored) setFavorites(new Set(JSON.parse(stored)))
    } catch { /* */ }
  }, [])
  const toggleFavorite = useCallback((id: string, e?: React.MouseEvent) => {
    e?.stopPropagation()
    setFavorites((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      localStorage.setItem("birdsense-species-favorites", JSON.stringify([...next]))
      return next
    })
  }, [])

  /* ── Compare mode ── */
  const [compareMode, setCompareMode] = useState(false)
  const [compareIds, setCompareIds] = useState<string[]>([])
  const toggleCompare = useCallback((id: string, e?: React.MouseEvent) => {
    e?.stopPropagation()
    setCompareIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 3 ? [...prev, id] : prev)
  }, [])
  const compareSpecies = compareIds.map((id) => ALL_SPECIES.find((s) => s.id === id)!).filter(Boolean)

  /* ── Sync state → URL params ── */
  useEffect(() => {
    const params = new URLSearchParams()
    if (searchQuery) params.set("q", searchQuery)
    if (sortBy !== "name-asc") params.set("sort", sortBy)
    if (selectedFamily !== "All") params.set("family", selectedFamily)
    const qs = params.toString()
    router.replace(qs ? `/species?${qs}` : "/species", { scroll: false })
  }, [searchQuery, sortBy, selectedFamily, router])

  /* ── Close sort dropdown on outside click ── */
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (sortRef.current && !sortRef.current.contains(e.target as Node)) {
        setSortDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", onClick)
    return () => document.removeEventListener("mousedown", onClick)
  }, [])

  /* ── Close modal with Escape ── */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setSelectedSpecies(null); setCompareMode(false) }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [])

  /* ── Filtered + sorted list ── */
  const filteredAll = useMemo(() => {
    const q = searchQuery.toLowerCase()
    let items = ALL_SPECIES.filter((s) => {
      const matchesSearch =
        s.commonName.toLowerCase().includes(q) ||
        s.scientificName.toLowerCase().includes(q)
      const matchesFamily = selectedFamily === "All" || s.family === selectedFamily
      return matchesSearch && matchesFamily
    })

    items.sort((a, b) => {
      switch (sortBy) {
        case "name-asc": return a.commonName.localeCompare(b.commonName)
        case "name-desc": return b.commonName.localeCompare(a.commonName)
        case "family": return a.family.localeCompare(b.family) || a.commonName.localeCompare(b.commonName)
        default: return 0
      }
    })
    return items
  }, [searchQuery, selectedFamily, sortBy])

  /* Reset visible count whenever the filter/search changes */
  useEffect(() => {
    setVisibleCount(BATCH_SIZE)
  }, [searchQuery, selectedFamily, sortBy])

  const visibleSpecies = useMemo(() => filteredAll.slice(0, visibleCount), [filteredAll, visibleCount])
  const hasMore = visibleCount < filteredAll.length

  const openDetail = useCallback((s: Species) => setSelectedSpecies(s), [])
  const closeDetail = useCallback(() => setSelectedSpecies(null), [])

  /* ══════════════════════════════════════════
     STATUS DISTRIBUTION
     ══════════════════════════════════════════ */
  const statusCounts = useMemo(() => {
    const counts = { common: 0, uncommon: 0, rare: 0 }
    ALL_SPECIES.forEach((s) => { counts[s.status]++ })
    return counts
  }, [])

  /* ══════════════════════════════════════════
     RENDER
     ══════════════════════════════════════════ */

  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      {/* Radial vignette */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />

      <Navbar />

      {/* ── Page Header ── */}
      <div className="px-4 lg:px-6 pt-4 lg:pt-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <NavSidebar />
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <Bird size={16} className="text-accent" />
                <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
                  SPECIES DATABASE
                </h1>
              </div>
              <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
                {ALL_SPECIES.length} SPECIES • {TOTAL_FAMILIES} FAMILIES • REAL ML MODEL DATA
              </p>
            </div>
          </div>
        </div>
      </div>

      <main className="flex-1 px-4 lg:px-6 py-6 lg:py-8 relative z-10 max-w-7xl mx-auto w-full space-y-5">

        {/* ────── STATUS DISTRIBUTION ────── */}
        <div className="border-2 border-foreground bg-background p-4 sm:p-5">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 size={14} className="text-accent" />
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground font-bold">
              SPECIES STATUS DISTRIBUTION
            </span>
          </div>
          <div className="flex items-end gap-4 h-24 sm:h-32">
            {([
              { label: "COMMON", count: statusCounts.common, color: "#22c55e" },
              { label: "UNCOMMON", count: statusCounts.uncommon, color: "#f59e0b" },
              { label: "RARE", count: statusCounts.rare, color: "#ef4444" },
            ]).map((item) => {
              const maxCount = Math.max(statusCounts.common, statusCounts.uncommon, statusCounts.rare, 1)
              return (
                <div key={item.label} className="flex-1 flex flex-col items-center gap-1 h-full justify-end">
                  <span className="font-mono text-sm font-bold" style={{ color: item.color }}>{item.count}</span>
                  <div
                    className="w-full min-h-[2px]"
                    style={{ height: `${(item.count / maxCount) * 100}%`, backgroundColor: item.color }}
                  />
                  <span className="font-mono text-[8px] sm:text-[9px] text-muted-foreground/60 text-center whitespace-nowrap">
                    {item.label}
                  </span>
                </div>
              )
            })}
          </div>
          <div className="border-t border-foreground/10 mt-3 pt-3 flex flex-wrap gap-x-6 gap-y-1">
            <span className="font-mono text-[10px] tracking-[0.1em] uppercase text-muted-foreground">
              TOTAL: <span className="text-foreground font-bold">{ALL_SPECIES.length}</span> SPECIES
            </span>
            <span className="font-mono text-[10px] tracking-[0.1em] uppercase text-muted-foreground">
              FAMILIES: <span className="text-accent font-bold">{TOTAL_FAMILIES}</span>
            </span>
          </div>
        </div>

        {/* ────── SEARCH BAR ────── */}
        <div className="flex items-center border-2 border-foreground bg-background">
          <Search size={14} className="text-muted-foreground ml-3 shrink-0" />
          <input
            type="text"
            placeholder="SEARCH SPECIES BY NAME..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 bg-transparent font-mono text-xs tracking-wider uppercase text-foreground placeholder:text-muted-foreground/40 px-3 py-2.5 outline-none focus:text-accent"
            aria-label="Search species by name"
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

        {/* ────── CONTROLS ROW ────── */}
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Sort dropdown */}
          <div className="relative shrink-0" ref={sortRef}>
            <button
              type="button"
              onClick={() => setSortDropdownOpen((o) => !o)}
              className={`flex items-center gap-2 px-4 py-2 border-2 font-mono text-[10px] tracking-[0.15em] uppercase cursor-pointer transition-none ${sortDropdownOpen ? "border-accent text-accent" : "border-foreground bg-background text-foreground hover:bg-muted"}`}
            >
              SORT: {SORT_LABELS[sortBy]}
              <ChevronDown size={10} className={sortDropdownOpen ? "rotate-180" : ""} />
            </button>
            {sortDropdownOpen && (
              <div className="absolute top-full left-0 mt-1 w-48 border-2 border-foreground bg-background z-40 shadow-[4px_4px_0px_0px_rgba(234,88,12,0.2)]">
                {(Object.keys(SORT_LABELS) as SortOption[]).map((key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => { setSortBy(key); setSortDropdownOpen(false) }}
                    className={`block w-full text-left px-4 py-2.5 font-mono text-[10px] tracking-[0.15em] uppercase cursor-pointer transition-none border-b border-foreground/10 last:border-b-0 ${sortBy === key ? "bg-accent/10 text-accent font-bold" : "text-foreground hover:bg-muted"}`}
                  >
                    {SORT_LABELS[key]}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Compare toggle */}
          <button
            type="button"
            onClick={() => { setCompareMode((p) => !p); if (compareMode) setCompareIds([]) }}
            className={`flex items-center gap-1.5 px-3 py-2 border-2 font-mono text-[10px] tracking-[0.15em] uppercase cursor-pointer transition-none ${compareMode ? "border-accent bg-accent text-white" : "border-foreground bg-background text-foreground hover:bg-muted"}`}
          >
            <GitCompareArrows size={12} /> COMPARE{compareIds.length > 0 ? ` (${compareIds.length})` : ""}
          </button>

          <div className="flex-1" />

          {/* Tiny stats */}
          <div className="flex items-center gap-1">
            <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
              {filteredAll.length} RESULTS
            </span>
          </div>
        </div>

        {/* ────── FAMILY FILTERS ────── */}
        <div className="flex gap-1.5 overflow-x-auto pb-1 -mx-4 px-4 sm:mx-0 sm:px-0 sm:flex-wrap scrollbar-none">
          <button
            type="button"
            onClick={() => setSelectedFamily("All")}
            className={`px-3 py-1.5 border-2 font-mono text-[10px] tracking-[0.15em] uppercase cursor-pointer transition-none whitespace-nowrap shrink-0 ${selectedFamily === "All" ? "border-accent bg-accent text-white" : "border-foreground hover:bg-muted"}`}
          >
            ALL
          </button>
          {FAMILIES.map((fam) => (
            <button
              key={fam}
              type="button"
              onClick={() => setSelectedFamily(fam)}
              className={`px-3 py-1.5 border-2 font-mono text-[10px] tracking-[0.15em] uppercase cursor-pointer transition-none whitespace-nowrap shrink-0 ${selectedFamily === fam ? "border-accent bg-accent text-white" : "border-foreground hover:bg-muted"}`}
            >
              {fam}
            </button>
          ))}
        </div>

        {/* ────── CONTENT: GRID or LIST ────── */}
        {filteredAll.length === 0 ? (
          <div className="border-2 border-foreground/20 bg-background py-16 flex flex-col items-center justify-center gap-4">
            <Bird size={32} className="text-muted-foreground/40" />
            <p className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
              NO SPECIES MATCH YOUR SEARCH
            </p>
            <p className="font-mono text-[10px] text-muted-foreground/60">
              Try adjusting your search or filter
            </p>
          </div>
        ) : (
          /* ══════ GRID VIEW ══════ */
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {visibleSpecies.map((sp, idx) => {
                const isFav = favorites.has(sp.id)
                const isComparing = compareIds.includes(sp.id)
                return (
                  <motion.div
                    key={sp.id}
                    role="button"
                    tabIndex={0}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: Math.min(idx, BATCH_SIZE - 1) * 0.03 }}
                    onClick={() => compareMode ? toggleCompare(sp.id) : openDetail(sp)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault()
                        compareMode ? toggleCompare(sp.id) : openDetail(sp)
                      }
                    }}
                    className={`border-2 bg-background p-4 text-left relative group cursor-pointer transition-none ${isComparing ? "border-accent shadow-[4px_4px_0px_0px_rgba(234,88,12,0.3)]" : "border-foreground hover:border-accent hover:shadow-[4px_4px_0px_0px_rgba(234,88,12,0.3)]"}`}
                  >
                    {/* Corner accents */}
                    <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent opacity-0 group-hover:opacity-100" />
                    <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent opacity-0 group-hover:opacity-100" />
                    <span className="absolute -bottom-[3px] -left-[3px] w-2 h-2 bg-accent opacity-0 group-hover:opacity-100" />
                    <span className="absolute -bottom-[3px] -right-[3px] w-2 h-2 bg-accent opacity-0 group-hover:opacity-100" />

                    {/* Top row: bird icon + favorite star */}
                    <div className="flex items-start justify-between mb-3">
                      <div className="w-12 h-12 border-2 border-foreground/20 flex items-center justify-center group-hover:border-accent/50">
                        <Bird size={24} className="text-muted-foreground/40 group-hover:text-accent" />
                      </div>
                      {!compareMode && (
                        <button type="button" onClick={(e) => toggleFavorite(sp.id, e)} className="p-1 cursor-pointer" aria-label="Toggle favorite">
                          <Star size={14} className={isFav ? "fill-accent text-accent" : "text-foreground/20 hover:text-accent"} />
                        </button>
                      )}
                      {compareMode && (
                        <span className={`w-5 h-5 border-2 flex items-center justify-center text-[10px] font-bold ${isComparing ? "border-accent bg-accent text-white" : "border-foreground/30"}`}>
                          {isComparing ? "✓" : ""}
                        </span>
                      )}
                    </div>

                    {/* Name */}
                    <h3 className="font-mono text-xs sm:text-sm tracking-[0.1em] uppercase text-foreground font-bold leading-tight">
                      {sp.commonName}
                    </h3>
                    <p className="font-mono text-[9px] tracking-wider text-muted-foreground italic mt-0.5">
                      {sp.scientificName}
                    </p>
                    <span className="font-mono text-[8px] tracking-[0.15em] uppercase text-muted-foreground/50 block mt-1">
                      {sp.family}
                    </span>
                    {detectionCounts[sp.commonName] && (
                      <span className="inline-flex items-center gap-1 mt-1 font-mono text-[8px] tracking-[0.1em] uppercase text-accent/70">
                        <Activity size={8} /> {detectionCounts[sp.commonName]} DETECTION{detectionCounts[sp.commonName] > 1 ? "S" : ""}
                      </span>
                    )}

                    {/* Status badge */}
                    <div className="mt-3 flex items-center justify-between">
                      <span className={`font-mono text-[9px] tracking-[0.15em] uppercase font-bold px-2 py-0.5 border ${sp.status === "common" ? "text-green-500 border-green-500/30 bg-green-500/10" :
                        sp.status === "uncommon" ? "text-amber-500 border-amber-500/30 bg-amber-500/10" :
                          "text-red-500 border-red-500/30 bg-red-500/10"
                        }`}>
                        {sp.status.toUpperCase()}
                      </span>
                      <span className="font-mono text-[8px] text-muted-foreground">
                        {sp.habitat.split(",")[0]}
                      </span>
                    </div>

                    {/* Footer */}
                    <div className="mt-3 border-t border-foreground/10 pt-2 font-mono text-[9px] tracking-[0.15em] uppercase text-accent text-center group-hover:text-foreground">
                      {compareMode ? (isComparing ? "SELECTED" : "SELECT TO COMPARE") : "VIEW DETAILS →"}
                    </div>
                  </motion.div>
                )
              })}
            </div>

            {/* ────── LOAD MORE ────── */}
            {hasMore && (
              <div className="flex flex-col items-center gap-2 pt-2">
                <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
                  SHOWING {visibleCount} OF {filteredAll.length} SPECIES
                </span>
                <button
                  type="button"
                  onClick={() => setVisibleCount((c) => Math.min(c + BATCH_SIZE, filteredAll.length))}
                  className="inline-flex items-center gap-2 px-6 py-2.5 border-2 border-accent bg-accent/10 font-mono text-xs tracking-[0.2em] uppercase text-accent hover:bg-accent hover:text-white cursor-pointer transition-none font-bold shadow-[4px_4px_0px_0px_rgba(234,88,12,0.2)] active:shadow-none active:translate-x-[4px] active:translate-y-[4px]"
                >
                  <ChevronDown size={14} />
                  LOAD MORE ({Math.min(BATCH_SIZE, filteredAll.length - visibleCount)} MORE)
                </button>
              </div>
            )}

            {!hasMore && filteredAll.length > BATCH_SIZE && (
              <div className="text-center pt-2">
                <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
                  ALL {filteredAll.length} SPECIES LOADED
                </span>
              </div>
            )}
          </>
        )}

        {/* ────── STATUS BAR ────── */}
        <div className="border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">
            SYS_STATUS:
          </span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            DATABASE LOADED • {ALL_SPECIES.length} SPECIES INDEXED
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>
      </main>

      {/* ════════════════════════════════════════════
          SPECIES DETAIL MODAL
          ════════════════════════════════════════════ */}
      {selectedSpecies && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/60" onClick={closeDetail} role="dialog" aria-label={`Details for ${selectedSpecies.commonName}`} aria-modal="true">
          <div
            className="border-2 border-foreground bg-background shadow-[8px_8px_0px_0px_hsl(var(--foreground))] w-full sm:max-w-lg sm:mx-4 h-[85vh] sm:h-auto sm:max-h-[90vh] overflow-y-auto relative"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Corner accents */}
            <span className="absolute -top-[3px] -left-[3px] w-2.5 h-2.5 bg-accent" />
            <span className="absolute -top-[3px] -right-[3px] w-2.5 h-2.5 bg-accent" />
            <span className="absolute -bottom-[3px] -left-[3px] w-2.5 h-2.5 bg-accent" />
            <span className="absolute -bottom-[3px] -right-[3px] w-2.5 h-2.5 bg-accent" />

            {/* Header */}
            <div className="border-b-2 border-foreground px-5 py-3 flex items-center justify-between">
              <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
                SPECIES DETAILS
              </span>
              <button
                type="button"
                onClick={closeDetail}
                className="w-8 h-8 border-2 border-foreground flex items-center justify-center text-muted-foreground hover:text-accent hover:border-accent cursor-pointer transition-none"
                aria-label="Close species details"
              >
                <X size={14} />
              </button>
            </div>

            <div className="p-5 space-y-5">
              {/* Top: Icon + Name */}
              <div className="flex items-start gap-4">
                <div className="w-16 h-16 border-2 border-accent/30 flex items-center justify-center shrink-0 bg-accent/5">
                  <Bird size={32} className="text-accent" />
                </div>
                <div>
                  <h2 className="font-mono text-lg sm:text-xl tracking-[0.1em] uppercase text-foreground font-bold leading-tight">
                    {selectedSpecies.commonName}
                  </h2>
                  <p className="font-mono text-xs tracking-wider text-muted-foreground italic">
                    {selectedSpecies.scientificName}
                  </p>
                </div>
              </div>

              {/* Info Table */}
              <div className="border-2 border-foreground divide-y divide-foreground/10">
                {[
                  ["FAMILY", selectedSpecies.family],
                  ["STATUS", selectedSpecies.status.toUpperCase()],
                  ["HABITAT", selectedSpecies.habitat],
                  ["CALL TYPE", selectedSpecies.callType],
                  ["DETECTIONS", detectionCounts[selectedSpecies.commonName] ? `${detectionCounts[selectedSpecies.commonName]} in history` : "None yet"],
                  ["FAVORITED", favorites.has(selectedSpecies.id) ? "★ Yes" : "No"],
                ].map(([label, value]) => (
                  <div key={label} className="flex">
                    <span className="w-28 sm:w-32 shrink-0 px-3 py-2 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground bg-muted/30 border-r border-foreground/10 flex items-center">
                      {label}
                    </span>
                    <span className="flex-1 px-3 py-2 font-mono text-[11px] tracking-wider text-foreground">
                      {value}
                    </span>
                  </div>
                ))}
              </div>

              {/* External Links */}
              <div>
                <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground font-bold mb-3 block">
                  EXTERNAL RESOURCES
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {([
                    ["eBIRD", EXTERNAL_LINKS.eBird(selectedSpecies.commonName)],
                    ["XENO-CANTO", EXTERNAL_LINKS.xenoCanto(selectedSpecies.commonName)],
                    ["WIKIPEDIA", EXTERNAL_LINKS.wikipedia(selectedSpecies.commonName)],
                    ["ALL ABOUT BIRDS", EXTERNAL_LINKS.allAboutBirds(selectedSpecies.commonName)],
                  ] as const).map(([label, url]) => (
                    <a
                      key={label}
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center justify-center gap-1.5 px-3 py-2 border-2 border-foreground font-mono text-[9px] tracking-[0.1em] uppercase text-foreground hover:border-accent hover:text-accent cursor-pointer transition-none"
                    >
                      <ExternalLink size={9} />
                      {label}
                    </a>
                  ))}
                </div>
              </div>

              {/* CTA Button */}
              <a
                href="/analyze"
                className="block w-full text-center border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.2em] uppercase font-bold py-3.5 shadow-[4px_4px_0px_0px_rgba(234,88,12,0.3)] hover:bg-background hover:text-accent active:shadow-none active:translate-x-[4px] active:translate-y-[4px] cursor-pointer transition-none"
              >
                <Zap size={14} className="inline mr-2 -mt-0.5" />
                ANALYZE THIS SPECIES
              </a>
            </div>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════
          COMPARE FLOATING BAR
          ════════════════════════════════════════════ */}
      {compareMode && compareIds.length > 0 && !selectedSpecies && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-40 border-2 border-accent bg-background shadow-[4px_4px_0px_0px_rgba(234,88,12,0.4)] px-5 py-3 flex items-center gap-4">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground">
            {compareIds.length} SPECIES SELECTED
          </span>
          <button
            type="button"
            disabled={compareIds.length < 2}
            onClick={() => { }}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => setCompareIds([])}
            className="px-3 py-1.5 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
          >
            CLEAR
          </button>
          {compareIds.length >= 2 && (
            <button
              type="button"
              onClick={() => { /* Compare modal is shown inline below */ }}
              className="px-4 py-1.5 border-2 border-accent bg-accent text-white font-mono text-[10px] tracking-[0.15em] uppercase font-bold cursor-pointer transition-none"
            >
              COMPARE NOW
            </button>
          )}
        </div>
      )}

      {/* ════════════════════════════════════════════
          COMPARE MODAL
          ════════════════════════════════════════════ */}
      {compareMode && compareIds.length >= 2 && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/60" onClick={() => { setCompareMode(false); setCompareIds([]) }}>
          <div
            className="border-2 border-foreground bg-background shadow-[8px_8px_0px_0px_hsl(var(--foreground))] w-full sm:max-w-3xl sm:mx-4 h-[85vh] sm:h-auto sm:max-h-[90vh] overflow-y-auto relative"
            onClick={(e) => e.stopPropagation()}
          >
            <span className="absolute -top-[3px] -left-[3px] w-2.5 h-2.5 bg-accent" />
            <span className="absolute -top-[3px] -right-[3px] w-2.5 h-2.5 bg-accent" />

            {/* Header */}
            <div className="border-b-2 border-foreground px-5 py-3 flex items-center justify-between">
              <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
                COMPARE SPECIES ({compareSpecies.length})
              </span>
              <button
                type="button"
                onClick={() => { setCompareMode(false); setCompareIds([]) }}
                className="w-8 h-8 border-2 border-foreground flex items-center justify-center text-muted-foreground hover:text-accent hover:border-accent cursor-pointer transition-none"
              >
                <X size={14} />
              </button>
            </div>

            <div className="p-5 overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b-2 border-foreground">
                    <th className="text-left px-3 py-2 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground">METRIC</th>
                    {compareSpecies.map((s) => (
                      <th key={s.id} className="text-center px-3 py-2 font-mono text-[9px] tracking-[0.2em] uppercase text-foreground font-bold">
                        {s.commonName}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-foreground/10">
                  {[
                    { label: "SCIENTIFIC NAME", fn: (s: Species) => <span className="italic text-muted-foreground">{s.scientificName}</span> },
                    { label: "FAMILY", fn: (s: Species) => s.family },
                    {
                      label: "STATUS", fn: (s: Species) => <span className={`font-bold ${s.status === "common" ? "text-green-500" :
                        s.status === "uncommon" ? "text-amber-500" : "text-red-500"
                        }`}>{s.status.toUpperCase()}</span>
                    },
                    { label: "HABITAT", fn: (s: Species) => s.habitat },
                    { label: "CALL TYPE", fn: (s: Species) => s.callType },
                  ].map((row) => (
                    <tr key={row.label}>
                      <td className="px-3 py-2 font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground whitespace-nowrap">{row.label}</td>
                      {compareSpecies.map((s) => (
                        <td key={s.id} className="px-3 py-2 font-mono text-[11px] text-center text-foreground">{row.fn(s)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
