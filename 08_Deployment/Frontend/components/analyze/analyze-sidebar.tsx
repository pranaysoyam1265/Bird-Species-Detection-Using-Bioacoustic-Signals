"use client"

import { useState, useEffect } from "react"
import { Settings, Search, Save, FileDown, RotateCcw, Copy, Check, Clock, Bird, Radio } from "lucide-react"
import { SPECIES_META, getSpeciesNames, getRecentSearches, addRecentSearch } from "@/lib/species-meta"

// Full species list supported by BirdSense detection model
const BIRD_SPECIES = getSpeciesNames()

export interface SidebarSettings {
  noiseReduction: boolean
  chunkDuration: number
  topK: number
  minConfidence: number
  searchSpecies: string
  sensitivity: number
}

interface AnalyzeSidebarProps {
  settings: SidebarSettings
  onSettingsChange: (s: SidebarSettings) => void
  hasResults: boolean
  onAnalyzeNew: () => void
  onExport: () => void
  onSave: () => void
  onCopyResults?: () => void
}

export function AnalyzeSidebar({ settings, onSettingsChange, hasResults, onAnalyzeNew, onExport, onSave, onCopyResults }: AnalyzeSidebarProps) {
  const [copied, setCopied] = useState(false)
  const [recentSearches, setRecentSearches] = useState<string[]>([])
  const [speciesFilter, setSpeciesFilter] = useState("")

  // Load recent searches on mount
  useEffect(() => {
    setRecentSearches(getRecentSearches())
  }, [])

  const update = (partial: Partial<SidebarSettings>) => {
    const next = { ...settings, ...partial }
    // Track recent searches
    if (partial.searchSpecies && partial.searchSpecies !== settings.searchSpecies) {
      addRecentSearch(partial.searchSpecies)
      setRecentSearches(getRecentSearches())
    }
    onSettingsChange(next)
  }

  const handleCopy = () => {
    onCopyResults?.()
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Selected species metadata
  const selectedMeta = settings.searchSpecies ? SPECIES_META[settings.searchSpecies] : null

  // Filtered species list for the picker
  const filteredSpecies = BIRD_SPECIES.filter((s) =>
    s.toLowerCase().includes(speciesFilter.toLowerCase())
  )

  return (
    <aside className="w-full lg:w-[280px] lg:min-w-[280px] flex flex-col gap-4 border-l-4 border-l-accent pl-0 relative">

      {/* ── ANALYSIS SETTINGS ── */}
      <div className="border-2 border-foreground bg-background">
        <div className="border-b-2 border-foreground px-4 py-2 flex items-center gap-2">
          <Settings size={12} />
          <span className="font-mono text-xs tracking-[0.25em] uppercase text-accent font-bold">
            ANALYSIS SETTINGS
          </span>
        </div>
        <div className="p-4 space-y-5">
          {/* Noise Reduction */}
          <label className="flex items-center gap-3 cursor-pointer">
            <span
              onClick={() => update({ noiseReduction: !settings.noiseReduction })}
              className={`w-4 h-4 border-2 border-foreground cursor-pointer flex items-center justify-center ${settings.noiseReduction ? "bg-foreground" : ""
                }`}
            >
              {settings.noiseReduction && <span className="text-background text-[10px] font-bold">✓</span>}
            </span>
            <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground">
              NOISE REDUCTION
            </span>
          </label>

          {/* Chunk Duration */}
          <div className="space-y-2">
            <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold block">
              CHUNK DURATION
            </span>
            <div className="flex gap-1">
              {[3, 4, 5, 6, 7].map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => update({ chunkDuration: v })}
                  className={`flex-1 py-1.5 font-mono text-[10px] tracking-wider uppercase border-2 border-foreground cursor-pointer transition-none ${settings.chunkDuration === v
                    ? "bg-accent text-white border-accent"
                    : "bg-background text-foreground hover:bg-muted"
                    }`}
                >
                  {v}s
                </button>
              ))}
            </div>
          </div>

          {/* Top Predictions */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
                TOP PREDICTIONS
              </span>
              <span className="font-mono text-[10px] tracking-wider text-accent font-bold">
                {settings.topK}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={10}
              value={settings.topK}
              onChange={(e) => update({ topK: parseInt(e.target.value) })}
              className="w-full h-1 appearance-none bg-foreground/20 outline-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:border-0 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>

          {/* Min Confidence */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
                MIN CONFIDENCE
              </span>
              <span className="font-mono text-[10px] tracking-wider text-accent font-bold">
                {settings.minConfidence}%
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={50}
              value={settings.minConfidence}
              onChange={(e) => update({ minConfidence: parseInt(e.target.value) })}
              className="w-full h-1 appearance-none bg-foreground/20 outline-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:border-0 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>

          {/* Sensitivity — only relevant when a focus species is set */}
          {settings.searchSpecies && (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
                  FOCUS SENSITIVITY
                </span>
                <span className="font-mono text-[10px] tracking-wider text-accent font-bold">
                  {settings.sensitivity}%
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={20}
                value={settings.sensitivity}
                onChange={(e) => update({ sensitivity: parseInt(e.target.value) })}
                className="w-full h-1 appearance-none bg-foreground/20 outline-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:border-0 [&::-webkit-slider-thumb]:cursor-pointer"
              />
            </div>
          )}
        </div>
      </div>

      {/* ── SPECIES FOCUS (OPTIONAL) ── */}
      <div className="border-2 border-foreground bg-background">
        <div className="border-b-2 border-foreground px-4 py-2 flex items-center gap-2">
          <Search size={12} />
          <span className="font-mono text-xs tracking-[0.25em] uppercase text-accent font-bold">
            SPECIES FOCUS
          </span>
          <span className="font-mono text-[8px] tracking-wider text-muted-foreground/50 ml-auto">OPTIONAL</span>
        </div>
        <div className="p-4 space-y-3">
          {/* ── Recent Searches ── */}
          {recentSearches.length > 0 && !settings.searchSpecies && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <Clock size={10} className="text-muted-foreground/60" />
                <span className="font-mono text-[8px] tracking-[0.2em] uppercase text-muted-foreground/60">
                  RECENT
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {recentSearches.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => update({ searchSpecies: s })}
                    className="px-2 py-1 border border-foreground/20 font-mono text-[9px] tracking-wider text-foreground hover:border-accent hover:text-accent cursor-pointer transition-none bg-transparent"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* ── Currently selected ── */}
          {settings.searchSpecies && (
            <div className="flex items-center justify-between px-2.5 py-1.5 border border-accent bg-accent/10">
              <span className="font-mono text-[10px] tracking-wider text-accent font-bold uppercase">
                {settings.searchSpecies}
              </span>
              <button
                type="button"
                onClick={() => { update({ searchSpecies: "" }); setSpeciesFilter("") }}
                className="font-mono text-[8px] tracking-wider text-muted-foreground hover:text-red-500 cursor-pointer bg-transparent border-none p-0"
              >
                CLEAR
              </button>
            </div>
          )}

          {/* ── Search input ── */}
          <input
            type="text"
            placeholder="Search bird to focus on..."
            value={speciesFilter}
            onChange={(e) => setSpeciesFilter(e.target.value)}
            className="w-full border-2 border-foreground bg-background px-3 py-2 font-mono text-xs tracking-wider uppercase placeholder:text-muted-foreground/40 outline-none focus:border-accent"
          />

          {/* ── Species list ── */}
          <div className="border border-foreground/30 overflow-y-auto max-h-[200px]">
            {filteredSpecies.map((s) => {
              const isSelected = s === settings.searchSpecies
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => { update({ searchSpecies: s }); setSpeciesFilter("") }}
                  className={`w-full text-left px-3 py-1.5 font-mono text-[10px] tracking-wider cursor-pointer border-none transition-none block ${isSelected
                    ? "bg-accent text-white font-bold"
                    : "bg-background text-foreground hover:bg-muted hover:text-accent"
                    }`}
                >
                  {s}
                </button>
              )
            })}
          </div>
        </div>

        {/* ── Species Info Card ── */}
        {selectedMeta && (
          <div className="border-t-2 border-foreground p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Bird size={12} className="text-accent" />
              <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
                SPECIES INFO
              </span>
            </div>

            {/* Name */}
            <div>
              <h3 className="font-mono text-sm tracking-wider uppercase text-foreground font-bold leading-tight">
                {selectedMeta.name}
              </h3>
              <p className="font-mono text-[9px] tracking-wider text-muted-foreground italic">
                {selectedMeta.scientificName}
              </p>
            </div>

            {/* Info table */}
            <div className="space-y-1.5">
              {[
                ["FAMILY", selectedMeta.family],
                ["HABITAT", selectedMeta.habitat],
                ["CALL", selectedMeta.callType],
                ["STATUS", selectedMeta.status.toUpperCase()],
              ].map(([label, value]) => (
                <div key={label} className="flex gap-2">
                  <span className="font-mono text-[8px] tracking-[0.15em] uppercase text-muted-foreground/60 w-14 shrink-0">
                    {label}
                  </span>
                  <span className="font-mono text-[9px] tracking-wider text-foreground leading-tight">
                    {value}
                  </span>
                </div>
              ))}
            </div>

            {/* Frequency range bar */}
            <div className="space-y-1">
              <div className="flex items-center gap-1.5">
                <Radio size={9} className="text-accent/60" />
                <span className="font-mono text-[8px] tracking-[0.15em] uppercase text-muted-foreground/60">
                  FREQ RANGE
                </span>
              </div>
              <div className="relative h-3 bg-foreground/5 border border-foreground/10 overflow-hidden">
                {/* Scale: 0-12 kHz */}
                <div
                  className="absolute top-0 bottom-0 bg-accent/30"
                  style={{
                    left: `${(selectedMeta.freqLow / 12) * 100}%`,
                    width: `${((selectedMeta.freqHigh - selectedMeta.freqLow) / 12) * 100}%`,
                  }}
                />
              </div>
              <div className="flex justify-between">
                <span className="font-mono text-[7px] text-muted-foreground/40">0 kHz</span>
                <span className="font-mono text-[8px] text-accent font-bold">
                  {selectedMeta.freqLow}–{selectedMeta.freqHigh} kHz
                </span>
                <span className="font-mono text-[7px] text-muted-foreground/40">12 kHz</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ACTIONS */}
      <div className="border-2 border-foreground bg-background">
        <div className="border-b-2 border-foreground px-4 py-2">
          <span className="font-mono text-xs tracking-[0.25em] uppercase text-accent font-bold">
            ACTIONS
          </span>
        </div>
        <div className="p-4 space-y-2">
          <button
            type="button"
            onClick={onSave}
            disabled={!hasResults}
            className="w-full flex items-center gap-3 px-4 py-2.5 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Save size={14} /> SAVE TO HISTORY
          </button>
          <button
            type="button"
            onClick={onExport}
            disabled={!hasResults}
            className="w-full flex items-center gap-3 px-4 py-2.5 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none hover:bg-muted disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <FileDown size={14} /> EXPORT RESULTS
          </button>
          <button
            type="button"
            onClick={handleCopy}
            disabled={!hasResults}
            className={`w-full flex items-center gap-3 px-4 py-2.5 border-2 font-mono text-xs tracking-[0.15em] uppercase cursor-pointer transition-none disabled:opacity-30 disabled:cursor-not-allowed ${copied
              ? "border-green-500 bg-green-500/10 text-green-500"
              : "border-foreground hover:bg-muted"
              }`}
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? "COPIED!" : "COPY RESULTS"}
          </button>
          <button
            type="button"
            onClick={onAnalyzeNew}
            className="w-full flex items-center gap-3 px-4 py-2.5 border-2 border-accent text-accent font-mono text-xs tracking-[0.15em] uppercase font-bold cursor-pointer transition-none hover:bg-accent hover:text-white"
          >
            <RotateCcw size={14} /> ANALYZE NEW
          </button>
        </div>
      </div>
    </aside>
  )
}
