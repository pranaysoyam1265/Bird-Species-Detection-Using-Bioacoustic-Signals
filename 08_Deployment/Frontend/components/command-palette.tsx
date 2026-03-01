"use client"

import { useState, useEffect, useRef, useMemo, useCallback } from "react"
import { useRouter } from "next/navigation"
import {
  Search,
  Home,
  Mic,
  History,
  Bird,
  User,
  Settings,
  X,
} from "lucide-react"

interface PaletteItem {
  id: string
  label: string
  description: string
  icon: React.ReactNode
  action: () => void
  category: "page" | "action"
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  const items: PaletteItem[] = useMemo(() => [
    { id: "home", label: "Home", description: "Go to landing page", icon: <Home size={14} />, action: () => router.push("/"), category: "page" },
    { id: "analyze", label: "Analyze", description: "Detect bird species from audio", icon: <Mic size={14} />, action: () => router.push("/analyze"), category: "page" },
    { id: "results", label: "Results", description: "View detection history", icon: <History size={14} />, action: () => router.push("/results"), category: "page" },
    { id: "species", label: "Species Database", description: "Browse 87 species", icon: <Bird size={14} />, action: () => router.push("/species"), category: "page" },
    { id: "profile", label: "Profile", description: "Account & activity", icon: <User size={14} />, action: () => router.push("/profile"), category: "page" },
    { id: "settings", label: "Settings", description: "Preferences & data", icon: <Settings size={14} />, action: () => router.push("/settings"), category: "page" },
  ], [router])

  const filtered = useMemo(() => {
    if (!query) return items
    const q = query.toLowerCase()
    return items.filter(
      (i) => i.label.toLowerCase().includes(q) || i.description.toLowerCase().includes(q)
    )
  }, [items, query])

  /* ── Keyboard: Ctrl+K to open, Escape to close ── */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
      if (e.key === "Escape" && open) {
        setOpen(false)
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open])

  /* ── Focus input when opened ── */
  useEffect(() => {
    if (open) {
      setQuery("")
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  /* ── Arrow key navigation ── */
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault()
        setSelectedIndex((i) => Math.min(i + 1, filtered.length - 1))
      } else if (e.key === "ArrowUp") {
        e.preventDefault()
        setSelectedIndex((i) => Math.max(i - 1, 0))
      } else if (e.key === "Enter" && filtered[selectedIndex]) {
        filtered[selectedIndex].action()
        setOpen(false)
      }
    },
    [filtered, selectedIndex]
  )

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[9999] flex items-start justify-center pt-[15vh] bg-black/50" onClick={() => setOpen(false)} role="dialog" aria-label="Command palette" aria-modal="true">
      <div
        className="w-full max-w-lg mx-4 border-2 border-foreground bg-background shadow-[8px_8px_0px_0px_hsl(var(--foreground))]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Corner accents */}
        <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent" />
        <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent" />

        {/* Search input */}
        <div className="flex items-center border-b-2 border-foreground px-4">
          <Search size={14} className="text-muted-foreground shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="TYPE A COMMAND OR SEARCH..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent font-mono text-xs tracking-wider uppercase text-foreground placeholder:text-muted-foreground/40 px-3 py-3.5 outline-none"
            aria-label="Search commands"
          />
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="text-muted-foreground hover:text-foreground cursor-pointer"
          >
            <X size={12} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[300px] overflow-y-auto" role="listbox" aria-label="Search results">
          {filtered.length === 0 ? (
            <div className="px-4 py-6 text-center">
              <p className="font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground">
                NO RESULTS FOUND
              </p>
            </div>
          ) : (
            <div className="py-1">
              {filtered.map((item, idx) => (
                <button
                  key={item.id}
                  type="button"
                  role="option"
                  aria-selected={idx === selectedIndex}
                  onClick={() => { item.action(); setOpen(false) }}
                  className={`w-full flex items-center gap-3 px-4 py-3 cursor-pointer transition-none text-left ${idx === selectedIndex ? "bg-accent/10 border-l-4 border-l-accent" : "border-l-4 border-l-transparent hover:bg-muted/50"}`}
                >
                  <div className={`w-8 h-8 border flex items-center justify-center shrink-0 ${idx === selectedIndex ? "border-accent text-accent" : "border-foreground/20 text-muted-foreground"}`}>
                    {item.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className={`font-mono text-xs tracking-[0.15em] uppercase block font-bold ${idx === selectedIndex ? "text-accent" : "text-foreground"}`}>
                      {item.label}
                    </span>
                    <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block">
                      {item.description}
                    </span>
                  </div>
                  <span className="font-mono text-[8px] tracking-[0.15em] uppercase text-muted-foreground/40 shrink-0">
                    {item.category}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-foreground/10 px-4 py-2 flex items-center justify-between">
          <span className="font-mono text-[8px] tracking-[0.15em] uppercase text-muted-foreground/50">
            ↑↓ NAVIGATE • ENTER SELECT • ESC CLOSE
          </span>
          <kbd className="px-1.5 py-0.5 border border-foreground/20 font-mono text-[8px] text-muted-foreground/50">
            CTRL+K
          </kbd>
        </div>
      </div>
    </div>
  )
}
