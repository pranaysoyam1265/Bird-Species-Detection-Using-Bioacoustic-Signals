"use client"

import { useEffect, useState } from "react"
import { Keyboard, X } from "lucide-react"

const SHORTCUTS = [
  { keys: "Ctrl+K", description: "Open command palette" },
  { keys: "Esc", description: "Close modal / palette" },
  { keys: "↑ ↓", description: "Navigate lists" },
  { keys: "Enter", description: "Select item" },
  { keys: "Del", description: "Delete selected (results)" },
]

const LS_KEY = "birdsense-shortcuts-seen"

export function KeyboardShortcutsToast() {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    try {
      if (localStorage.getItem(LS_KEY)) return
      const timer = setTimeout(() => setVisible(true), 2000)
      return () => clearTimeout(timer)
    } catch { /* SSR guard */ }
  }, [])

  const dismiss = () => {
    setVisible(false)
    try { localStorage.setItem(LS_KEY, "1") } catch { /* */ }
  }

  if (!visible) return null

  return (
    <div className="fixed bottom-6 right-6 z-[100] border-2 border-foreground bg-background shadow-[6px_6px_0px_0px_hsl(var(--foreground))] max-w-xs w-full animate-in slide-in-from-bottom-4 fade-in duration-300">
      {/* Corner accents */}
      <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent" />
      <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent" />

      {/* Header */}
      <div className="border-b border-foreground/10 px-4 py-2.5 flex items-center justify-between">
        <span className="flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] uppercase text-accent font-bold">
          <Keyboard size={12} /> KEYBOARD SHORTCUTS
        </span>
        <button
          type="button"
          onClick={dismiss}
          className="text-muted-foreground hover:text-foreground cursor-pointer"
          aria-label="Dismiss shortcuts toast"
        >
          <X size={12} />
        </button>
      </div>

      {/* Shortcuts list */}
      <div className="p-3 space-y-1.5">
        {SHORTCUTS.map((s) => (
          <div key={s.keys} className="flex items-center justify-between">
            <span className="font-mono text-[10px] tracking-wider text-muted-foreground">
              {s.description}
            </span>
            <kbd className="px-2 py-0.5 border border-foreground/20 bg-muted/30 font-mono text-[9px] tracking-wider text-foreground">
              {s.keys}
            </kbd>
          </div>
        ))}
      </div>

      {/* Dismiss button */}
      <div className="border-t border-foreground/10 px-4 py-2">
        <button
          type="button"
          onClick={dismiss}
          className="w-full text-center font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground cursor-pointer"
        >
          GOT IT — DON&apos;T SHOW AGAIN
        </button>
      </div>
    </div>
  )
}
