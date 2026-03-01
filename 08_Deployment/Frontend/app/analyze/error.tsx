"use client"

import { useEffect } from "react"

export default function AnalyzeError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error("[Analyze Error Boundary]", error)
  }, [error])

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-8">
      <div className="border-2 border-destructive/50 bg-background p-8 max-w-lg w-full text-center space-y-6">
        {/* glitch header */}
        <div className="space-y-2">
          <h2 className="font-mono text-xl font-bold tracking-wider text-destructive uppercase">
            ⚠ ANALYSIS ERROR
          </h2>
          <div className="h-px bg-destructive/30 w-full" />
        </div>

        <p className="font-mono text-sm text-muted-foreground leading-relaxed">
          Something went wrong during bird audio analysis. This could be due to
          a network issue, an unsupported audio format, or a temporary server problem.
        </p>

        {error.message && (
          <pre className="font-mono text-[10px] text-destructive/70 bg-destructive/5 border border-destructive/20 p-3 text-left overflow-x-auto whitespace-pre-wrap">
            {error.message}
          </pre>
        )}

        <div className="flex gap-3 justify-center pt-2">
          <button
            onClick={reset}
            className="font-mono text-xs font-bold tracking-wider px-6 py-2.5 border-2 border-foreground bg-foreground text-background hover:bg-transparent hover:text-foreground cursor-pointer transition-colors"
          >
            ↻ RETRY
          </button>
          <a
            href="/"
            className="font-mono text-xs font-bold tracking-wider px-6 py-2.5 border-2 border-foreground/30 text-foreground/70 hover:border-foreground hover:text-foreground transition-colors inline-flex items-center"
          >
            ← HOME
          </a>
        </div>
      </div>
    </div>
  )
}
