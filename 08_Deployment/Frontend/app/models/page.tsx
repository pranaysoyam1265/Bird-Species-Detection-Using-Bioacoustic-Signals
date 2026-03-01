"use client"

import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import { ModelComparisonAnimation } from "@/components/model-comparison/ModelComparisonAnimation"
import { MODELS } from "@/components/model-comparison/data/models"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"
import { useEffect } from "react"
import Link from "next/link"

export default function ModelsPage() {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!loading && !user) router.replace("/login")
  }, [user, loading, router])

  if (loading || !user) return (
    <div className="min-h-screen dot-grid-bg flex items-center justify-center scanline-overlay">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-accent border-t-transparent animate-spin" />
        <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
          AUTHENTICATING...
        </span>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />
      <Navbar />

      {/* Header — matches Analyze / Settings */}
      <div className="px-4 lg:px-6 pt-4 lg:pt-6 flex items-center gap-3">
        <NavSidebar />
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 text-accent"><path d="M12 2 2 7l10 5 10-5-10-5Z" /><path d="m2 17 10 5 10-5" /><path d="m2 12 10 5 10-5" /></svg>
            <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
              MODEL SELECTION
            </h1>
          </div>
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            DEEP LEARNING ARCHITECTURE COMPARISON
          </p>
        </div>
      </div>

      <main className="flex-1 p-4 lg:p-6 space-y-6">
        {/* Subtitle */}
        <p className="font-mono text-xs tracking-[0.1em] text-muted-foreground max-w-2xl">
          How we evaluated 6 deep learning architectures and selected EfficientNet-B2
          as the optimal model for bird species detection from bioacoustic signals.
        </p>

        {/* Full animation */}
        <ModelComparisonAnimation />

        {/* Detailed breakdown */}
        <div className="border-2 border-foreground">
          <div className="border-b-2 border-foreground px-4 py-3 bg-card/50">
            <p className="font-mono text-[10px] tracking-[0.2em] uppercase text-accent font-bold">
              Detailed Model Breakdown
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {MODELS.map((m) => (
              <div
                key={m.id}
                className={`p-4 border-b sm:border-r border-foreground/20 ${m.verdict === "winner" ? "bg-accent/5" : ""
                  }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="font-mono text-xs font-bold tracking-[0.15em] uppercase text-foreground">
                      {m.name}
                    </p>
                    <p className="font-mono text-[8px] tracking-wider text-muted-foreground uppercase">
                      {m.fullName}
                    </p>
                  </div>
                  {m.verdict === "winner" ? (
                    <span className="text-xs bg-accent text-white px-1.5 py-0.5 font-mono font-bold tracking-wider uppercase">
                      Winner
                    </span>
                  ) : (
                    <span className="text-[8px] bg-red-500/20 text-red-400 px-1.5 py-0.5 font-mono tracking-wider uppercase">
                      Out
                    </span>
                  )}
                </div>
                <div className="space-y-1 mb-2">
                  <div className="flex justify-between">
                    <span className="font-mono text-[8px] text-foreground/40 uppercase">Accuracy</span>
                    <span className="font-mono text-[9px] text-foreground font-bold">{m.accuracy}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono text-[8px] text-foreground/40 uppercase">Top-5</span>
                    <span className="font-mono text-[9px] text-foreground font-bold">{m.topFiveAccuracy}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono text-[8px] text-foreground/40 uppercase">Params</span>
                    <span className="font-mono text-[9px] text-foreground">{m.parameters}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono text-[8px] text-foreground/40 uppercase">Inference</span>
                    <span className="font-mono text-[9px] text-foreground">{m.inferenceTime}ms</span>
                  </div>
                </div>
                {m.eliminationReason && (
                  <p className="font-mono text-[7px] tracking-wider text-red-400/70 uppercase border-t border-foreground/10 pt-1">
                    ✕ {m.eliminationReason}
                  </p>
                )}
                {m.verdict === "winner" && (
                  <div className="border-t border-foreground/10 pt-1 space-y-0.5">
                    {m.pros.map((p) => (
                      <p key={p} className="font-mono text-[7px] tracking-wider text-accent/80 uppercase">
                        ✓ {p}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="flex items-center justify-center gap-4 py-4">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-6 py-3 border-2 border-accent bg-accent/10 font-mono text-xs tracking-[0.2em] uppercase text-accent hover:bg-accent hover:text-white transition-colors font-bold"
          >
            Try BirdSense →
          </Link>
        </div>

        {/* Footer status */}
        <div className="flex items-center gap-4 border-t border-border pt-4">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">
            SYS_STATUS:
          </span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            Model evaluation complete • EfficientNet-B2 deployed
          </span>
        </div>
      </main>
    </div>
  )
}
