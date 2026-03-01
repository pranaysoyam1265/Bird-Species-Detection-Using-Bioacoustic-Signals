"use client"

import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Bird, Home, Search, Mic } from "lucide-react"

export default function NotFound() {
  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      {/* Radial vignette */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />

      <Navbar />

      <main className="flex-1 flex flex-col items-center justify-center px-4 relative z-10 -mt-16">
        {/* Glitch title */}
        <div className="relative mb-6 select-none">
          <span className="font-pixel text-[80px] sm:text-[120px] lg:text-[160px] tracking-tight text-foreground/5 leading-none">
            404
          </span>
          <span className="absolute inset-0 font-pixel text-[80px] sm:text-[120px] lg:text-[160px] tracking-tight text-accent leading-none animate-glitch">
            404
          </span>
        </div>

        {/* Bird icon */}
        <div className="w-20 h-20 border-2 border-foreground flex items-center justify-center mb-6 relative">
          <Bird size={40} className="text-muted-foreground/40" />
          <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent" />
          <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent" />
          <span className="absolute -bottom-[3px] -left-[3px] w-2 h-2 bg-accent" />
          <span className="absolute -bottom-[3px] -right-[3px] w-2 h-2 bg-accent" />
        </div>

        {/* Message */}
        <div className="border-2 border-foreground bg-background p-6 max-w-md w-full text-center space-y-3 shadow-[6px_6px_0px_0px_hsl(var(--foreground))]">
          <h1 className="font-mono text-lg tracking-[0.2em] uppercase text-foreground font-bold">
            SIGNAL NOT FOUND
          </h1>
          <p className="font-mono text-xs tracking-[0.1em] text-muted-foreground">
            The acoustic signature you&apos;re looking for doesn&apos;t exist in our database.
          </p>

          <div className="border border-foreground/20 bg-muted/30 px-4 py-2 font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            <span className="text-accent">ERR_404:</span> ROUTE_NOT_FOUND
          </div>
        </div>

        {/* Navigation links */}
        <div className="mt-6 flex flex-wrap gap-3 justify-center">
          <Link
            href="/"
            className="flex items-center gap-2 px-5 py-2.5 border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.15em] uppercase font-bold shadow-[3px_3px_0px_0px_rgba(234,88,12,0.3)] hover:bg-background hover:text-accent cursor-pointer transition-none"
          >
            <Home size={14} /> HOME
          </Link>
          <Link
            href="/analyze"
            className="flex items-center gap-2 px-5 py-2.5 border-2 border-foreground bg-background text-foreground font-mono text-xs tracking-[0.15em] uppercase font-bold hover:bg-muted cursor-pointer transition-none"
          >
            <Mic size={14} /> ANALYZE
          </Link>
          <Link
            href="/species"
            className="flex items-center gap-2 px-5 py-2.5 border-2 border-foreground bg-background text-foreground font-mono text-xs tracking-[0.15em] uppercase font-bold hover:bg-muted cursor-pointer transition-none"
          >
            <Search size={14} /> SPECIES
          </Link>
        </div>

        {/* Status bar */}
        <div className="mt-8 border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">
            SYS_STATUS:
          </span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            PAGE NOT FOUND • AWAITING REDIRECT
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>
      </main>
    </div>
  )
}
