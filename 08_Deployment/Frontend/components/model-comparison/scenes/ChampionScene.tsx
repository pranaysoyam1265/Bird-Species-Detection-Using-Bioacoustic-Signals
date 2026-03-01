"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { MODELS } from "../data/models"
import { CountingNumber } from "../ui/CountingNumber"

const winner = MODELS.find(m => m.verdict === "winner")!

export function ChampionScene() {
  const [showStats, setShowStats] = useState(false)
  const [showCta, setShowCta] = useState(false)

  useEffect(() => {
    const t1 = setTimeout(() => setShowStats(true), 1500)
    const t2 = setTimeout(() => setShowCta(true), 3500)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full h-full flex flex-col items-center justify-center gap-4 sm:gap-6 p-4 sm:p-8"
    >
      {/* Title */}
      <motion.p
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ type: "spring", stiffness: 200, damping: 15 }}
        className="font-mono text-[10px] sm:text-xs tracking-[0.25em] uppercase text-accent font-bold"
      >
        🏆 The Champion
      </motion.p>

      {/* Champion card */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.5, type: "spring", stiffness: 180, damping: 18 }}
        className="border-2 border-accent bg-card p-4 sm:p-6 w-full max-w-sm shadow-[0_0_40px_hsl(var(--accent)/0.25)]"
      >
        <p className="font-mono text-sm sm:text-base font-bold tracking-[0.2em] uppercase text-accent mb-0.5">
          {winner.name}
        </p>
        <p className="font-mono text-[8px] sm:text-[9px] tracking-wider text-muted-foreground mb-3 uppercase">
          {winner.fullName}
        </p>

        {/* Stats grid */}
        <div className="space-y-2">
          {/* Accuracy bar */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="font-mono text-[8px] sm:text-[9px] tracking-[0.15em] uppercase text-foreground/60">
                Test Accuracy
              </span>
              <span className="font-mono text-[9px] sm:text-xs font-bold text-accent">
                <CountingNumber value={winner.accuracy} trigger={showStats} suffix="%" />
              </span>
            </div>
            <div className="h-2 sm:h-3 bg-foreground/10 border border-foreground/20 overflow-hidden">
              <motion.div
                className="h-full bg-accent"
                initial={{ width: 0 }}
                animate={showStats ? { width: `${winner.accuracy}%` } : {}}
                transition={{ duration: 1.5, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Top-5 bar */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="font-mono text-[8px] sm:text-[9px] tracking-[0.15em] uppercase text-foreground/60">
                Top-5 Accuracy
              </span>
              <span className="font-mono text-[9px] sm:text-xs font-bold text-accent">
                <CountingNumber value={winner.topFiveAccuracy} trigger={showStats} suffix="%" />
              </span>
            </div>
            <div className="h-2 sm:h-3 bg-foreground/10 border border-foreground/20 overflow-hidden">
              <motion.div
                className="h-full bg-accent"
                initial={{ width: 0 }}
                animate={showStats ? { width: `${winner.topFiveAccuracy}%` } : {}}
                transition={{ duration: 1.5, delay: 0.2, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Inline stats */}
          <div className="flex gap-4 pt-1">
            <div>
              <p className="font-mono text-[7px] sm:text-[8px] tracking-wider text-foreground/40 uppercase">Params</p>
              <p className="font-mono text-[10px] sm:text-xs font-bold text-foreground">{winner.parameters}</p>
            </div>
            <div>
              <p className="font-mono text-[7px] sm:text-[8px] tracking-wider text-foreground/40 uppercase">Inference</p>
              <p className="font-mono text-[10px] sm:text-xs font-bold text-foreground">{winner.inferenceTime}ms</p>
            </div>
            <div>
              <p className="font-mono text-[7px] sm:text-[8px] tracking-wider text-foreground/40 uppercase">Species</p>
              <p className="font-mono text-[10px] sm:text-xs font-bold text-foreground">87</p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={showCta ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.4 }}
      >
        <Link
          href="/analyze"
          className="inline-flex items-center gap-2 px-5 py-2.5 border-2 border-accent bg-accent/10 font-mono text-[10px] sm:text-xs tracking-[0.2em] uppercase text-accent hover:bg-accent hover:text-white transition-colors font-bold"
        >
          🎤 Try It Now →
        </Link>
      </motion.div>
    </motion.div>
  )
}
