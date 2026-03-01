"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { MODELS } from "../data/models"
import { ModelCard } from "../ui/ModelCard"

export function VerdictScene() {
  const [showScore, setShowScore] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setShowScore(true), 2500)
    return () => clearTimeout(t)
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full h-full flex flex-col items-center justify-center gap-3 sm:gap-5 p-4 sm:p-8"
    >
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="font-mono text-[10px] sm:text-xs tracking-[0.25em] uppercase text-accent font-bold"
      >
        ⚖️ Calculating Optimal Model
      </motion.p>

      {/* Model grid with elimination */}
      <div className="grid grid-cols-3 gap-2 sm:gap-3 w-full max-w-lg">
        {MODELS.map((m, i) => (
          <ModelCard
            key={m.id}
            model={m}
            state={m.verdict === "winner" ? "winner" : "eliminated"}
            delay={0.3 + i * 0.15}
          />
        ))}
      </div>

      {/* Score calculation box */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={showScore ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.5 }}
        className="border-2 border-accent/50 bg-card/80 p-3 sm:p-4 max-w-sm w-full"
      >
        <p className="font-mono text-[7px] sm:text-[8px] tracking-[0.2em] uppercase text-muted-foreground mb-1">
          Composite Score = Accuracy × Efficiency
        </p>
        <p className="font-mono text-xs sm:text-sm tracking-wider text-accent font-bold">
          EfficientNet-B2: 96.06 × 0.94 = 90.3
        </p>
        <p className="font-mono text-[7px] sm:text-[8px] tracking-wider text-foreground/60 mt-1 uppercase">
          Highest balanced score across all metrics
        </p>
      </motion.div>
    </motion.div>
  )
}
