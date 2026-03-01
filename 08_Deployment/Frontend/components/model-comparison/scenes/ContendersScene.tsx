"use client"

import { motion } from "framer-motion"
import { MODELS } from "../data/models"
import { ModelCard } from "../ui/ModelCard"

export function ContendersScene() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full h-full flex flex-col items-center justify-center gap-3 sm:gap-5 p-4 sm:p-8"
    >
      {/* Title */}
      <motion.p
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="font-mono text-[10px] sm:text-xs tracking-[0.25em] uppercase text-accent font-bold"
      >
        🤖 The Contenders
      </motion.p>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4, duration: 0.4 }}
        className="font-mono text-[8px] sm:text-[10px] tracking-[0.15em] uppercase text-muted-foreground"
      >
        6 deep learning architectures evaluated
      </motion.p>

      {/* 3×2 grid of model cards */}
      <div className="grid grid-cols-3 gap-2 sm:gap-3 w-full max-w-lg">
        {MODELS.map((m, i) => (
          <ModelCard
            key={m.id}
            model={m}
            state={m.verdict === "winner" ? "default" : "default"}
            delay={0.8 + i * 0.3}
          />
        ))}
      </div>
    </motion.div>
  )
}
