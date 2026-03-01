"use client"

import { motion } from "framer-motion"
import type { ModelData } from "../data/models"

export function ModelCard({
  model,
  state = "default",
  delay = 0,
}: {
  model: ModelData
  state?: "default" | "winner" | "eliminated"
  delay?: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{
        opacity: state === "eliminated" ? 0.3 : 1,
        y: 0,
        scale: state === "winner" ? 1.05 : 1,
      }}
      transition={{ duration: 0.5, delay, type: "spring", stiffness: 200, damping: 20 }}
      className={`relative p-3 sm:p-4 border-2 font-mono transition-none ${state === "winner"
          ? "border-accent bg-accent/5 shadow-[0_0_30px_hsl(var(--accent)/0.3)]"
          : state === "eliminated"
            ? "border-foreground/20 bg-muted/20"
            : "border-foreground bg-card"
        }`}
    >
      {/* Status indicator */}
      {state === "winner" && (
        <motion.span
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: delay + 0.3, type: "spring" }}
          className="absolute -top-2 -right-2 w-6 h-6 bg-accent text-white flex items-center justify-center text-[10px] font-bold z-10"
        >
          ✓
        </motion.span>
      )}
      {state === "eliminated" && (
        <motion.span
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: delay + 0.3, type: "spring" }}
          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white flex items-center justify-center text-[10px] font-bold z-10"
        >
          ✕
        </motion.span>
      )}

      {/* Icon placeholder */}
      <div className={`w-8 h-8 sm:w-10 sm:h-10 border flex items-center justify-center mb-2 ${state === "winner" ? "border-accent bg-accent/10" : "border-foreground/20 bg-muted/30"
        }`}>
        <span className="text-[10px] sm:text-xs font-bold text-accent">
          {model.name.slice(0, 2).toUpperCase()}
        </span>
      </div>

      <p className="text-[10px] sm:text-xs font-bold tracking-[0.15em] uppercase text-foreground leading-tight">
        {model.name}
      </p>
      <p className="text-[7px] sm:text-[8px] tracking-wider text-muted-foreground mt-0.5 uppercase">
        {model.parameters} • {model.inferenceTime}ms
      </p>
      {state === "winner" && (
        <p className="text-[8px] tracking-wider text-accent font-bold mt-1 uppercase">
          {model.accuracy}% ACC
        </p>
      )}
    </motion.div>
  )
}
