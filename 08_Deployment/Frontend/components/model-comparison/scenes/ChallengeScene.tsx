"use client"

import { motion } from "framer-motion"

export function ChallengeScene() {
  /* Animated waveform bars */
  const bars = Array.from({ length: 40 }, () => 0.2 + Math.random() * 0.8)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full h-full flex flex-col items-center justify-center gap-4 sm:gap-6 p-4 sm:p-8"
    >
      {/* Title */}
      <motion.p
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="font-mono text-[10px] sm:text-xs tracking-[0.25em] uppercase text-accent font-bold"
      >
        🎯 The Challenge
      </motion.p>

      {/* Waveform */}
      <motion.div
        initial={{ opacity: 0, scaleY: 0 }}
        animate={{ opacity: 1, scaleY: 1 }}
        transition={{ delay: 0.3, duration: 0.6, ease: "easeOut" }}
        className="border-2 border-foreground/30 bg-card/50 px-3 sm:px-6 py-3 sm:py-4 w-full max-w-md"
      >
        <div className="flex items-end justify-center gap-[2px] h-10 sm:h-14">
          {bars.map((h, i) => (
            <motion.div
              key={i}
              className="w-[3px] sm:w-1 bg-accent"
              initial={{ height: 0 }}
              animate={{ height: `${h * 100}%` }}
              transition={{
                delay: 0.5 + i * 0.02,
                duration: 0.3,
                repeat: Infinity,
                repeatType: "reverse",
                repeatDelay: 0.5 + Math.random() * 1,
              }}
            />
          ))}
        </div>
        <p className="font-mono text-[7px] sm:text-[8px] tracking-[0.2em] uppercase text-muted-foreground mt-2 text-center">
          audio_signal.wav — 87 species
        </p>
      </motion.div>

      {/* Question mark */}
      <motion.span
        initial={{ opacity: 0, scale: 0 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1.5, type: "spring", stiffness: 300, damping: 15 }}
        className="text-2xl sm:text-3xl"
      >
        ❓
      </motion.span>

      {/* Text */}
      <div className="text-center space-y-1">
        {["Identify 87 bird species", "from audio with maximum", "accuracy & efficiency"].map(
          (line, i) => (
            <motion.p
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 2 + i * 0.25, duration: 0.3 }}
              className="font-mono text-[9px] sm:text-xs tracking-[0.15em] uppercase text-foreground/80"
            >
              {line}
            </motion.p>
          )
        )}
      </div>
    </motion.div>
  )
}
