"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { MODELS } from "../data/models"
import { MetricBar } from "../ui/MetricBar"

const METRICS = [
  {
    title: "Metric 1: Accuracy",
    hint: "Higher = Better",
    key: "accuracy" as const,
    max: 100,
    fmt: (m: typeof MODELS[number]) => `${m.accuracy}%`,
    val: (m: typeof MODELS[number]) => m.accuracy,
    winId: "efficientnetb2",
    loseId: "",
  },
  {
    title: "Metric 2: Model Size",
    hint: "Lower = Better",
    key: "parametersNum" as const,
    max: 140,
    fmt: (m: typeof MODELS[number]) => m.parameters,
    val: (m: typeof MODELS[number]) => m.parametersNum,
    winId: "efficientnetb0",
    loseId: "vgg16",
  },
  {
    title: "Metric 3: Inference Speed",
    hint: "Lower = Better",
    key: "inferenceTime" as const,
    max: 80,
    fmt: (m: typeof MODELS[number]) => `${m.inferenceTime}ms`,
    val: (m: typeof MODELS[number]) => m.inferenceTime,
    winId: "mobilenetv3",
    loseId: "vgg16",
  },
]

export function BattleScene() {
  const [visibleMetric, setVisibleMetric] = useState(0)

  useEffect(() => {
    const t1 = setTimeout(() => setVisibleMetric(1), 3500)
    const t2 = setTimeout(() => setVisibleMetric(2), 6500)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full h-full flex flex-col gap-2 sm:gap-3 p-3 sm:p-6 overflow-y-auto"
    >
      <motion.p
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-[10px] sm:text-xs tracking-[0.25em] uppercase text-accent font-bold text-center"
      >
        ⚔️ The Battle
      </motion.p>

      {METRICS.map((metric, mi) => (
        <motion.div
          key={metric.key}
          initial={{ opacity: 0, y: 20 }}
          animate={mi <= visibleMetric ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.4 }}
          className="border border-foreground/20 bg-card/30 p-2 sm:p-3"
        >
          <div className="flex items-center justify-between mb-2">
            <p className="font-mono text-[8px] sm:text-[10px] tracking-[0.2em] uppercase text-foreground font-bold">
              {metric.title}
            </p>
            <p className="font-mono text-[7px] sm:text-[8px] tracking-wider text-muted-foreground uppercase">
              {metric.hint}
            </p>
          </div>
          <div className="space-y-1">
            {MODELS.map((m, idx) => (
              <MetricBar
                key={m.id}
                label={m.name}
                value={metric.val(m)}
                maxValue={metric.max}
                displayValue={mi <= visibleMetric ? metric.fmt(m) : ""}
                isWinner={m.id === metric.winId}
                isLoser={m.id === metric.loseId}
                delay={mi <= visibleMetric ? 0.3 + idx * 0.1 : 0}
              />
            ))}
          </div>
        </motion.div>
      ))}
    </motion.div>
  )
}
