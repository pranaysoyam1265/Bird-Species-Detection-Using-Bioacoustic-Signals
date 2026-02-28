"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useEffect, useState } from "react"
import Lottie from "lottie-react"
import micAnimationData from "@/assets/mic.json"

const LEFT_LABELS = ["Record", "Upload", "Extract"]
const RIGHT_LABELS = ["Classify", "Score", "Report"]

/* ── Individual icon renderers (each uses its correct viewBox) ── */
const ICONS = [
  // 0: Mic (Lottie animation)
  (s: number) => (
    <div className="dark:invert" style={{ width: s * 2.8, height: s * 2.8, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <Lottie animationData={micAnimationData} loop autoplay style={{ width: "100%", height: "100%" }} />
    </div>
  ),
  // 1: Soundwave (viewBox 32)
  (s: number) => (
    <svg viewBox="0 0 32 32" width={s} height={s} fill="currentColor">
      <path d="M2 10a1 1 0 0 0-1 1v10a1 1 0 0 0 2 0V11a1 1 0 0 0-1-1zm4-5a1 1 0 0 0-1 1v20a1 1 0 0 0 2 0V6a1 1 0 0 0-1-1zm12 5a1 1 0 0 0-1 1v10a1 1 0 0 0 2 0V11a1 1 0 0 0-1-1zm-4-5a1 1 0 0 0-1 1v20a1 1 0 0 0 2 0V6a1 1 0 0 0-1-1zm-4-4a1 1 0 0 0-1 1v28a1 1 0 0 0 2 0V2a1 1 0 0 0-1-1zm20 9a1 1 0 0 0-1 1v10a1 1 0 0 0 2 0V11a1 1 0 0 0-1-1zm-4-5a1 1 0 0 0-1 1v20a1 1 0 0 0 2 0V6a1 1 0 0 0-1-1zm-4-4a1 1 0 0 0-1 1v28a1 1 0 0 0 2 0V2a1 1 0 0 0-1-1z" />
    </svg>
  ),
  // 2: Frequency / mixer (viewBox 512)
  (s: number) => (
    <svg viewBox="0 0 512 512" width={s} height={s} fill="currentColor">
      <path d="M61.28 216H258.093a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H350.381a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 1 0 0 20zm242.957-37.393A27.394 27.394 0 1 1 277.026 206 27.334 27.334 0 0 1 304.237 178.607zM61.28 116H161.619a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H253.907a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 0 0 0 20zM207.763 78.607A27.394 27.394 0 1 1 180.552 106 27.334 27.334 0 0 1 207.763 78.607zM61.28 316H161.619a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H253.907a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 1 0 0 20zm146.483-37.393A27.394 27.394 0 1 1 180.552 306 27.334 27.334 0 0 1 207.763 278.607zM61.28 416H258.093a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H350.381a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 1 0 0 20zm242.957-37.393A27.394 27.394 0 1 1 277.026 406 27.334 27.334 0 0 1 304.237 378.607z" />
    </svg>
  ),
  // 3: Pulse (viewBox 24)
  (s: number) => (
    <svg viewBox="0 0 24 24" width={s} height={s} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  ),
  // 4: Radar (viewBox 24)
  (s: number) => (
    <svg viewBox="0 0 24 24" width={s} height={s} fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M19.07 4.93A10 10 0 0 0 6.99 3.34" />
      <path d="M4 6h.01" />
      <path d="M2.29 9.62A10 10 0 1 0 21.31 8.35" />
      <path d="M16.24 7.76A6 6 0 1 0 8.23 16.67" />
      <path d="M12 18h.01" />
      <circle cx="12" cy="12" r="2" />
      <path d="m13.41 10.59 5.66-5.66" />
    </svg>
  ),
  // 5: Bird (viewBox 32)
  (s: number) => (
    <svg viewBox="0 0 32 32" width={s} height={s} fill="currentColor">
      <path d="M10.0281 9.88989C10.5804 9.88989 11.0281 9.44218 11.0281 8.88989C11.0281 8.33761 10.5804 7.88989 10.0281 7.88989C9.47585 7.88989 9.02814 8.33761 9.02814 8.88989C9.02814 9.44218 9.47585 9.88989 10.0281 9.88989Z" />
      <path d="M25.2258 26L25.2225 25.99H29.1181C30.0704 25.99 30.8381 25.2223 30.8381 24.27V15.12C30.8381 13.8559 29.5198 13.0437 28.3915 13.5609L22.0602 16.5096L19.8887 10H19.8844L18.5222 5.91995C19.9492 5.54385 20.9981 4.24692 20.9981 2.7C20.9981 1.76771 20.2304 1 19.2981 1H11.8381C8.55625 1 5.80766 3.26158 5.04484 6.30661L2.04118 7.88181L2.03886 7.88303C1.06651 8.39622 0.899966 9.70187 1.68056 10.4564L4.2342 12.8731L3.56465 18.6759C3.1167 22.5778 6.15894 26 10.0881 26H10.7781L10.2781 28.5001H8.47812C7.74812 28.5001 7.14812 29.0201 7.00812 29.7101C6.96812 29.8601 7.09812 30.0001 7.24812 30.0001H14.7181L14.7279 29.9999H19.7781C19.9381 29.9999 20.0581 29.8599 20.0281 29.7099C19.8881 29.0199 19.2781 28.4999 18.5581 28.4999H16.7781L17.2781 26H25.2258Z" />
    </svg>
  ),
]

function CyclingIcon() {
  const [step, setStep] = useState(0)
  const iconSize = 38

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((s) => (s + 1) % ICONS.length)
    }, 3500)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, scale: 0.6 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.6 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          style={{ display: "flex", alignItems: "center", justifyContent: "center", color: "hsl(var(--foreground))" }}
        >
          {ICONS[step](iconSize)}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

function PillLabel({
  label,
  x,
  y,
  delay,
}: {
  label: string
  x: number
  y: number
  delay: number
}) {
  return (
    <motion.g
      initial={{ opacity: 0, x: x > 400 ? 20 : -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay }}
    >
      <rect
        x={x}
        y={y}
        width={80}
        height={26}
        rx={13}
        fill="none"
        stroke="hsl(var(--foreground))"
        strokeWidth={1.5}
      />
      <text
        x={x + 40}
        y={y + 17}
        textAnchor="middle"
        fill="hsl(var(--foreground))"
        fontSize={10}
        fontFamily="var(--font-mono), monospace"
        fontWeight={500}
        letterSpacing="0.05em"
      >
        {label}
      </text>
    </motion.g>
  )
}

export function WorkflowDiagram() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className="h-[200px] w-full" />
  }

  const centerX = 400
  const centerY = 100

  return (
    <div className="relative w-full max-w-[800px] mx-auto">
      <svg
        viewBox="0 0 800 200"
        className="w-full h-auto"
        role="img"
        aria-label="Workflow diagram showing bird audio analysis pipeline: Record, Upload, Extract, Classify, Score, Report"
      >
        {/* Left lines from center to left labels */}
        {LEFT_LABELS.map((_, i) => {
          const pillX = 60
          const pillY = 30 + i * 60
          return (
            <motion.line
              key={`left-line-${i}`}
              x1={centerX - 40}
              y1={centerY}
              x2={pillX + 80}
              y2={pillY + 13}
              stroke="hsl(var(--border))"
              strokeWidth={1}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 + i * 0.1 }}
            />
          )
        })}

        {/* Right lines from center to right labels */}
        {RIGHT_LABELS.map((_, i) => {
          const pillX = 660
          const pillY = 30 + i * 60
          return (
            <motion.line
              key={`right-line-${i}`}
              x1={centerX + 40}
              y1={centerY}
              x2={pillX}
              y2={pillY + 13}
              stroke="hsl(var(--border))"
              strokeWidth={1}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 + i * 0.1 }}
            />
          )
        })}

        {/* Data packets flowing along lines */}
        {LEFT_LABELS.map((_, i) => {
          const pillX = 60
          const pillY = 30 + i * 60
          return (
            <motion.circle
              key={`left-packet-${i}`}
              r={3}
              fill="var(--accent-hex)"
              initial={{ cx: pillX + 80, cy: pillY + 13 }}
              animate={{
                cx: [pillX + 80, centerX - 40],
                cy: [pillY + 13, centerY],
              }}
              transition={{
                duration: 1.8,
                delay: 0.8 + i * 0.6,
                repeat: Infinity,
                repeatDelay: 3,
                ease: "linear",
              }}
            />
          )
        })}

        {RIGHT_LABELS.map((_, i) => {
          const pillX = 660
          const pillY = 30 + i * 60
          return (
            <motion.circle
              key={`right-packet-${i}`}
              r={3}
              fill="var(--accent-hex)"
              initial={{ cx: centerX + 40, cy: centerY }}
              animate={{
                cx: [centerX + 40, pillX],
                cy: [centerY, pillY + 13],
              }}
              transition={{
                duration: 1.8,
                delay: 1.2 + i * 0.6,
                repeat: Infinity,
                repeatDelay: 3,
                ease: "linear",
              }}
            />
          )
        })}

        {/* Left pill labels */}
        {LEFT_LABELS.map((label, i) => (
          <PillLabel
            key={`left-${label}`}
            label={label}
            x={60}
            y={30 + i * 60}
            delay={0.1 + i * 0.1}
          />
        ))}

        {/* Right pill labels */}
        {RIGHT_LABELS.map((label, i) => (
          <PillLabel
            key={`right-${label}`}
            label={label}
            x={660}
            y={30 + i * 60}
            delay={0.1 + i * 0.1}
          />
        ))}

        {/* Center: cycling loader icons */}
        <motion.g
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4, delay: 0.1 }}
        >
          <rect
            x={centerX - 36}
            y={centerY - 36}
            width={72}
            height={72}
            fill="hsl(var(--muted))"
            stroke="hsl(var(--border))"
            strokeWidth={1.5}
          />
          {/* Cycling icons via foreignObject */}
          <foreignObject x={centerX - 25} y={centerY - 25} width={50} height={50}>
            <CyclingIcon />
          </foreignObject>
        </motion.g>
      </svg>
    </div>
  )
}
