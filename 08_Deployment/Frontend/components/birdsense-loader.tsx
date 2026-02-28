"use client"

import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"

/* ─────────────────────────────────────────────
   INLINE SVG ICONS — sequence:
   mic → soundwave → frequency → pulse → radar → bird
   ───────────────────────────────────────────── */

function MicIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
    </svg>
  )
}

function SoundwaveIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="currentColor"
    >
      <path d="M2 10a1 1 0 0 0-1 1v10a1 1 0 0 0 2 0V11a1 1 0 0 0-1-1zm4-5a1 1 0 0 0-1 1v20a1 1 0 0 0 2 0V6a1 1 0 0 0-1-1zm12 5a1 1 0 0 0-1 1v10a1 1 0 0 0 2 0V11a1 1 0 0 0-1-1zm-4-5a1 1 0 0 0-1 1v20a1 1 0 0 0 2 0V6a1 1 0 0 0-1-1zm-4-4a1 1 0 0 0-1 1v28a1 1 0 0 0 2 0V2a1 1 0 0 0-1-1zm20 9a1 1 0 0 0-1 1v10a1 1 0 0 0 2 0V11a1 1 0 0 0-1-1zm-4-5a1 1 0 0 0-1 1v20a1 1 0 0 0 2 0V6a1 1 0 0 0-1-1zm-4-4a1 1 0 0 0-1 1v28a1 1 0 0 0 2 0V2a1 1 0 0 0-1-1z" />
    </svg>
  )
}

function FrequencyIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      fill="currentColor"
    >
      <path d="M61.28 216H258.093a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H350.381a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 1 0 0 20zm242.957-37.393A27.394 27.394 0 1 1 277.026 206 27.334 27.334 0 0 1 304.237 178.607zM61.28 116H161.619a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H253.907a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 0 0 0 20zM207.763 78.607A27.394 27.394 0 1 1 180.552 106 27.334 27.334 0 0 1 207.763 78.607zM61.28 316H161.619a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H253.907a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 1 0 0 20zm146.483-37.393A27.394 27.394 0 1 1 180.552 306 27.334 27.334 0 0 1 207.763 278.607zM61.28 416H258.093a47.168 47.168 0 0 0 92.288 0H450.72a10 10 0 0 0 0-20H350.381a47.168 47.168 0 0 0-92.288 0H61.28a10 10 0 1 0 0 20zm242.957-37.393A27.394 27.394 0 1 1 277.026 406 27.334 27.334 0 0 1 304.237 378.607z" />
    </svg>
  )
}

function PulseIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  )
}

function RadarIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19.07 4.93A10 10 0 0 0 6.99 3.34" />
      <path d="M4 6h.01" />
      <path d="M2.29 9.62A10 10 0 1 0 21.31 8.35" />
      <path d="M16.24 7.76A6 6 0 1 0 8.23 16.67" />
      <path d="M12 18h.01" />
      <path d="M17.99 11.66A6 6 0 0 1 15.77 16.67" />
      <circle cx="12" cy="12" r="2" />
      <path d="m13.41 10.59 5.66-5.66" />
    </svg>
  )
}

function BirdIcon({ size = 48 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="currentColor"
    >
      <path d="M10.0281 9.88989C10.5804 9.88989 11.0281 9.44218 11.0281 8.88989C11.0281 8.33761 10.5804 7.88989 10.0281 7.88989C9.47585 7.88989 9.02814 8.33761 9.02814 8.88989C9.02814 9.44218 9.47585 9.88989 10.0281 9.88989Z" />
      <path d="M25.2258 26L25.2225 25.99H29.1181C30.0704 25.99 30.8381 25.2223 30.8381 24.27V15.12C30.8381 13.8559 29.5198 13.0437 28.3915 13.5609L22.0602 16.5096L19.8887 10H19.8844L18.5222 5.91995C19.9492 5.54385 20.9981 4.24692 20.9981 2.7C20.9981 1.76771 20.2304 1 19.2981 1H11.8381C8.55625 1 5.80766 3.26158 5.04484 6.30661L2.04118 7.88181L2.03886 7.88303C1.06651 8.39622 0.899966 9.70187 1.68056 10.4564L4.2342 12.8731L3.56465 18.6759C3.1167 22.5778 6.15894 26 10.0881 26H10.7781L10.2781 28.5001H8.47812C7.74812 28.5001 7.14812 29.0201 7.00812 29.7101C6.96812 29.8601 7.09812 30.0001 7.24812 30.0001H14.7181L14.7279 29.9999H19.7781C19.9381 29.9999 20.0581 29.8599 20.0281 29.7099C19.8881 29.0199 19.2781 28.4999 18.5581 28.4999H16.7781L17.2781 26H25.2258ZM6.84588 7.67694C7.02108 5.06562 9.19085 3 11.8381 3H18.9645C18.8295 3.59192 18.3026 4.03 17.6681 4.03H15.2901L16.1886 5.54108C16.3624 5.83345 16.5025 6.15921 16.6128 6.51568L16.6161 6.52622L18.4436 12H18.4476L20.8723 19.2691L20.8792 19.2658L22.1857 23.1824H19.2896C15.6653 23.1824 12.7196 20.2458 12.7196 16.6124C12.7196 16.3362 12.4957 16.1124 12.2196 16.1124C11.9434 16.1124 11.7196 16.3362 11.7196 16.6124C11.7196 20.2278 14.2509 23.2442 17.6307 24H10.0881C7.35749 24 5.23991 21.6219 5.55159 18.9041L6.84427 7.70084L6.84588 7.67694ZM23.5093 23.99L21.7917 18.8409L28.8381 15.5591V23.99H23.5093ZM15.7781 26L15.2781 28.4999H13.4781L13.4549 28.5001H11.7781L12.2781 26H15.7781ZM4.2239 10.1097L3.4663 9.39278L4.36258 8.92275L4.2239 10.1097Z" />
    </svg>
  )
}

/* ─────────────────────────────────────────────
   LOADER SEQUENCE DEFINITION
   ───────────────────────────────────────────── */

const STEPS = [
  { icon: MicIcon, label: "RECORDING" },
  { icon: SoundwaveIcon, label: "EXTRACTING" },
  { icon: FrequencyIcon, label: "ANALYZING" },
  { icon: PulseIcon, label: "PROCESSING" },
  { icon: RadarIcon, label: "DETECTING" },
  { icon: BirdIcon, label: "IDENTIFIED" },
]

/* ─────────────────────────────────────────────
   MAIN LOADER COMPONENT
   ───────────────────────────────────────────── */

interface BirdSenseLoaderProps {
  /** Size of the icon area in px */
  size?: number
  /** Text to show below the animation. If null, shows step labels. */
  statusText?: string | null
  /** If true, renders fullscreen overlay */
  fullscreen?: boolean
}

export function BirdSenseLoader({
  size = 48,
  statusText = null,
  fullscreen = false,
}: BirdSenseLoaderProps) {
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % STEPS.length)
    }, 800)
    return () => clearInterval(interval)
  }, [])

  const currentStep = STEPS[activeStep]
  const IconComponent = currentStep.icon

  const content = (
    <div className="flex flex-col items-center gap-6">
      {/* Icon container with border */}
      <div className="relative border-2 border-foreground p-6 bg-background">
        {/* Corner accents */}
        <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent" />
        <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent" />
        <span className="absolute -bottom-[3px] -left-[3px] w-2 h-2 bg-accent" />
        <span className="absolute -bottom-[3px] -right-[3px] w-2 h-2 bg-accent" />

        {/* Animated icon */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeStep}
            initial={{ opacity: 0, scale: 0.5, filter: "blur(8px)" }}
            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, scale: 1.3, filter: "blur(6px)" }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="text-foreground"
          >
            <IconComponent size={size} />
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Step progress dots */}
      <div className="flex items-center gap-1.5">
        {STEPS.map((_, i) => (
          <span
            key={i}
            className="h-1.5 transition-all duration-200"
            style={{
              width: i === activeStep ? 16 : 6,
              backgroundColor:
                i === activeStep
                  ? "var(--accent-hex)"
                  : "hsl(var(--muted-foreground))",
            }}
          />
        ))}
      </div>

      {/* Label */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeStep}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.2 }}
          className="flex flex-col items-center gap-1"
        >
          <span className="text-xs font-mono tracking-[0.25em] uppercase text-foreground font-bold">
            {statusText ?? currentStep.label}
          </span>
          <span className="text-[10px] font-mono tracking-[0.2em] uppercase text-muted-foreground">
            STEP {String(activeStep + 1).padStart(2, "0")}/{String(STEPS.length).padStart(2, "0")}
          </span>
        </motion.div>
      </AnimatePresence>

      {/* Pulsing live circle */}
      <div className="flex items-center gap-3">
        <svg width="24" height="24" viewBox="0 0 48 48">
          <circle cx="24" cy="24" r="20" fill="none" stroke="hsl(var(--border))" strokeWidth="1.5" />
          {/* Crosshair lines */}
          <line x1="24" y1="8" x2="24" y2="40" stroke="hsl(var(--foreground))" strokeWidth="2" />
          <line x1="8" y1="24" x2="40" y2="24" stroke="hsl(var(--foreground))" strokeWidth="2" />
          <line x1="12" y1="12" x2="36" y2="36" stroke="hsl(var(--foreground))" strokeWidth="1.5" />
          <line x1="36" y1="12" x2="12" y2="36" stroke="hsl(var(--foreground))" strokeWidth="1.5" />
          {/* Pulsing orange ring */}
          <circle cx="24" cy="24" r="18" fill="none" stroke="var(--accent-hex)" strokeWidth="1.5">
            <animate attributeName="r" values="18;22;18" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.8;0.2;0.8" dur="2s" repeatCount="indefinite" />
          </circle>
        </svg>
        <span className="text-[10px] font-mono tracking-widest text-muted-foreground uppercase">
          birdsense.detect
        </span>
      </div>
    </div>
  )

  if (fullscreen) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-sm dot-grid-bg">
        {content}
      </div>
    )
  }

  return content
}
