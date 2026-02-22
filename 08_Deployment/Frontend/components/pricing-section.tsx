"use client"

import { useEffect, useState } from "react"
import { ArrowRight, Check } from "lucide-react"
import { motion } from "framer-motion"

const ease = [0.22, 1, 0.36, 1] as const

/* ── data-stream status line ── */
function StatusLine() {
  const [accuracy, setAccuracy] = useState("0.00")

  useEffect(() => {
    const interval = setInterval(() => {
      setAccuracy((95.8 + Math.random() * 0.5).toFixed(2))
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex items-center gap-2 text-[10px] tracking-widest text-muted-foreground uppercase font-mono">
      <span className="h-1.5 w-1.5 bg-[#ea580c]" />
      <span>detection accuracy: {accuracy}%</span>
    </div>
  )
}

/* ── blinking cursor indicator ── */
function BlinkDot() {
  return <span className="inline-block h-2 w-2 bg-[#ea580c] animate-blink" />
}

/* ── feature config ── */
interface Feature {
  id: string
  name: string
  icon: string
  descriptor: string
  tag: string | null
  description: string
  bullets: string[]
  cta: string
  highlighted: boolean
}

const FEATURES: Feature[] = [
  {
    id: "audio-analysis",
    name: "AUDIO ANALYSIS",
    icon: "🎵",
    descriptor: "CORE",
    tag: null,
    description: "Upload any bird recording and get instant AI-powered identification.",
    bullets: [
      "Multiple format support (WAV, MP3, FLAC)",
      "Noise reduction preprocessing",
      "Chunk-based analysis for long recordings",
      "Real-time processing",
      "Confidence scoring for each prediction",
      "Works with field recordings",
    ],
    cta: "TRY NOW",
    highlighted: false,
  },
  {
    id: "species-search",
    name: "SPECIES SEARCH",
    icon: "🔍",
    descriptor: "SMART",
    tag: "HIGHLIGHTED",
    description: "Search your recordings for specific bird species you're looking for.",
    bullets: [
      "Target any of 87 species",
      "Adjustable detection sensitivity",
      "Temporal detection timeline",
      "Pinpoint exact timestamps",
      "Audio segment playback",
      "Confidence-based filtering",
    ],
    cta: "SEARCH NOW",
    highlighted: true,
  },
  {
    id: "visual-insights",
    name: "VISUAL INSIGHTS",
    icon: "📊",
    descriptor: "DETAILED",
    tag: null,
    description: "Deep dive into spectrograms, waveforms, and detection patterns.",
    bullets: [
      "Mel-spectrogram visualization",
      "Interactive waveform display",
      "Detection timeline view",
      "Top-K predictions chart",
      "Exportable results (CSV/JSON)",
      "Segment-by-segment breakdown",
    ],
    cta: "EXPLORE",
    highlighted: false,
  },
]

/* ── single feature card ── */
function FeatureCard({ feature, index }: { feature: Feature; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, filter: "blur(4px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ delay: index * 0.12, duration: 0.6, ease }}
      className={`flex flex-col h-full ${feature.highlighted
        ? "border-2 border-foreground bg-foreground text-background"
        : "border-2 border-foreground bg-background text-foreground"
        }`}
    >
      {/* Card header */}
      <div
        className={`flex items-center justify-between px-5 py-3 border-b-2 ${feature.highlighted ? "border-background/20" : "border-foreground"
          }`}
      >
        <span className="text-[10px] tracking-[0.2em] uppercase font-mono">
          {feature.name}
        </span>
        <div className="flex items-center gap-2">
          {feature.tag && (
            <span className="bg-[#ea580c] text-background text-[9px] tracking-[0.15em] uppercase px-2 py-0.5 font-mono">
              {feature.tag}
            </span>
          )}
          <span className="text-[10px] tracking-[0.2em] font-mono opacity-50">
            {String(index + 1).padStart(2, "0")}
          </span>
        </div>
      </div>

      {/* Icon + descriptor block */}
      <div className="px-5 pt-6 pb-4">
        <div className="flex items-baseline gap-3">
          <span className="text-4xl lg:text-5xl">{feature.icon}</span>
          <span
            className={`text-xs font-mono tracking-widest uppercase ${feature.highlighted ? "text-background/50" : "text-muted-foreground"
              }`}
          >
            {feature.descriptor}
          </span>
        </div>
        <p
          className={`text-xs font-mono mt-3 leading-relaxed ${feature.highlighted ? "text-background/60" : "text-muted-foreground"
            }`}
        >
          {feature.description}
        </p>
      </div>

      {/* Bullet list */}
      <div
        className={`flex-1 px-5 py-4 border-t-2 ${feature.highlighted ? "border-background/20" : "border-foreground"
          }`}
      >
        <div className="flex flex-col gap-3">
          {feature.bullets.map((bullet, bi) => (
            <motion.div
              key={bullet}
              initial={{ opacity: 0, x: -8 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.12 + 0.3 + bi * 0.04, duration: 0.35, ease }}
              className="flex items-start gap-3"
            >
              <Check
                size={12}
                strokeWidth={2.5}
                className="mt-0.5 shrink-0 text-[#ea580c]"
              />
              <span className="text-xs font-mono leading-relaxed">
                {bullet}
              </span>
            </motion.div>
          ))}
        </div>
      </div>

      {/* CTA */}
      <div className="px-5 pb-5 pt-3">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          className={`group w-full flex items-center justify-center gap-0 text-xs font-mono tracking-wider uppercase ${feature.highlighted
            ? "bg-background text-foreground"
            : "bg-foreground text-background"
            }`}
        >
          <span className="flex items-center justify-center w-9 h-9 bg-[#ea580c]">
            <ArrowRight size={14} strokeWidth={2} className="text-background" />
          </span>
          <span className="flex-1 py-2.5">{feature.cta}</span>
        </motion.button>
      </div>
    </motion.div>
  )
}

/* ── main features section ── */
export function PricingSection() {
  return (
    <section className="w-full px-6 py-20 lg:px-12">
      {/* Section label */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.5, ease }}
        className="flex items-center gap-4 mb-8"
      >
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          {"// SECTION: TOP_FEATURES"}
        </span>
        <div className="flex-1 border-t border-border" />
        <BlinkDot />
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          006
        </span>
      </motion.div>

      {/* Section header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-60px" }}
        transition={{ duration: 0.6, ease }}
        className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 mb-12"
      >
        <div className="flex flex-col gap-3">
          <h2 className="text-2xl lg:text-3xl font-mono font-bold tracking-tight uppercase text-foreground text-balance">
            Explore our core features
          </h2>
          <p className="text-xs lg:text-sm font-mono text-muted-foreground leading-relaxed max-w-md">
            Powerful AI-driven bird detection with real-time analysis, advanced visualization, and intelligent species search across 87 North American birds.
          </p>
        </div>
        <StatusLine />
      </motion.div>

      {/* Feature cards grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-0">
        {FEATURES.map((feature, i) => (
          <FeatureCard key={feature.id} feature={feature} index={i} />
        ))}
      </div>

      {/* Bottom note */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.5, duration: 0.5, ease }}
        className="flex items-center gap-3 mt-6"
      >
        <span className="text-[10px] tracking-[0.2em] uppercase text-muted-foreground font-mono">
          {"* All features available across plans. Upload audio and start detecting today."}
        </span>
        <div className="flex-1 border-t border-border" />
      </motion.div>
    </section>
  )
}
