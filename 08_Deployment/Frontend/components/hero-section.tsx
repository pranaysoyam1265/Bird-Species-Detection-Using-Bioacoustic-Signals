"use client"

import { WorkflowDiagram } from "@/components/workflow-diagram"
import { ArrowRight } from "lucide-react"
import { motion } from "framer-motion"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"

const ease = [0.22, 1, 0.36, 1] as const

export function HeroSection() {
  const { user } = useAuth()
  const router = useRouter()

  const handleUploadClick = () => {
    router.push(user ? "/analyze" : "/login")
  }

  return (
    <section className="relative w-full px-12 pt-6 pb-12 lg:px-24 lg:pt-10 lg:pb-16">
      <div className="flex flex-col items-center text-center">
        {/* Top headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.7, ease }}
          className="font-pixel text-4xl sm:text-6xl lg:text-7xl xl:text-8xl tracking-tight text-foreground mb-2 select-none"
        >
          LISTEN. DETECT.
        </motion.h1>

        {/* Central Workflow Diagram */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.15, ease }}
          className="w-full max-w-2xl my-4 lg:my-6"
        >
          <WorkflowDiagram />
        </motion.div>

        {/* Bottom headline */}
        <motion.h1
          initial={{ opacity: 0, y: 30, filter: "blur(8px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.7, delay: 0.25, ease }}
          className="font-pixel text-4xl sm:text-6xl lg:text-7xl xl:text-8xl tracking-tight text-foreground mb-4 select-none"
          aria-hidden="true"
        >
          IDENTIFY.
        </motion.h1>

        {/* Sub-headline */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.45, ease }}
          className="text-xs lg:text-sm text-muted-foreground max-w-md mb-6 leading-relaxed font-mono"
        >
          BirdSense is the AI-powered detection layer between field recordings and species identification. 96% accuracy. 87 species. Spectrogram-level analysis.
        </motion.p>

        <motion.button
          type="button"
          onClick={handleUploadClick}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6, ease }}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.97 }}
          className="group inline-flex items-center gap-0 bg-foreground text-background text-sm font-mono tracking-wider uppercase cursor-pointer"
        >
          <span className="flex items-center justify-center w-10 h-10 bg-accent">
            <motion.span
              className="inline-flex"
              whileHover={{ x: 3 }}
              transition={{ type: "spring", stiffness: 400, damping: 20 }}
            >
              <ArrowRight size={16} strokeWidth={2} className="text-background" />
            </motion.span>
          </span>
          <span className="px-5 py-2.5">
            Upload Audio
          </span>
        </motion.button>
      </div>
    </section>
  )
}
