"use client"

import { useState, useEffect, useRef } from "react"
import { AnimatePresence } from "framer-motion"
import { ChallengeScene } from "./scenes/ChallengeScene"
import { ContendersScene } from "./scenes/ContendersScene"
import { BattleScene } from "./scenes/BattleScene"
import { VerdictScene } from "./scenes/VerdictScene"
import { ChampionScene } from "./scenes/ChampionScene"
import { SCENE_DURATIONS, TOTAL_DURATION } from "./data/models"

const SCENES = [ChallengeScene, ContendersScene, BattleScene, VerdictScene, ChampionScene]

export function ModelComparisonAnimation() {
  const [currentScene, setCurrentScene] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [hasEnded, setHasEnded] = useState(false)
  const progressRef = useRef(0)
  const containerRef = useRef<HTMLDivElement>(null)

  /* Start when the component scrolls into view */
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isPlaying && !hasEnded) setIsPlaying(true)
      },
      { threshold: 0.3 }
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [isPlaying, hasEnded])

  /* Timer loop */
  useEffect(() => {
    if (!isPlaying) return
    const interval = setInterval(() => {
      progressRef.current += 100 / (TOTAL_DURATION * 10) // 10 fps
      if (progressRef.current >= 100) {
        setIsPlaying(false)
        setHasEnded(true)
        setCurrentScene(4)
        return
      }
      let acc = 0
      for (let i = 0; i < SCENE_DURATIONS.length; i++) {
        const pct = (SCENE_DURATIONS[i] / TOTAL_DURATION) * 100
        if (progressRef.current < acc + pct) {
          setCurrentScene(i)
          break
        }
        acc += pct
      }
    }, 100)
    return () => clearInterval(interval)
  }, [isPlaying])



  const Scene = SCENES[currentScene]

  return (
    <div ref={containerRef} className="relative flex flex-col h-full bg-background overflow-hidden">
      {/* Header bar — matches other bento cards */}
      <div className="flex items-center justify-between border-b-2 border-foreground px-4 py-2">
        <span className="text-[10px] tracking-widest text-muted-foreground uppercase">
          model.comparison
        </span>
        <span className="text-[10px] tracking-widest text-muted-foreground">
          {`SCENE:${String(currentScene + 1).padStart(2, "0")}/05`}
        </span>
      </div>

      {/* Viewport — fills remaining space */}
      <div className="flex-1 relative overflow-hidden min-h-0">
        <AnimatePresence mode="wait">
          <Scene key={`scene-${currentScene}`} />
        </AnimatePresence>
      </div>
    </div>
  )
}
