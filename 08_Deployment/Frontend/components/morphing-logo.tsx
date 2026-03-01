"use client"

/*
  Animated bird logo for BirdSense.
  Cycles through 4 real bird SVGs from /public with smooth CSS crossfade.
*/

import { useEffect, useState } from "react"
import Image from "next/image"

interface MorphingLogoProps {
  size?: number
}

const BIRD_SVGS = [
  "/bird-svgrepo-com.svg",
  "/bird-svgrepo-com (1).svg",
  "/bird-svgrepo-com (2).svg",
  "/bird-svgrepo-com (3).svg",
]

// How long each bird is shown (ms)
const HOLD_MS = 2500
// Crossfade duration (CSS transition)
const FADE_MS = 700

export function MorphingLogo({ size = 28 }: MorphingLogoProps) {
  const [current, setCurrent] = useState(0)
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    const interval = setInterval(() => {
      // Fade out
      setVisible(false)
      setTimeout(() => {
        setCurrent((c) => (c + 1) % BIRD_SVGS.length)
        // Fade in
        setVisible(true)
      }, FADE_MS)
    }, HOLD_MS + FADE_MS)

    return () => clearInterval(interval)
  }, [])

  return (
    <div
      style={{
        width: size,
        height: size,
        position: "relative",
        flexShrink: 0,
      }}
      aria-label="BirdSense logo"
    >
      <Image
        src={BIRD_SVGS[current]}
        alt="BirdSense bird logo"
        width={size}
        height={size}
        style={{
          width: size,
          height: size,
          objectFit: "contain",
          opacity: visible ? 1 : 0,
          transition: `opacity ${FADE_MS}ms ease-in-out`,
          filter: "invert(42%) sepia(93%) saturate(1352%) hue-rotate(5deg) brightness(119%) contrast(119%)",
        }}
        priority
        unoptimized
      />
    </div>
  )
}
