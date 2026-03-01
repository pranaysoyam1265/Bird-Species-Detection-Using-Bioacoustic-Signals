"use client"

import { useEffect, useRef, useState } from "react"

export function CountingNumber({
  value,
  duration = 1,
  suffix = "",
  decimals = 1,
  trigger = true,
}: {
  value: number
  duration?: number
  suffix?: string
  decimals?: number
  trigger?: boolean
}) {
  const [display, setDisplay] = useState("0")
  const rafRef = useRef<number>(0)

  useEffect(() => {
    if (!trigger) { setDisplay("0"); return }
    const start = performance.now()
    const dur = duration * 1000
    const step = (now: number) => {
      const elapsed = now - start
      const progress = Math.min(elapsed / dur, 1)
      // ease-out
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplay((eased * value).toFixed(decimals) + suffix)
      if (progress < 1) rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(rafRef.current)
  }, [value, duration, suffix, decimals, trigger])

  return <span>{display}</span>
}
