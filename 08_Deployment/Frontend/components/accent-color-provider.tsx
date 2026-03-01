"use client"

import { useEffect } from "react"

const ACCENT_KEY = "birdsense-pref-accent"
const DEFAULT_ACCENT = "var(--accent-hex)"

/**
 * Convert a hex colour to an HSL string without the hsl() wrapper,
 * e.g. "var(--accent-hex)" → "21 90% 48%", suitable for Tailwind's
 * `hsl(var(--accent))` pattern.
 */
function hexToHslValues(hex: string): string {
  const h = hex.replace("#", "")
  const r = parseInt(h.substring(0, 2), 16) / 255
  const g = parseInt(h.substring(2, 4), 16) / 255
  const b = parseInt(h.substring(4, 6), 16) / 255

  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  let s = 0
  let hue = 0

  if (max !== min) {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    switch (max) {
      case r: hue = ((g - b) / d + (g < b ? 6 : 0)); break
      case g: hue = ((b - r) / d + 2); break
      case b: hue = ((r - g) / d + 4); break
    }
    hue *= 60
  }

  return `${Math.round(hue)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`
}

/**
 * Reads the saved accent colour from localStorage and sets the
 * `--accent` CSS custom property on <html> (the one Tailwind reads via
 * its `accent` colour utility defined in tailwind.config). Also sets
 * `--accent-hex` for any one-off inline-style usage.
 */
export function AccentColorProvider() {
  useEffect(() => {
    try {
      const saved = localStorage.getItem(ACCENT_KEY)
      const color = saved || DEFAULT_ACCENT
      document.documentElement.style.setProperty("--accent", hexToHslValues(color))
      document.documentElement.style.setProperty("--accent-hex", color)
    } catch {
      document.documentElement.style.setProperty("--accent", hexToHslValues(DEFAULT_ACCENT))
      document.documentElement.style.setProperty("--accent-hex", DEFAULT_ACCENT)
    }
  }, [])

  return null
}
