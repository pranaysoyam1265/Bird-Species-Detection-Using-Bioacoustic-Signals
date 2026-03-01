/* ─── Shared formatting utilities ─── */

/**
 * Format a duration in seconds as m:ss
 * @example fmtDuration(125) → "2:05"
 */
export function fmtDuration(s: number): string {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}:${sec.toString().padStart(2, "0")}`
}

/**
 * Get a colour for a confidence percentage.
 * ≥ 90 → green, ≥ 70 → yellow, ≥ 50 → accent orange, else red.
 */
export function getConfColor(c: number): string {
  if (c >= 90) return "#22c55e"
  if (c >= 70) return "#eab308"
  if (c >= 50) return "var(--accent-hex)"
  return "#ef4444"
}

/**
 * Get a colour for an accuracy percentage (species cards).
 * ≥ 95 → green, ≥ 90 → lime, ≥ 85 → yellow, ≥ 80 → accent orange, else red.
 */
export function getAccuracyColor(accuracy: number): string {
  if (accuracy >= 95) return "#22c55e"
  if (accuracy >= 90) return "#84cc16"
  if (accuracy >= 85) return "#eab308"
  if (accuracy >= 80) return "var(--accent-hex)"
  return "#ef4444"
}
