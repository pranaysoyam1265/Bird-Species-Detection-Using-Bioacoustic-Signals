// ─── Detection History Store ──────────────────────────────────────────
// Local storage functions (offline / legacy) + API-backed functions (server)

export interface DetectionSegment {
  startTime: number
  endTime: number
  confidence: number
  species: string
}

export interface DetectionRecord {
  id: string
  filename: string
  date: string        // ISO date  e.g. "2024-02-23"
  time: string        // HH:MM:SS  e.g. "08:32:15"
  duration: number    // audio length in seconds
  topSpecies: string
  topScientific: string
  topConfidence: number
  predictions: { species: string; confidence: number }[]
  segments: DetectionSegment[]
  audioUrl?: string   // base64 data URI (omitted for large files)
}

const STORAGE_KEY = "birdsense_history"

// ════════════════════════════════════════════════════════════════
// LOCAL STORAGE  (offline fallback — kept for backward compat)
// ════════════════════════════════════════════════════════════════

// ── Read ──

export function getDetections(): DetectionRecord[] {
  if (typeof window === "undefined") return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const records: DetectionRecord[] = JSON.parse(raw)
    // newest first
    return records.sort((a, b) => `${b.date}T${b.time}`.localeCompare(`${a.date}T${a.time}`))
  } catch {
    return []
  }
}

export function getDetectionById(id: string): DetectionRecord | undefined {
  return getDetections().find((r) => r.id === id)
}

// ── Write ──

export function saveDetection(record: DetectionRecord): void {
  const existing = getDetections()
  existing.unshift(record)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(existing))
}

// ── Delete ──

export function deleteDetections(ids: string[]): void {
  const idSet = new Set(ids)
  const remaining = getDetections().filter((r) => !idSet.has(r.id))
  localStorage.setItem(STORAGE_KEY, JSON.stringify(remaining))
}

export function clearAllDetections(): void {
  localStorage.removeItem(STORAGE_KEY)
}

// ════════════════════════════════════════════════════════════════
// API-BACKED FUNCTIONS  (server-side storage via /api/history)
// ════════════════════════════════════════════════════════════════

/**
 * Fetch detection history from the server.
 * Falls back to localStorage if the API call fails.
 */
export async function fetchDetections(
  limit = 50,
  offset = 0,
): Promise<{ detections: DetectionRecord[]; total: number }> {
  try {
    const res = await fetch(`/api/history?limit=${limit}&offset=${offset}`)
    if (!res.ok) throw new Error("API error")
    return await res.json()
  } catch {
    // Fallback to localStorage
    const all = getDetections()
    return { detections: all.slice(offset, offset + limit), total: all.length }
  }
}

/**
 * Fetch a single detection by ID from the server.
 */
export async function fetchDetectionById(id: string): Promise<DetectionRecord | null> {
  try {
    const res = await fetch(`/api/history/${id}`)
    if (!res.ok) {
      // Fallback to localStorage
      return getDetectionById(id) || null
    }
    return await res.json()
  } catch {
    return getDetectionById(id) || null
  }
}

/**
 * Delete detections on the server.
 * Also removes from localStorage for consistency.
 */
export async function deleteDetectionsApi(ids: string[]): Promise<void> {
  // Remove from localStorage immediately
  deleteDetections(ids)

  try {
    await fetch("/api/history", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    })
  } catch {
    // Server delete failed — localStorage already cleaned
  }
}

/**
 * Send audio to the server for ML detection.
 * Returns the detection result if successful.
 */
export async function detectAudio(
  file: File,
  options: { topK?: number; confidenceThreshold?: number; noiseReduction?: boolean } = {},
): Promise<DetectionRecord> {
  const form = new FormData()
  form.append("audio_file", file)
  if (options.topK) form.append("top_k", String(options.topK))
  if (options.confidenceThreshold != null) form.append("confidence_threshold", String(options.confidenceThreshold / 100))
  if (options.noiseReduction) form.append("noise_reduction", "true")

  const res = await fetch("/api/detect", { method: "POST", body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "DETECTION_FAILED" }))
    throw new Error(err.error || err.detail || "Detection failed")
  }

  const result = await res.json()

  // Convert API response to DetectionRecord format
  const record: DetectionRecord = {
    id: crypto.randomUUID(),
    filename: file.name,
    date: new Date().toISOString().split("T")[0],
    time: new Date().toTimeString().split(" ")[0],
    duration: Math.round(result.duration),
    topSpecies: result.top_species,
    topScientific: result.top_scientific,
    topConfidence: result.top_confidence,
    predictions: result.predictions.map((p: { species: string; confidence: number }) => ({
      species: p.species,
      confidence: p.confidence,
    })),
    segments: result.segments.map((s: { start_time: number; end_time: number; species: string; confidence: number }) => ({
      startTime: s.start_time,
      endTime: s.end_time,
      species: s.species,
      confidence: s.confidence,
    })),
  }

  // Also save to localStorage as cache
  saveDetection(record)

  return record
}

// ════════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════════

/** Convert a File to a base64 data URI. Resolves to undefined if file > 2 MB. */
export function fileToDataUri(file: File): Promise<string | undefined> {
  const MAX_SIZE = 2 * 1024 * 1024 // 2 MB
  if (file.size > MAX_SIZE) return Promise.resolve(undefined)

  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => resolve(undefined)
    reader.readAsDataURL(file)
  })
}

/** Build a DetectionRecord from Analyze page state. */
export function buildRecord(
  file: { name: string },
  duration: number,
  results: {
    topSpecies: string
    topScientific: string
    topConfidence: number
    predictions: { species: string; confidence: number }[]
    segments: DetectionSegment[]
  },
  audioUrl?: string,
): DetectionRecord {
  const now = new Date()
  return {
    id: crypto.randomUUID(),
    filename: file.name,
    date: now.toISOString().split("T")[0],
    time: now.toTimeString().split(" ")[0],
    duration: Math.round(duration),
    topSpecies: results.topSpecies,
    topScientific: results.topScientific,
    topConfidence: results.topConfidence,
    predictions: results.predictions,
    segments: results.segments,
    audioUrl,
  }
}

/** Count how many detections exist per species name (top species only). */
export function getDetectionCountsBySpecies(): Record<string, number> {
  const records = getDetections()
  const counts: Record<string, number> = {}
  for (const r of records) {
    counts[r.topSpecies] = (counts[r.topSpecies] || 0) + 1
  }
  return counts
}

