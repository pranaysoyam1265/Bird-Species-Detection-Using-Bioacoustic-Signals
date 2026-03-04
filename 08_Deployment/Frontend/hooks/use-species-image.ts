"use client"

import { useState, useEffect } from "react"

const cache = new Map<string, string>()

const DEFAULT_BIRD_IMG =
  "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=600&h=400&fit=crop"

/**
 * Fetches a species-specific image from Wikipedia using the page image API.
 * Uses the scientific name for more accurate results, falls back to common name.
 * Results are cached in memory to avoid repeated API calls.
 */
export function useSpeciesImage(
  commonName: string,
  scientificName?: string
): { src: string; loading: boolean } {
  const [src, setSrc] = useState(() => cache.get(commonName) ?? DEFAULT_BIRD_IMG)
  const [loading, setLoading] = useState(!cache.has(commonName))

  useEffect(() => {
    if (cache.has(commonName)) {
      setSrc(cache.get(commonName)!)
      setLoading(false)
      return
    }

    let cancelled = false

    async function fetchImage() {
      // Try scientific name first (more unique on Wikipedia), then common name
      const queries = [
        scientificName,
        commonName,
      ].filter(Boolean) as string[]

      for (const query of queries) {
        try {
          const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(query.replace(/ /g, "_"))}`
          const res = await fetch(url)
          if (!res.ok) continue

          const data = await res.json()
          const imgUrl: string | undefined =
            data?.originalimage?.source ?? data?.thumbnail?.source

          if (imgUrl && !cancelled) {
            cache.set(commonName, imgUrl)
            setSrc(imgUrl)
            setLoading(false)
            return
          }
        } catch {
          // try next query
        }
      }

      // All queries exhausted — keep default
      if (!cancelled) {
        cache.set(commonName, DEFAULT_BIRD_IMG)
        setLoading(false)
      }
    }

    fetchImage()
    return () => { cancelled = true }
  }, [commonName, scientificName])

  return { src, loading }
}
