import { NextRequest, NextResponse } from "next/server"

/**
 * GET /api/xeno-canto?species=Troglodytes+troglodytes
 *
 * Proxies a request to the Xeno-Canto API v3 to fetch recording locations
 * for a given bird species (by scientific name).
 * Returns an array of location objects with lat, lng, location, country, elevation.
 */

const GBIF_API = "https://api.gbif.org/v1/occurrence/search"

export const dynamic = "force-dynamic"

export async function GET(req: NextRequest) {
  const species = req.nextUrl.searchParams.get("species")
  if (!species) {
    return NextResponse.json({ error: "Missing species parameter" }, { status: 400 })
  }

  try {
    const url = `${GBIF_API}?scientificName=${encodeURIComponent(species)}&limit=100&hasCoordinate=true`

    const res = await fetch(url, {
      headers: { "Accept": "application/json" },
      next: { revalidate: 86400 }, // Cache for 24 hours
    })

    if (!res.ok) {
      console.error("[gbif] API error:", res.status)
      return NextResponse.json({ locations: [] })
    }

    const data = await res.json()
    const results = data.results || []

    // Extract unique locations with coordinates
    const seen = new Set<string>()
    const locations: Array<{
      lat: number
      lng: number
      location: string
      country: string
      elevation: string
      recordist: string
      url: string
      quality: string
    }> = []

    for (const rec of results) {
      const lat = rec.decimalLatitude
      const lng = rec.decimalLongitude
      if (typeof lat !== "number" || typeof lng !== "number") continue

      // Deduplicate by rounding coordinates to ~1km grid
      const key = `${lat.toFixed(2)},${lng.toFixed(2)}`
      if (seen.has(key)) continue
      seen.add(key)

      locations.push({
        lat,
        lng,
        location: rec.stateProvince || rec.locality || "Unknown Region",
        country: rec.country || "Unknown",
        elevation: rec.elevation !== undefined ? `${rec.elevation} m` : "—",
        recordist: rec.recordedBy || rec.publisher || "eBird/iNat Observer",
        url: rec.references || (rec.gbifID ? `https://www.gbif.org/occurrence/${rec.gbifID}` : ""),
        quality: rec.basisOfRecord || "OBSERVATION",
      })
    }

    return NextResponse.json({
      species: species,
      totalRecordings: data.count || 0,
      locations,
    })
  } catch (err) {
    console.error("[gbif] Fetch error:", err)
    return NextResponse.json({ locations: [] })
  }
}
