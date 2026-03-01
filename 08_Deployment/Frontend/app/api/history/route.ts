import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { getDetectionsByUser, countDetectionsByUser, deleteDetectionsByUser } from "@/lib/db"

export const dynamic = "force-dynamic"

export async function GET(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const { searchParams } = new URL(req.url)
  const limit = Math.min(Number(searchParams.get("limit") || 50), 100)
  const offset = Math.max(Number(searchParams.get("offset") || 0), 0)

  const rows = getDetectionsByUser(user.id, limit, offset)
  const total = countDetectionsByUser(user.id)

  // Parse JSON strings back to objects
  const detections = rows.map((r) => ({
    id: r.id,
    filename: r.filename,
    date: r.date,
    time: r.time,
    duration: r.duration,
    topSpecies: r.top_species,
    topScientific: r.top_scientific,
    topConfidence: r.top_confidence,
    predictions: JSON.parse(r.predictions),
    segments: JSON.parse(r.segments),
    audioUrl: r.audio_url,
  }))

  return NextResponse.json({ detections, total, limit, offset })
}

/**
 * Bulk delete:  DELETE /api/history  body: { ids: string[] }
 */
export async function DELETE(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const body = await req.json().catch(() => null)
  const ids: string[] = body?.ids
  if (!Array.isArray(ids) || ids.length === 0) {
    return NextResponse.json({ error: "INVALID_IDS" }, { status: 400 })
  }

  const deleted = deleteDetectionsByUser(user.id, ids)
  return NextResponse.json({ deleted })
}
