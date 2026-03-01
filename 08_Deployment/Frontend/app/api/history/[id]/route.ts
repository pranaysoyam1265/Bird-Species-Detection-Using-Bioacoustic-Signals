import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { getDetectionById, deleteDetectionById } from "@/lib/db"

export const dynamic = "force-dynamic"

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const { id } = await params
  const row = getDetectionById(user.id, id)

  if (!row) {
    return NextResponse.json({ error: "NOT_FOUND" }, { status: 404 })
  }

  return NextResponse.json({
    id: row.id,
    filename: row.filename,
    date: row.date,
    time: row.time,
    duration: row.duration,
    topSpecies: row.top_species,
    topScientific: row.top_scientific,
    topConfidence: row.top_confidence,
    predictions: JSON.parse(row.predictions),
    segments: JSON.parse(row.segments),
    audioUrl: row.audio_url,
  })
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const { id } = await params
  const deleted = deleteDetectionById(user.id, id)

  if (!deleted) {
    return NextResponse.json({ error: "NOT_FOUND" }, { status: 404 })
  }

  return NextResponse.json({ deleted: true })
}
