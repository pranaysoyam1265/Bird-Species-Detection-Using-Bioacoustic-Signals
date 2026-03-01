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

  // ── Call Backend History ──
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/history?user_id=${user.id}&limit=${limit}&offset=${offset}`)

  if (!backendRes.ok) {
    return NextResponse.json({ error: "FAILED_TO_FETCH_HISTORY" }, { status: 500 })
  }

  const { detections, total } = await backendRes.json()

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
