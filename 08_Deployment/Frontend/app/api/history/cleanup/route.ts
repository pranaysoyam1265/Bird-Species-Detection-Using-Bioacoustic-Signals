import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"

export const dynamic = "force-dynamic"

export async function POST(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  // ── Call Backend Cleanup ──
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/history/cleanup?user_id=${user.id}`, {
    method: "POST",
  })

  if (!backendRes.ok) {
    return NextResponse.json({ error: "Failed to run cleanup" }, { status: 500 })
  }

  const result = await backendRes.json()
  return NextResponse.json(result)
}
