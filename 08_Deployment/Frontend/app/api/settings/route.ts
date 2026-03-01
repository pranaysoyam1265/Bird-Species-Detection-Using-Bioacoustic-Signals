import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { getUserSettings, upsertUserSettings } from "@/lib/db"

export const dynamic = "force-dynamic"

export async function GET(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const { searchParams } = new URL(req.url)

  // ── Call Backend Settings ──
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/settings/${user.id}`)

  if (!backendRes.ok) {
    return NextResponse.json({ error: "FAILED_TO_FETCH_SETTINGS" }, { status: 500 })
  }

  const { settings } = await backendRes.json()
  return NextResponse.json({ settings })
}

export async function PUT(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const body = await req.json().catch(() => null)
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "INVALID_BODY" }, { status: 400 })
  }

  // ── Call Backend Settings ──
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/settings/${user.id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })

  if (!backendRes.ok) {
    return NextResponse.json({ error: "FAILED_TO_UPDATE_SETTINGS" }, { status: 500 })
  }

  return NextResponse.json({ settings: body })
}
