import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { getUserSettings, upsertUserSettings } from "@/lib/db"

export const dynamic = "force-dynamic"

export async function GET() {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const settings = getUserSettings(user.id)
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

  upsertUserSettings(user.id, body)
  return NextResponse.json({ settings: body })
}
