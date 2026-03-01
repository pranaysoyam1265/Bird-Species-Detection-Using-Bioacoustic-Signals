import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { getApiKeysByUser, insertApiKey, deleteApiKeyById } from "@/lib/db"
import { randomBytes } from "crypto"
import bcrypt from "bcryptjs"

export const dynamic = "force-dynamic"

export async function GET() {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const keys = getApiKeysByUser(user.id)
  return NextResponse.json({ api_keys: keys })
}

export async function POST(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const body = await req.json().catch(() => null)
  if (!body || !body.name || typeof body.name !== "string") {
    return NextResponse.json({ error: "Key name is required" }, { status: 400 })
  }

  const id = crypto.randomUUID()
  const rawKey = `bsk_${randomBytes(24).toString("hex")}`

  // Hash the key before storing — the raw key is only ever returned once
  const keyHash = await bcrypt.hash(rawKey, 10)
  insertApiKey(user.id, id, body.name, keyHash)

  return NextResponse.json({
    id,
    name: body.name,
    key: rawKey, // Only sent back once — store it safely!
    created_at: new Date().toISOString()
  })
}

export async function DELETE(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const { searchParams } = new URL(req.url)
  const id = searchParams.get("id")

  if (!id) {
    return NextResponse.json({ error: "Missing API Key ID" }, { status: 400 })
  }

  const success = deleteApiKeyById(user.id, id)
  if (!success) {
    return NextResponse.json({ error: "API Key not found or belongs to another user" }, { status: 404 })
  }

  return NextResponse.json({ success: true })
}
