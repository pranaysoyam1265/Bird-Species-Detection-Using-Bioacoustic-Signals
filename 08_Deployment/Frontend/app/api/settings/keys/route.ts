import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"

export const dynamic = "force-dynamic"

export async function GET(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/settings/${user.id}/keys`)

  if (!backendRes.ok) {
    return NextResponse.json({ error: "FAILED_TO_FETCH_KEYS" }, { status: 500 })
  }

  const result = await backendRes.json()
  return NextResponse.json(result)
}

export async function POST(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const body = await req.json().catch(() => null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/settings/${user.id}/keys`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  })

  if (!backendRes.ok) {
    return NextResponse.json({ error: "FAILED_TO_CREATE_KEY" }, { status: 500 })
  }

  const result = await backendRes.json()
  return NextResponse.json(result, { status: 201 })
}

export async function DELETE(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const { searchParams } = new URL(req.url)
  const keyId = searchParams.get("id")
  if (!keyId) {
    return NextResponse.json({ error: "MISSING_KEY_ID" }, { status: 400 })
  }

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/settings/${user.id}/keys/${keyId}`, {
    method: "DELETE",
  })

  if (!backendRes.ok) {
    return NextResponse.json({ error: "FAILED_TO_DELETE_KEY" }, { status: 500 })
  }

  return NextResponse.json({ deleted: true })
}
