import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"

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
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/history/${id}?user_id=${user.id}`)

  if (!backendRes.ok) {
    return NextResponse.json({ error: "NOT_FOUND" }, { status: 404 })
  }

  const detection = await backendRes.json()
  return NextResponse.json(detection)
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
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  const backendRes = await fetch(`${API_URL}/history/${id}?user_id=${user.id}`, {
    method: "DELETE",
  })

  if (!backendRes.ok) {
    return NextResponse.json({ error: "NOT_FOUND" }, { status: 404 })
  }

  return NextResponse.json({ deleted: true })
}
