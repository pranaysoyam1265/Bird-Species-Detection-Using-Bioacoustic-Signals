import { NextResponse } from "next/server"
import { getSession } from "@/lib/auth"

export const dynamic = "force-dynamic"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function GET(request: Request) {
  try {
    const user = await getSession()
    if (!user || user.role !== "admin") {
      return NextResponse.json({ error: "ADMIN_REQUIRED" }, { status: 403 })
    }

    const { searchParams } = new URL(request.url)
    const endpoint = searchParams.get("endpoint") || "stats"

    const backendUrl = `${API_URL}/admin/${endpoint}?user_id=${user.id}`
    const res = await fetch(backendUrl)
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Backend error" }))
      return NextResponse.json({ error: err.detail || "ADMIN_ERROR" }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Admin API error:", error)
    return NextResponse.json({ error: "INTERNAL_SERVER_ERROR" }, { status: 500 })
  }
}

export async function PUT(request: Request) {
  try {
    const user = await getSession()
    if (!user || user.role !== "admin") {
      return NextResponse.json({ error: "ADMIN_REQUIRED" }, { status: 403 })
    }

    const body = await request.json()
    const { targetId, role } = body

    const backendUrl = `${API_URL}/admin/users/${targetId}/role?user_id=${user.id}`
    const res = await fetch(backendUrl, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role }),
    })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Backend error" }))
      return NextResponse.json({ error: err.detail || "ROLE_UPDATE_FAILED" }, { status: res.status })
    }

    return NextResponse.json(await res.json())
  } catch (error) {
    console.error("Admin role update error:", error)
    return NextResponse.json({ error: "INTERNAL_SERVER_ERROR" }, { status: 500 })
  }
}

export async function DELETE(request: Request) {
  try {
    const user = await getSession()
    if (!user || user.role !== "admin") {
      return NextResponse.json({ error: "ADMIN_REQUIRED" }, { status: 403 })
    }

    const { searchParams } = new URL(request.url)
    const targetId = searchParams.get("targetId")
    if (!targetId) {
      return NextResponse.json({ error: "MISSING_TARGET_ID" }, { status: 400 })
    }

    const backendUrl = `${API_URL}/admin/users/${targetId}?user_id=${user.id}`
    const res = await fetch(backendUrl, { method: "DELETE" })

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Backend error" }))
      return NextResponse.json({ error: err.detail || "DELETE_FAILED" }, { status: res.status })
    }

    return NextResponse.json(await res.json())
  } catch (error) {
    console.error("Admin delete error:", error)
    return NextResponse.json({ error: "INTERNAL_SERVER_ERROR" }, { status: 500 })
  }
}
