import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"

export const dynamic = "force-dynamic"

export async function POST(req: NextRequest) {
  try {
    const user = await getSession()
    if (!user) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 })
    }

    const { currentPassword, newPassword } = await req.json()

    if (!currentPassword || !newPassword) {
      return NextResponse.json({ error: "Both currentPassword and newPassword are required" }, { status: 400 })
    }

    if (newPassword.length < 6) {
      return NextResponse.json({ error: "New password must be at least 6 characters" }, { status: 400 })
    }

    // ── Call Backend to change password ──
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    const backendRes = await fetch(`${API_URL}/auth/change-password`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: user.id,
        current_password: currentPassword,
        new_password: newPassword,
      }),
    })

    if (!backendRes.ok) {
      const err = await backendRes.json().catch(() => ({ detail: "Password change failed" }))
      return NextResponse.json({ error: err.detail || "PASSWORD_CHANGE_FAILED" }, { status: backendRes.status })
    }

    return NextResponse.json({ success: true, message: "Password updated successfully" })
  } catch (err) {
    console.error("[change-password]", err)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
