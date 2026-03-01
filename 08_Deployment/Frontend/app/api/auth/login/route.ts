import { NextResponse } from "next/server"
import { loginSchema } from "@/lib/validations"
import { findUserByEmail } from "@/lib/db"
import { comparePassword, setAuthCookie } from "@/lib/auth"

export const dynamic = "force-dynamic"

export async function POST(request: Request) {
  try {
    const body = await request.json()

    // Validate input
    const result = loginSchema.safeParse(body)
    if (!result.success) {
      const errors = result.error.flatten().fieldErrors
      return NextResponse.json(
        { error: "VALIDATION_FAILED", details: errors },
        { status: 400 }
      )
    }

    const { email, password } = result.data

    // ── Call Backend Auth ──
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    const backendRes = await fetch(`${API_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })

    if (!backendRes.ok) {
      return NextResponse.json(
        { error: "INVALID_CREDENTIALS" },
        { status: 401 }
      )
    }

    const { user } = await backendRes.json()

    // ── Set auth cookie locally ──
    await setAuthCookie(user)

    return NextResponse.json({ user })
  } catch (error) {
    console.error("Login error:", error)
    return NextResponse.json(
      { error: "INTERNAL_SERVER_ERROR" },
      { status: 500 }
    )
  }
}
