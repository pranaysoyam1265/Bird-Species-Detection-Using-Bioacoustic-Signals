import { NextResponse } from "next/server"
import { signupSchema } from "@/lib/validations"
import { setAuthCookie } from "@/lib/auth"

export const dynamic = "force-dynamic"

export async function POST(request: Request) {
  try {
    const body = await request.json()

    // Validate input
    const result = signupSchema.safeParse(body)
    if (!result.success) {
      const errors = result.error.flatten().fieldErrors
      return NextResponse.json(
        { error: "VALIDATION_FAILED", details: errors },
        { status: 400 }
      )
    }

    const { email, password, name } = result.data

    // ── Call Backend Registration ──
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    const backendRes = await fetch(`${API_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name }),
    })

    if (!backendRes.ok) {
      const err = await backendRes.json().catch(() => ({ detail: "Registration failed" }))
      const status = backendRes.status === 400 ? 409 : backendRes.status
      return NextResponse.json(
        { error: err.detail || "REGISTRATION_FAILED" },
        { status }
      )
    }

    const { user } = await backendRes.json()

    // Set auth cookie locally
    await setAuthCookie(user)

    return NextResponse.json({ user }, { status: 201 })
  } catch (error) {
    console.error("Signup error:", error)
    return NextResponse.json(
      { error: "INTERNAL_SERVER_ERROR" },
      { status: 500 }
    )
  }
}
