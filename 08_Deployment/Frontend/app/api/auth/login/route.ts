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

    // Find user
    const user = findUserByEmail(email)
    if (!user) {
      return NextResponse.json(
        { error: "INVALID_CREDENTIALS" },
        { status: 401 }
      )
    }

    // Compare password
    const valid = await comparePassword(password, user.password)
    if (!valid) {
      return NextResponse.json(
        { error: "INVALID_CREDENTIALS" },
        { status: 401 }
      )
    }

    // Set auth cookie
    await setAuthCookie(user.id)

    // Return user without password
    const { password: _, ...safeUser } = user
    return NextResponse.json({ user: safeUser })
  } catch (error) {
    console.error("Login error:", error)
    return NextResponse.json(
      { error: "INTERNAL_SERVER_ERROR" },
      { status: 500 }
    )
  }
}
