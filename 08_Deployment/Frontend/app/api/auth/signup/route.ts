import { NextResponse } from "next/server"
import { signupSchema } from "@/lib/validations"
import { findUserByEmail, createUser } from "@/lib/db"
import { hashPassword, setAuthCookie } from "@/lib/auth"

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

    // Check if email already exists
    const existing = findUserByEmail(email)
    if (existing) {
      return NextResponse.json(
        { error: "EMAIL_ALREADY_EXISTS" },
        { status: 409 }
      )
    }

    // Hash password and create user
    const hashed = await hashPassword(password)
    const user = createUser(email, hashed, name)

    // Set auth cookie
    await setAuthCookie(user.id)

    return NextResponse.json({ user }, { status: 201 })
  } catch (error) {
    console.error("Signup error:", error)
    return NextResponse.json(
      { error: "INTERNAL_SERVER_ERROR" },
      { status: 500 }
    )
  }
}
