import { NextRequest, NextResponse } from "next/server"
import { findUserByIdFull, updateUserPassword } from "@/lib/db"
import { verifyToken } from "@/lib/auth"
import bcrypt from "bcryptjs"

export const dynamic = "force-dynamic"

export async function POST(req: NextRequest) {
  try {
    // Verify auth
    const token = req.cookies.get("birdsense-token")?.value
    if (!token) {
      return NextResponse.json({ error: "Not authenticated" }, { status: 401 })
    }

    const payload = verifyToken(token)
    if (!payload) {
      return NextResponse.json({ error: "Invalid token" }, { status: 401 })
    }

    const { currentPassword, newPassword } = await req.json()

    if (!currentPassword || !newPassword) {
      return NextResponse.json({ error: "Both currentPassword and newPassword are required" }, { status: 400 })
    }

    if (newPassword.length < 6) {
      return NextResponse.json({ error: "New password must be at least 6 characters" }, { status: 400 })
    }

    const user = findUserByIdFull(payload.userId)
    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 })
    }

    // Verify current password
    const valid = await bcrypt.compare(currentPassword, user.password)
    if (!valid) {
      return NextResponse.json({ error: "Current password is incorrect" }, { status: 403 })
    }

    // Hash new password and update
    const newHash = await bcrypt.hash(newPassword, 10)
    updateUserPassword(user.id, newHash)

    return NextResponse.json({ success: true, message: "Password updated successfully" })
  } catch (err) {
    console.error("[change-password]", err)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
