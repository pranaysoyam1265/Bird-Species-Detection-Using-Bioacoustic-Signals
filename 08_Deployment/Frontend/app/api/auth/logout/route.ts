import { NextResponse } from "next/server"
import { clearAuthCookie } from "@/lib/auth"

export const dynamic = "force-dynamic"

export async function POST() {
  try {
    await clearAuthCookie()
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Logout error:", error)
    return NextResponse.json(
      { error: "INTERNAL_SERVER_ERROR" },
      { status: 500 }
    )
  }
}
