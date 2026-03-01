import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { getUserSettings, getDb } from "@/lib/db"

export const dynamic = "force-dynamic"

export async function POST(req: NextRequest) {
  const user = await getSession()
  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED" }, { status: 401 })
  }

  const settings = getUserSettings(user.id)
  const autoClearDays = Number(settings.autoClear) || 0

  if (autoClearDays <= 0) {
    return NextResponse.json({ message: "Auto-clear is disabled for this user", deletedCount: 0 })
  }

  try {
    const db = getDb() // Warning: assuming getDb is exported or we can just run a query

    // SQLite DATE('now', '-X days')
    const result = db.prepare(`
      DELETE FROM detections 
      WHERE user_id = ? AND date < DATE('now', '-' || ? || ' days')
    `).run(user.id, autoClearDays)

    return NextResponse.json({ success: true, deletedCount: result.changes })
  } catch (err) {
    console.error("[Cleanup API Error]", err)
    return NextResponse.json({ error: "Failed to run cleanup" }, { status: 500 })
  }
}
