import { NextResponse } from "next/server"

const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000"

export const dynamic = "force-dynamic"

export async function GET() {
  try {
    const res = await fetch(`${FASTAPI_URL}/species`, { cache: "force-cache" })
    if (!res.ok) {
      return NextResponse.json({ error: "SPECIES_FETCH_FAILED" }, { status: 502 })
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json({ error: "FASTAPI_UNREACHABLE" }, { status: 502 })
  }
}
