import { NextResponse } from "next/server"

const FASTAPI_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000"

export const dynamic = "force-dynamic"

export async function GET() {
  try {
    const res = await fetch(`${FASTAPI_URL}/health`, { next: { revalidate: 60 } })
    if (!res.ok) throw new Error("Backend unavailable")
    const data = await res.json()

    return NextResponse.json({
      status: data.status,
      numSpecies: data.num_species,
      device: data.device,
      modelPath: data.model_path
    })
  } catch (err) {
    return NextResponse.json({ status: "offline", numSpecies: 0, device: "N/A" })
  }
}
