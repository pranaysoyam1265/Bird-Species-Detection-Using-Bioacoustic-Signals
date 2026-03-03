import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"

const FASTAPI_URL = process.env.FASTAPI_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

/**
 * Proxy route for bird detection.
 * Forwards audio to the FastAPI backend for ML inference.
 * The backend handles database persistence.
 */
export const dynamic = "force-dynamic"

export async function POST(req: NextRequest) {
  const user = await getSession()

  if (!user) {
    return NextResponse.json({ error: "UNAUTHORIZED: Missing or invalid session" }, { status: 401 })
  }

  try {
    // ── Read incoming multipart form ──
    const incoming = await req.formData()
    const audioEntry = incoming.get("audio_file")

    if (!audioEntry || typeof audioEntry === "string") {
      return NextResponse.json({ error: "No audio file provided" }, { status: 400 })
    }

    const fileName = audioEntry.name || "upload.wav"
    const fileType = audioEntry.type || "audio/wav"

    // ── Build a fresh multipart body ──
    const fileBytes = Buffer.from(await audioEntry.arrayBuffer())
    const blob = new Blob([fileBytes], { type: fileType })
    const outgoing = new FormData()
    outgoing.append("audio_file", blob, fileName)
    outgoing.append("user_id", user.id.toString())

    // Forward optional params
    const topK = incoming.get("top_k")
    const confThresh = incoming.get("confidence_threshold")
    const noiseRed = incoming.get("noise_reduction")
    const chunkDur = incoming.get("chunk_duration")
    if (topK) outgoing.append("top_k", topK as string)
    if (confThresh) outgoing.append("confidence_threshold", confThresh as string)
    if (noiseRed) outgoing.append("noise_reduction", noiseRed as string)
    if (chunkDur) outgoing.append("chunk_duration", chunkDur as string)

    // ── Forward to FastAPI ──
    const fastApiRes = await fetch(`${FASTAPI_URL}/detect`, {
      method: "POST",
      body: outgoing,
    })

    if (!fastApiRes.ok) {
      const err = await fastApiRes.json().catch(() => ({ detail: "Inference failed" }))
      console.error("[/api/detect] FastAPI error:", fastApiRes.status, err)
      return NextResponse.json(
        { error: err.detail || "DETECTION_FAILED" },
        { status: fastApiRes.status },
      )
    }

    const result = await fastApiRes.json()
    return NextResponse.json(result)
  } catch (err) {
    console.error("[/api/detect] Error:", err)
    return NextResponse.json(
      { error: "DETECTION_FAILED", detail: String(err) },
      { status: 500 },
    )
  }
}
