import { NextRequest, NextResponse } from "next/server"
import { getSession } from "@/lib/auth"
import { insertDetection, getAllApiKeys, getUserSettings, findUserById } from "@/lib/db"
import { randomUUID } from "crypto"
import bcrypt from "bcryptjs"

const FASTAPI_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000"

export const dynamic = "force-dynamic"

export async function POST(req: NextRequest) {
  let user = await getSession()

  // ── Auth check ──
  if (!user) {
    // Check for API Key if no session cookie
    const apiKeyHeader = req.headers.get("x-api-key") || req.headers.get("authorization")?.replace("Bearer ", "")
    if (apiKeyHeader) {
      const allKeys = getAllApiKeys()
      for (const keyRow of allKeys) {
        const matches = await bcrypt.compare(apiKeyHeader, keyRow.key_hash)
        if (matches) {
          const dbUser = findUserById(keyRow.user_id)
          if (dbUser) { user = dbUser; break }
        }
      }
    }

    if (!user) {
      return NextResponse.json({ error: "UNAUTHORIZED: Missing or invalid API key/session" }, { status: 401 })
    }
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

    // ── Build a fresh multipart body manually ──
    const fileBytes = Buffer.from(await audioEntry.arrayBuffer())
    const blob = new Blob([fileBytes], { type: fileType })
    const outgoing = new FormData()
    outgoing.append("audio_file", blob, fileName)

    // Forward optional params
    const topK = incoming.get("top_k")
    const confThresh = incoming.get("confidence_threshold")
    const noiseRed = incoming.get("noise_reduction")
    if (topK) outgoing.append("top_k", topK as string)
    if (confThresh) outgoing.append("confidence_threshold", confThresh as string)
    if (noiseRed) outgoing.append("noise_reduction", noiseRed as string)

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

    // ── Save to database ──
    const now = new Date()

    const detectionRecord = {
      id: randomUUID(),
      user_id: user.id,
      filename: fileName,
      date: now.toISOString().split("T")[0],
      time: now.toTimeString().split(" ")[0],
      duration: Math.round(result.duration),
      top_species: result.top_species,
      top_scientific: result.top_scientific,
      top_confidence: result.top_confidence,
      predictions: JSON.stringify(result.predictions),
      segments: JSON.stringify(result.segments),
      audio_url: null,
    }

    insertDetection(detectionRecord)

    // ── Check Webhooks ──
    const settings = getUserSettings(user.id)
    if (settings && typeof settings.webhookUrl === "string" && settings.webhookUrl.startsWith("http")) {
      // Fire and forget — we don't await this to avoid blocking the response
      fetch(settings.webhookUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event: "detection.completed",
          timestamp: new Date().toISOString(),
          data: {
            ...detectionRecord,
            predictions: result.predictions,
            segments: result.segments
          }
        })
      }).catch(err => console.error("[Webhook Error]:", err))
    }

    return NextResponse.json(result)
  } catch (err) {
    console.error("[/api/detect] Error:", err)
    return NextResponse.json(
      { error: "DETECTION_FAILED", detail: String(err) },
      { status: 500 },
    )
  }
}
