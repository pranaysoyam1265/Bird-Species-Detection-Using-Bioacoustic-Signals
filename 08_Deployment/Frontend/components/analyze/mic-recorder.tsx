"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { Mic, MicOff, Square, Loader2 } from "lucide-react"

interface MicRecorderProps {
  onRecordingComplete: (file: File) => void
}

export function MicRecorder({ onRecordingComplete }: MicRecorderProps) {
  const [state, setState] = useState<"idle" | "requesting" | "recording" | "processing" | "error">("idle")
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState("")
  const [levels, setLevels] = useState<number[]>([])
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number>(0)
  const startTimeRef = useRef(0)

  const cleanup = useCallback(() => {
    cancelAnimationFrame(rafRef.current)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    mediaRecorderRef.current = null
    analyserRef.current = null
  }, [])

  useEffect(() => () => cleanup(), [cleanup])

  const startRecording = async () => {
    setError("")
    setState("requesting")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Set up analyser for live levels
      const audioCtx = new AudioContext()
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 64
      source.connect(analyser)
      analyserRef.current = analyser

      // MediaRecorder
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" })
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        setState("processing")
        const blob = new Blob(chunksRef.current, { type: "audio/webm" })
        const file = new File([blob], `recording_${Date.now()}.webm`, { type: "audio/webm" })
        cleanup()
        onRecordingComplete(file)
        setState("idle")
        setElapsed(0)
        setLevels([])
      }

      recorder.start(100)
      startTimeRef.current = Date.now()
      setState("recording")

      // Animate levels
      const tick = () => {
        if (analyserRef.current) {
          const data = new Uint8Array(analyserRef.current.frequencyBinCount)
          analyserRef.current.getByteFrequencyData(data)
          const slice = Array.from(data.slice(0, 24)).map((v) => v / 255)
          setLevels(slice)
          setElapsed(Date.now() - startTimeRef.current)
        }
        rafRef.current = requestAnimationFrame(tick)
      }
      rafRef.current = requestAnimationFrame(tick)
    } catch (err) {
      cleanup()
      setState("error")
      setError("Microphone access denied. Please allow microphone permissions.")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop()
    }
  }

  const cancelRecording = () => {
    cleanup()
    setState("idle")
    setElapsed(0)
    setLevels([])
    chunksRef.current = []
  }

  const fmtTime = (ms: number) => {
    const s = Math.floor(ms / 1000)
    const m = Math.floor(s / 60)
    const sec = s % 60
    const centis = Math.floor((ms % 1000) / 10)
    return `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}.${centis.toString().padStart(2, "0")}`
  }

  return (
    <div className="border-2 border-foreground bg-background">
      <div className="border-b-2 border-foreground px-4 py-2 flex items-center gap-2">
        <Mic size={12} />
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          RECORD AUDIO
        </span>
      </div>
      <div className="p-6 flex flex-col items-center gap-4">
        {state === "idle" && (
          <>
            <Mic size={32} className="text-muted-foreground" />
            <button
              type="button"
              onClick={startRecording}
              className="px-6 py-3 border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.2em] uppercase font-bold cursor-pointer transition-none hover:bg-background hover:text-accent"
            >
              START RECORDING
            </button>
          </>
        )}

        {state === "requesting" && (
          <>
            <Loader2 size={32} className="animate-spin text-accent" />
            <p className="font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground">
              REQUESTING MICROPHONE...
            </p>
          </>
        )}

        {state === "recording" && (
          <>
            {/* Live waveform bars */}
            <div className="flex items-end gap-[2px] h-[60px]">
              {levels.map((l, i) => (
                <div
                  key={i}
                  className="w-[6px] bg-accent transition-all duration-75"
                  style={{ height: `${Math.max(l * 60, 3)}px` }}
                />
              ))}
            </div>

            {/* Timer */}
            <div className="text-center space-y-1">
              <div className="flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-red-500 animate-pulse" />
                <span className="font-mono text-lg font-bold tracking-wider text-foreground">
                  {fmtTime(elapsed)}
                </span>
              </div>
              <p className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">
                RECORDING...
              </p>
            </div>

            {/* Controls */}
            <div className="flex gap-3">
              <button
                type="button"
                onClick={stopRecording}
                className="flex items-center gap-2 px-5 py-2.5 border-2 border-accent bg-accent text-white font-mono text-xs tracking-[0.15em] uppercase font-bold cursor-pointer transition-none"
              >
                <Square size={12} /> STOP
              </button>
              <button
                type="button"
                onClick={cancelRecording}
                className="flex items-center gap-2 px-5 py-2.5 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase text-foreground cursor-pointer transition-none hover:bg-muted"
              >
                CANCEL
              </button>
            </div>
          </>
        )}

        {state === "processing" && (
          <>
            <Loader2 size={32} className="animate-spin text-accent" />
            <p className="font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground">
              PROCESSING RECORDING...
            </p>
          </>
        )}

        {state === "error" && (
          <>
            <MicOff size={32} className="text-red-500" />
            <p className="font-mono text-xs tracking-[0.15em] uppercase text-red-500 text-center">
              {error}
            </p>
            <button
              type="button"
              onClick={() => { setState("idle"); setError(""); }}
              className="px-4 py-2 border-2 border-foreground font-mono text-xs tracking-[0.15em] uppercase cursor-pointer hover:bg-muted"
            >
              TRY AGAIN
            </button>
          </>
        )}
      </div>
    </div>
  )
}
