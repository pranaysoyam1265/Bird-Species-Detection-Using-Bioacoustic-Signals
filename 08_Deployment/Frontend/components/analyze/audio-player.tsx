"use client"

import { useRef, useState, useEffect, useCallback } from "react"
import { Play, Pause, X, Volume2, VolumeX, Repeat, Gauge } from "lucide-react"

interface AudioPlayerProps {
  file: File
  onRemove: () => void
  onAudioDecoded?: (buffer: AudioBuffer) => void
}

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2]

export function AudioPlayer({ file, onRemove, onAudioDecoded }: AudioPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const spectrogramRef = useRef<HTMLCanvasElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.8)
  const [muted, setMuted] = useState(false)
  const [waveform, setWaveform] = useState<number[]>([])
  const [url, setUrl] = useState("")
  const [loop, setLoop] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [showSpeedMenu, setShowSpeedMenu] = useState(false)
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [hoverTime, setHoverTime] = useState<string | null>(null)
  const [spectrogramData, setSpectrogramData] = useState<number[]>([])

  const rafRef = useRef<number>(0)
  const speedMenuRef = useRef<HTMLDivElement>(null)

  // Create object URL for audio
  useEffect(() => {
    const u = URL.createObjectURL(file)
    setUrl(u)
    setPlaying(false)
    setCurrent(0)
    return () => URL.revokeObjectURL(u)
  }, [file])

  // Decode audio and extract waveform + spectrogram data
  useEffect(() => {
    const reader = new FileReader()
    reader.onload = async () => {
      try {
        const ctx = new AudioContext()
        const buffer = await ctx.decodeAudioData(reader.result as ArrayBuffer)
        onAudioDecoded?.(buffer)
        setDuration(buffer.duration)

        // Extract waveform peaks
        const raw = buffer.getChannelData(0)
        const samples = 200
        const blockSize = Math.floor(raw.length / samples)
        const peaks: number[] = []
        for (let i = 0; i < samples; i++) {
          let sum = 0
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(raw[i * blockSize + j])
          }
          peaks.push(sum / blockSize)
        }
        const max = Math.max(...peaks, 0.01)
        setWaveform(peaks.map(p => p / max))

        // Extract mini spectrogram (RMS energy in frequency bands)
        const spectSamples = 200
        const spectBlockSize = Math.floor(raw.length / spectSamples)
        const spectData: number[] = []
        for (let i = 0; i < spectSamples; i++) {
          let energy = 0
          for (let j = 0; j < Math.min(spectBlockSize, 512); j++) {
            const val = raw[i * spectBlockSize + j] || 0
            energy += val * val
          }
          spectData.push(Math.sqrt(energy / Math.min(spectBlockSize, 512)))
        }
        const spectMax = Math.max(...spectData, 0.01)
        setSpectrogramData(spectData.map(v => v / spectMax))

        ctx.close()
      } catch {
        setWaveform(Array.from({ length: 200 }, () => Math.random() * 0.5 + 0.2))
        setSpectrogramData(Array.from({ length: 200 }, () => Math.random()))
      }
    }
    reader.readAsArrayBuffer(file)
  }, [file, onAudioDecoded])

  // Draw waveform with gradient
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || waveform.length === 0) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const W = rect.width
    const H = rect.height
    const barW = W / waveform.length
    const progress = duration > 0 ? currentTime / duration : 0

    ctx.clearRect(0, 0, W, H)

    // Get accent color from CSS variable
    const accentHex = getComputedStyle(document.documentElement).getPropertyValue("--accent-hex").trim() || "#ea580c"

    waveform.forEach((v, i) => {
      const x = i * barW
      const h = v * H * 0.85
      const y = (H - h) / 2
      const pct = i / waveform.length

      if (pct <= progress) {
        // Gradient for played portion
        const grad = ctx.createLinearGradient(x, y + h, x, y)
        grad.addColorStop(0, accentHex)
        grad.addColorStop(0.5, accentHex)
        grad.addColorStop(1, hexToRgba(accentHex, 0.35))
        ctx.fillStyle = grad
      } else {
        // Gradient for unplayed portion
        const grad = ctx.createLinearGradient(x, y + h, x, y)
        grad.addColorStop(0, "hsl(0 0% 40%)")
        grad.addColorStop(1, "hsl(0 0% 25%)")
        ctx.fillStyle = grad
      }

      ctx.fillRect(x, y, Math.max(barW - 1, 1), h)
    })

    // Hover preview line
    if (hoverX !== null) {
      ctx.strokeStyle = hexToRgba(accentHex, 0.6)
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.beginPath()
      ctx.moveTo(hoverX, 0)
      ctx.lineTo(hoverX, H)
      ctx.stroke()
      ctx.setLineDash([])
    }
  }, [waveform, currentTime, duration, hoverX])

  // Draw mini spectrogram
  useEffect(() => {
    const canvas = spectrogramRef.current
    if (!canvas || spectrogramData.length === 0) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    const W = rect.width
    const H = rect.height
    const barW = W / spectrogramData.length
    const progress = duration > 0 ? currentTime / duration : 0

    ctx.clearRect(0, 0, W, H)

    spectrogramData.forEach((v, i) => {
      const x = i * barW
      const pct = i / spectrogramData.length

      // Map energy to hue: low energy = dark/cool, high energy = warm/orange
      const intensity = v
      const hue = 21 // orange hue
      const lightness = 10 + intensity * 40
      const saturation = 40 + intensity * 50
      const alpha = pct <= progress ? 0.9 : 0.35

      ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`
      ctx.fillRect(x, 0, Math.max(barW, 1), H)
    })
  }, [spectrogramData, currentTime, duration])

  // Animation frame for time update
  useEffect(() => {
    const tick = () => {
      if (audioRef.current) {
        setCurrent(audioRef.current.currentTime)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    if (playing) {
      rafRef.current = requestAnimationFrame(tick)
    }
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Only handle if this player's container is visible / page is focused
      const target = e.target as HTMLElement
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return

      switch (e.code) {
        case "Space":
          e.preventDefault()
          togglePlay()
          break
        case "ArrowLeft":
          e.preventDefault()
          skipTime(-5)
          break
        case "ArrowRight":
          e.preventDefault()
          skipTime(5)
          break
        case "KeyM":
          e.preventDefault()
          toggleMute()
          break
        case "KeyL":
          e.preventDefault()
          setLoop(prev => !prev)
          break
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, duration])

  // Close speed menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(e.target as Node)) {
        setShowSpeedMenu(false)
      }
    }
    document.addEventListener("click", handler)
    return () => document.removeEventListener("click", handler)
  }, [])

  // Sync loop & speed to audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.loop = loop
    }
  }, [loop])

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = speed
    }
  }, [speed])

  const togglePlay = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (playing) {
      audio.pause()
    } else {
      audio.play()
    }
    setPlaying(!playing)
  }, [playing])

  const skipTime = useCallback((delta: number) => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = Math.max(0, Math.min(audio.currentTime + delta, duration))
    setCurrent(audio.currentTime)
  }, [duration])

  const toggleMute = useCallback(() => {
    setMuted(prev => {
      if (audioRef.current) audioRef.current.muted = !prev
      return !prev
    })
  }, [])

  const seek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value)
    if (audioRef.current) {
      audioRef.current.currentTime = t
      setCurrent(t)
    }
  }, [])

  const changeVolume = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value)
    setVolume(v)
    if (audioRef.current) audioRef.current.volume = v
    if (v > 0 && muted) {
      setMuted(false)
      if (audioRef.current) audioRef.current.muted = false
    }
  }, [muted])

  // Click-to-seek on waveform
  const handleWaveformClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const audio = audioRef.current
    if (!canvas || !audio || duration === 0) return
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const pct = x / rect.width
    const newTime = pct * duration
    audio.currentTime = newTime
    setCurrent(newTime)
  }, [duration])

  // Hover seek preview
  const handleWaveformHover = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || duration === 0) return
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const pct = x / rect.width
    const time = pct * duration
    setHoverX(x)
    setHoverTime(fmt(time))
  }, [duration])

  const handleWaveformLeave = useCallback(() => {
    setHoverX(null)
    setHoverTime(null)
  }, [])

  const fmt = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`
  }

  return (
    <div ref={containerRef} className="border-2 border-foreground bg-background">
      <audio
        ref={audioRef}
        src={url || undefined}
        onEnded={() => { if (!loop) setPlaying(false) }}
      />

      {/* Header */}
      <div className="flex items-center justify-between border-b-2 border-foreground px-4 py-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground truncate">
            {file.name}
          </span>
          {loop && (
            <span className="shrink-0 font-mono text-[8px] tracking-[0.15em] uppercase text-accent bg-accent/10 border border-accent/30 px-1.5 py-0.5">
              LOOP
            </span>
          )}
          {speed !== 1 && (
            <span className="shrink-0 font-mono text-[8px] tracking-[0.15em] uppercase text-accent bg-accent/10 border border-accent/30 px-1.5 py-0.5">
              {speed}×
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {/* Keyboard hint */}
          <span className="hidden md:inline font-mono text-[7px] tracking-[0.15em] uppercase text-muted-foreground/40">
            SPACE:PLAY · ←→:SKIP · M:MUTE · L:LOOP
          </span>
          <button
            type="button"
            onClick={onRemove}
            className="text-muted-foreground hover:text-red-500 cursor-pointer ml-2"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Waveform — clickable + hoverable */}
      <div className="px-4 py-4 relative">
        <canvas
          ref={canvasRef}
          className="w-full h-[80px] cursor-pointer"
          style={{ imageRendering: "pixelated" }}
          onClick={handleWaveformClick}
          onMouseMove={handleWaveformHover}
          onMouseLeave={handleWaveformLeave}
        />
        {/* Hover timestamp tooltip */}
        {hoverX !== null && hoverTime !== null && (
          <div
            className="absolute top-1 pointer-events-none"
            style={{ left: `calc(1rem + ${hoverX}px)`, transform: "translateX(-50%)" }}
          >
            <div className="bg-foreground text-background font-mono text-[9px] tracking-wider px-2 py-0.5">
              {hoverTime}
            </div>
          </div>
        )}
      </div>

      {/* Mini Spectrogram Bar */}
      <div className="px-4 pb-2">
        <canvas
          ref={spectrogramRef}
          className="w-full h-3 border border-foreground/10"
          style={{ imageRendering: "pixelated" }}
        />
      </div>

      {/* Controls */}
      <div className="border-t-2 border-foreground px-4 py-3 flex items-center gap-3">
        {/* Play/Pause */}
        <button
          type="button"
          onClick={togglePlay}
          className="w-8 h-8 border-2 border-foreground flex items-center justify-center cursor-pointer hover:bg-accent hover:border-accent hover:text-white transition-none"
          title="Play/Pause (Space)"
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
        </button>

        {/* Loop toggle */}
        <button
          type="button"
          onClick={() => setLoop(!loop)}
          className={`w-8 h-8 border-2 flex items-center justify-center cursor-pointer transition-none ${loop
            ? "border-accent bg-accent/10 text-accent"
            : "border-foreground/30 text-muted-foreground hover:border-foreground hover:text-foreground"
            }`}
          title="Toggle Loop (L)"
        >
          <Repeat size={13} />
        </button>

        {/* Speed selector */}
        <div ref={speedMenuRef} className="relative">
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setShowSpeedMenu(!showSpeedMenu) }}
            className={`h-8 px-2.5 border-2 flex items-center gap-1.5 cursor-pointer transition-none font-mono text-[10px] tracking-wider ${speed !== 1
              ? "border-accent bg-accent/10 text-accent"
              : "border-foreground/30 text-muted-foreground hover:border-foreground hover:text-foreground"
              }`}
            title="Playback Speed"
          >
            <Gauge size={12} />
            <span className="font-bold">{speed}×</span>
          </button>

          {/* Speed dropdown */}
          {showSpeedMenu && (
            <div className="absolute bottom-full left-0 mb-1 border-2 border-foreground bg-background shadow-xl z-50 w-20">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => { setSpeed(s); setShowSpeedMenu(false) }}
                  className={`w-full px-3 py-1.5 font-mono text-[10px] tracking-wider text-left cursor-pointer transition-none ${s === speed
                    ? "bg-accent text-white font-bold"
                    : "text-foreground hover:bg-accent/10 hover:text-accent"
                    }`}
                >
                  {s}×
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Seek slider */}
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={currentTime}
          onChange={seek}
          className="flex-1 h-1 appearance-none bg-foreground/20 outline-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:border-0"
        />

        {/* Time display */}
        <span className="font-mono text-[10px] tracking-wider text-muted-foreground whitespace-nowrap">
          {fmt(currentTime)} / {fmt(duration)}
        </span>

        {/* Volume */}
        <div className="hidden sm:flex items-center gap-2">
          <button
            type="button"
            onClick={toggleMute}
            className="text-muted-foreground hover:text-foreground cursor-pointer"
            title="Mute (M)"
          >
            {muted || volume === 0 ? <VolumeX size={13} /> : <Volume2 size={13} />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={muted ? 0 : volume}
            onChange={changeVolume}
            className="w-16 h-1 appearance-none bg-foreground/20 outline-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2 [&::-webkit-slider-thumb]:h-2 [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:border-0"
          />
        </div>
      </div>
    </div>
  )
}

/* ─── Utility ─── */
function hexToRgba(hex: string, alpha: number): string {
  hex = hex.replace("#", "")
  const r = parseInt(hex.substring(0, 2), 16)
  const g = parseInt(hex.substring(2, 4), 16)
  const b = parseInt(hex.substring(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
