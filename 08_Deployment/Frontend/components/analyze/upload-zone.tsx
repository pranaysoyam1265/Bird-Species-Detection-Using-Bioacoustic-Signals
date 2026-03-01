"use client"

import { useRef, useState, useCallback } from "react"
import { Upload, Check, AlertCircle, Music } from "lucide-react"

interface UploadZoneProps {
  onFileSelect: (file: File) => void
  currentFile: File | null
  error?: string
}

const ACCEPTED = ".wav,.mp3,.flac,.ogg,.m4a"
const ACCEPTED_TYPES = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/x-m4a", "audio/mp4", "audio/x-wav", "audio/wave"]

export function UploadZone({ onFileSelect, currentFile, error }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = useCallback((file: File) => {
    if (file && (ACCEPTED_TYPES.some(t => file.type.includes(t.split("/")[1])) || file.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i))) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const onDragLeave = useCallback(() => setDragOver(false), [])
  const onClick = useCallback(() => inputRef.current?.click(), [])
  const onChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }, [handleFile])

  const hasFile = !!currentFile
  const hasError = !!error

  return (
    <div
      onClick={onClick}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`
        relative cursor-pointer transition-none
        border-2 p-8 sm:p-12
        flex flex-col items-center justify-center gap-4 text-center
        min-h-[200px] overflow-hidden
        ${hasError
          ? "border-red-500 bg-red-500/5"
          : hasFile
            ? "border-accent bg-accent/5"
            : dragOver
              ? "border-accent border-solid bg-accent/8"
              : "border-dashed border-foreground/30 hover:border-accent/50 bg-background"
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        onChange={onChange}
        className="hidden"
      />

      {/* Faint waveform SVG texture */}
      {!hasFile && !hasError && (
        <svg
          className="absolute inset-0 w-full h-full opacity-[0.04] pointer-events-none"
          viewBox="0 0 400 200"
          preserveAspectRatio="none"
        >
          <polyline
            points="0,100 20,60 40,130 60,40 80,150 100,70 120,110 140,30 160,160 180,80 200,100 220,50 240,140 260,70 280,120 300,45 320,155 340,85 360,105 380,55 400,100"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
        </svg>
      )}

      {hasError ? (
        <>
          <AlertCircle size={32} className="text-red-500" />
          <p className="font-mono text-xs tracking-[0.2em] uppercase text-red-500">{error}</p>
        </>
      ) : hasFile ? (
        <>
          <div className="flex items-center gap-3">
            <Check size={20} className="text-accent" />
            <Music size={20} className="text-accent" />
          </div>
          <p className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold">
            {currentFile.name}
          </p>
          <p className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground">
            {(currentFile.size / (1024 * 1024)).toFixed(2)} MB • CLICK TO CHANGE
          </p>
        </>
      ) : (
        <>
          {/* Pulsing ring upload icon */}
          <div className="animate-pulse-ring rounded-full p-3 bg-accent/10">
            <Upload size={32} className="text-accent" />
          </div>
          <p className="font-mono text-xs sm:text-sm tracking-[0.2em] uppercase text-foreground font-bold">
            DRAG &amp; DROP AUDIO FILE HERE
          </p>
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/70">
            or click to browse
          </p>
          <div className="flex flex-wrap justify-center gap-1.5 mt-2">
            {["WAV", "MP3", "FLAC", "OGG", "M4A"].map((fmt) => (
              <span
                key={fmt}
                className="font-mono text-[9px] tracking-[0.15em] border border-foreground/20 px-1.5 py-0.5 text-muted-foreground/60"
              >
                {fmt}
              </span>
            ))}
          </div>
        </>
      )}

      {/* Corner dots */}
      <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent" />
      <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent" />
      <span className="absolute -bottom-[3px] -left-[3px] w-2 h-2 bg-accent" />
      <span className="absolute -bottom-[3px] -right-[3px] w-2 h-2 bg-accent" />
    </div>
  )
}
