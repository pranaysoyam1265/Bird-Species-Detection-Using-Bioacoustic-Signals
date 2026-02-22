"use client"

import { useEffect, useState } from "react"

const SPECIES = [
  { name: "NORTHERN CARDINAL", status: "DETECTED", confidence: "96.1%" },
  { name: "BLUE JAY", status: "DETECTED", confidence: "94.8%" },
  { name: "AMERICAN ROBIN", status: "DETECTED", confidence: "92.3%" },
  { name: "BLACK-CAPPED CHICKADEE", status: "DETECTED", confidence: "89.7%" },
]

export function StatusCard() {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setTick((t) => t + 1)
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between border-b-2 border-foreground px-4 py-2">
        <span className="text-[10px] tracking-widest text-muted-foreground uppercase">
          species.detections
        </span>
        <span className="text-[10px] tracking-widest text-muted-foreground">
          {`SCAN:${String(tick).padStart(4, "0")}`}
        </span>
      </div>
      <div className="flex-1 flex flex-col p-4 gap-0">
        {/* Table header */}
        <div className="grid grid-cols-3 gap-2 border-b border-border pb-2 mb-2">
          <span className="text-[9px] tracking-[0.15em] uppercase text-muted-foreground">Species</span>
          <span className="text-[9px] tracking-[0.15em] uppercase text-muted-foreground">Status</span>
          <span className="text-[9px] tracking-[0.15em] uppercase text-muted-foreground text-right">Confidence</span>
        </div>
        {SPECIES.map((species) => (
          <div
            key={species.name}
            className="grid grid-cols-3 gap-2 py-2 border-b border-border last:border-none"
          >
            <span className="text-xs font-mono text-foreground">{species.name}</span>
            <div className="flex items-center gap-2">
              <span
                className="h-1.5 w-1.5"
                style={{
                  backgroundColor: species.status === "DETECTED" ? "#ea580c" : "hsl(var(--muted-foreground))",
                }}
              />
              <span className="text-xs font-mono text-muted-foreground">{species.status}</span>
            </div>
            <span className="text-xs font-mono text-foreground text-right">{species.confidence}</span>
          </div>
        ))}
        {/* Accuracy bar */}
        <div className="mt-auto pt-4">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
              Model Accuracy
            </span>
            <span className="text-[9px] font-mono text-foreground">96%</span>
          </div>
          <div className="h-2 w-full border border-foreground">
            <div className="h-full bg-foreground" style={{ width: "96%" }} />
          </div>
        </div>
      </div>
    </div>
  )
}
