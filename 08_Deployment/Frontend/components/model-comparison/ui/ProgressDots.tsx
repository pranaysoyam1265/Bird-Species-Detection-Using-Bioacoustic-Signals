"use client"

export function ProgressDots({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center justify-center gap-2 py-3">
      {Array.from({ length: total }).map((_, i) => (
        <span
          key={i}
          className={`w-2 h-2 transition-none ${i === current ? "bg-accent" : "bg-foreground/20"
            }`}
        />
      ))}
    </div>
  )
}
