"use client"

import { ExternalLink } from "lucide-react"

interface ExternalLinksProps {
  species: string
}

const LINKS = [
  { label: "eBird", icon: "🦅", url: (s: string) => `https://ebird.org/species/${s.replace(/ /g, "_").toLowerCase()}` },
  { label: "Xeno-Canto", icon: "🎵", url: (s: string) => `https://xeno-canto.org/explore?query=${encodeURIComponent(s)}` },
  { label: "Wikipedia", icon: "📖", url: (s: string) => `https://en.wikipedia.org/wiki/${s.replace(/ /g, "_")}` },
  { label: "AllAboutBirds", icon: "🐦", url: (s: string) => `https://www.allaboutbirds.org/guide/${s.replace(/ /g, "_")}` },
]

export function ExternalLinks({ species }: ExternalLinksProps) {
  return (
    <div className="border-2 border-foreground bg-background">
      <div className="border-b-2 border-foreground px-4 py-2">
        <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground">
          LEARN MORE
        </span>
      </div>
      <div className="p-3 flex flex-wrap gap-2">
        {LINKS.map((link) => (
          <a
            key={link.label}
            href={link.url(species)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-2 border border-foreground font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground hover:border-accent cursor-pointer transition-none"
          >
            <span>{link.icon}</span>
            {link.label}
            <ExternalLink size={9} />
          </a>
        ))}
      </div>
    </div>
  )
}
