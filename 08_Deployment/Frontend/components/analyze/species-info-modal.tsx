"use client"

import { X, ExternalLink } from "lucide-react"

interface SpeciesData {
  common: string
  scientific: string
  family: string
  habitat: string
  call: string
  trainingSamples: number
  validationAccuracy: number
}

// Mock species database
const SPECIES_DB: Record<string, SpeciesData> = {
  "Northern Cardinal": {
    common: "Northern Cardinal", scientific: "Cardinalis cardinalis",
    family: "Cardinalidae", habitat: "Woodlands, gardens, shrublands",
    call: 'Loud, clear whistles "cheer-cheer-cheer"',
    trainingSamples: 2847, validationAccuracy: 97.2,
  },
  "Pyrrhuloxia": {
    common: "Pyrrhuloxia", scientific: "Cardinalis sinuatus",
    family: "Cardinalidae", habitat: "Desert scrub, mesquite thickets",
    call: "Sharp, metallic chip notes and whistled phrases",
    trainingSamples: 1234, validationAccuracy: 93.1,
  },
  "Summer Tanager": {
    common: "Summer Tanager", scientific: "Piranga rubra",
    family: "Cardinalidae", habitat: "Open woodlands, forest edges",
    call: 'Robin-like phrases "pik-i-tuk-i-tuk"',
    trainingSamples: 1890, validationAccuracy: 94.8,
  },
  "Scarlet Tanager": {
    common: "Scarlet Tanager", scientific: "Piranga olivacea",
    family: "Cardinalidae", habitat: "Deciduous forests, canopy",
    call: 'Hoarse phrases like "robin with sore throat"',
    trainingSamples: 1567, validationAccuracy: 92.5,
  },
  "House Finch": {
    common: "House Finch", scientific: "Haemorhous mexicanus",
    family: "Fringillidae", habitat: "Urban areas, parks, suburbs",
    call: "Long, warbling song ending in a buzzy note",
    trainingSamples: 3210, validationAccuracy: 96.4,
  },
}

function getDefault(species: string): SpeciesData {
  return {
    common: species, scientific: "Species sp.",
    family: "Unknown", habitat: "Various habitats",
    call: "Species-specific vocalizations",
    trainingSamples: Math.floor(Math.random() * 2000 + 500),
    validationAccuracy: parseFloat((Math.random() * 10 + 88).toFixed(1)),
  }
}

interface SpeciesInfoModalProps {
  species: string
  onClose: () => void
}

export function SpeciesInfoModal({ species, onClose }: SpeciesInfoModalProps) {
  const data = SPECIES_DB[species] || getDefault(species)
  const slug = species.replace(/ /g, "_")
  const encoded = encodeURIComponent(species)

  const links = [
    { label: "eBird", url: `https://ebird.org/species/${slug.toLowerCase()}` },
    { label: "Xeno-Canto", url: `https://xeno-canto.org/explore?query=${encoded}` },
    { label: "Wikipedia", url: `https://en.wikipedia.org/wiki/${slug}` },
    { label: "AllAboutBirds", url: `https://www.allaboutbirds.org/guide/${slug}` },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60" />
      <div
        className="relative border-2 border-foreground bg-background w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b-2 border-foreground px-4 py-3">
          <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground">
            SPECIES INFO
          </span>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Name */}
          <div className="text-center">
            <span className="text-3xl block mb-2">🐦</span>
            <h3 className="font-mono text-lg font-bold tracking-[0.1em] uppercase text-foreground">
              {data.common}
            </h3>
            <p className="font-mono text-xs tracking-[0.15em] text-muted-foreground italic">
              {data.scientific}
            </p>
          </div>

          {/* Details */}
          <div className="space-y-3 border-t-2 border-foreground pt-4">
            <div className="flex gap-2">
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground w-[80px] shrink-0">
                FAMILY
              </span>
              <span className="font-mono text-xs text-foreground">{data.family}</span>
            </div>
            <div className="flex gap-2">
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground w-[80px] shrink-0">
                HABITAT
              </span>
              <span className="font-mono text-xs text-foreground">{data.habitat}</span>
            </div>
            <div className="flex gap-2">
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground w-[80px] shrink-0">
                CALL
              </span>
              <span className="font-mono text-xs text-foreground">{data.call}</span>
            </div>
          </div>

          {/* Model stats */}
          <div className="border-t-2 border-foreground pt-4 space-y-2">
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground block">
              MODEL STATS
            </span>
            <div className="grid grid-cols-2 gap-3">
              <div className="border border-foreground/30 p-3">
                <span className="font-mono text-[10px] tracking-wider uppercase text-muted-foreground block">
                  TRAINING SAMPLES
                </span>
                <span className="font-mono text-sm font-bold text-accent">
                  {data.trainingSamples.toLocaleString()}
                </span>
              </div>
              <div className="border border-foreground/30 p-3">
                <span className="font-mono text-[10px] tracking-wider uppercase text-muted-foreground block">
                  VAL ACCURACY
                </span>
                <span className="font-mono text-sm font-bold text-green-500">
                  {data.validationAccuracy}%
                </span>
              </div>
            </div>
          </div>

          {/* External links */}
          <div className="flex flex-wrap gap-2 border-t-2 border-foreground pt-4">
            {links.map((link) => (
              <a
                key={link.label}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase text-foreground hover:bg-foreground hover:text-background cursor-pointer transition-none"
              >
                <ExternalLink size={10} />
                {link.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
