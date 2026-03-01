"use client"

import { useState, useEffect, useRef } from "react"
import { X, MapPin, Globe, Mountain, Mic, Loader2 } from "lucide-react"

/* ── Types ── */
interface SightingLocation {
  lat: number
  lng: number
  location: string
  country: string
  elevation: string
  recordist: string
  url: string
  quality: string
}

interface SpeciesMapModalProps {
  species: string
  scientificName?: string
  onClose: () => void
}

export function SpeciesMapModal({ species, scientificName, onClose }: SpeciesMapModalProps) {
  const [loading, setLoading] = useState(true)
  const [locations, setLocations] = useState<SightingLocation[]>([])
  const [totalRecordings, setTotalRecordings] = useState(0)
  const [selected, setSelected] = useState<SightingLocation | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const mapRef = useRef<HTMLDivElement>(null)
  const leafletMap = useRef<any>(null)
  const markersRef = useRef<any[]>([])

  // ── Fetch locations from Xeno-Canto via our API proxy ──
  useEffect(() => {
    const searchName = scientificName || species
    fetch(`/api/xeno-canto?species=${encodeURIComponent(searchName)}`)
      .then((r) => r.json())
      .then((data) => {
        setLocations(data.locations || [])
        setTotalRecordings(data.totalRecordings || 0)
        if (data.locations?.length > 0) {
          setSelected(data.locations[0])
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [species, scientificName])

  // ── Initialize Leaflet map once locations arrive ──
  useEffect(() => {
    if (loading || locations.length === 0 || !mapRef.current) return
    if (leafletMap.current) return // already initialized

    // Dynamically import Leaflet
    const initMap = async () => {
      const L = (await import("leaflet")).default

      // Fix default marker icons for webpack/Turbopack
      delete (L.Icon.Default.prototype as any)._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      })

      const center = locations[0]
      const map = L.map(mapRef.current!, {
        center: [center.lat, center.lng],
        zoom: 4,
        zoomControl: true,
        attributionControl: false,
      })

      // Dark tile layer for aesthetic
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
      }).addTo(map)

      // Custom green marker icon
      const greenIcon = L.divIcon({
        className: "custom-marker",
        html: `<div style="width:12px;height:12px;background:#22c55e;border:2px solid #fff;border-radius:50%;box-shadow:0 0 8px rgba(34,197,94,0.6);"></div>`,
        iconSize: [12, 12],
        iconAnchor: [6, 6],
      })

      const selectedIcon = L.divIcon({
        className: "custom-marker",
        html: `<div style="width:16px;height:16px;background:#f59e0b;border:2px solid #fff;border-radius:50%;box-shadow:0 0 12px rgba(245,158,11,0.8);"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      })

      // Add markers
      const markers: any[] = []
      locations.forEach((loc, i) => {
        const icon = i === 0 ? selectedIcon : greenIcon
        const marker = L.marker([loc.lat, loc.lng], { icon })
          .addTo(map)
          .bindPopup(
            `<div style="font-family:monospace;font-size:11px;">
              <strong>${loc.location}</strong><br/>
              ${loc.country} · ${loc.elevation}<br/>
              <em>by ${loc.recordist}</em>
            </div>`,
            { className: "dark-popup" }
          )
        marker.on("click", () => setSelected(loc))
        markers.push(marker)
      })

      // Fit bounds to show all markers
      if (markers.length > 1) {
        const group = L.featureGroup(markers)
        map.fitBounds(group.getBounds().pad(0.2))
      }

      leafletMap.current = map
      markersRef.current = markers
      setMapReady(true)
    }

    initMap()

    return () => {
      if (leafletMap.current) {
        leafletMap.current.off()
        leafletMap.current.remove()
        leafletMap.current = null
      }
      if (mapRef.current) {
        // @ts-ignore Let Leaflet know it can be re-initialized
        delete mapRef.current._leaflet_id
      }
    }
  }, [loading, locations])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/80" onClick={onClose} />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-[1000px] max-h-[90vh] overflow-y-auto border-2 border-foreground bg-background">
        {/* Header */}
        <div className="border-b-2 border-foreground px-4 py-3 flex items-center justify-between sticky top-0 bg-background z-20">
          <div className="flex items-center gap-2">
            <MapPin size={14} className="text-accent" />
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-muted-foreground">
              SIGHTING LOCATIONS — {species.toUpperCase()}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 border border-foreground/40 hover:bg-foreground hover:text-background cursor-pointer transition-none"
          >
            <X size={14} />
          </button>
        </div>

        {/* Stats bar */}
        <div className="border-b border-foreground/30 px-4 py-2 flex items-center gap-6">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 bg-[#22c55e] rounded-full" />
            <span className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
              {locations.length} UNIQUE SITES
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <Mic size={9} className="text-muted-foreground" />
            <span className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
              {totalRecordings.toLocaleString()} TOTAL RECORDINGS
            </span>
          </div>
          {scientificName && (
            <span className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground italic">
              {scientificName}
            </span>
          )}
        </div>

        {/* Map */}
        <div className="relative h-[350px] sm:h-[420px] border-b-2 border-foreground bg-[#1a1a2e]">
          {/* Leaflet CSS */}
          <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          />

          {loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 z-10">
              <Loader2 size={20} className="text-accent animate-spin" />
              <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
                FETCHING SIGHTING DATA...
              </span>
            </div>
          )}

          {!loading && locations.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 z-10">
              <Globe size={24} className="text-muted-foreground" />
              <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
                NO LOCATION DATA AVAILABLE
              </span>
            </div>
          )}

          <div ref={mapRef} className="w-full h-full" />
        </div>

        {/* Selected location details */}
        {selected && (
          <div className="p-5 space-y-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2.5 h-2.5 bg-[#f59e0b] rounded-full" />
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-accent font-bold">
                SELECTED RECORDING SITE
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Location */}
              <div className="sm:col-span-2 space-y-3">
                <div>
                  <div className="flex items-center gap-1.5 mb-1">
                    <MapPin size={10} className="text-accent" />
                    <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-accent">
                      LOCATION
                    </span>
                  </div>
                  <p className="font-mono text-sm font-bold text-foreground">
                    {selected.location}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <Globe size={10} className="text-blue-400" />
                      <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-blue-400">
                        COUNTRY
                      </span>
                    </div>
                    <p className="font-mono text-xs text-foreground">{selected.country}</p>
                  </div>
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <Mountain size={10} className="text-emerald-400" />
                      <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-emerald-400">
                        ELEVATION
                      </span>
                    </div>
                    <p className="font-mono text-xs text-foreground">{selected.elevation}</p>
                  </div>
                </div>

                <div>
                  <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground block mb-1">
                    RECORDIST
                  </span>
                  <p className="font-mono text-xs text-foreground">{selected.recordist}</p>
                </div>
              </div>

              {/* Coordinates card */}
              <div className="border border-foreground/20 p-3 space-y-2 bg-foreground/5">
                <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground block">
                  COORDINATES
                </span>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="font-mono text-[10px] text-muted-foreground">LAT</span>
                    <span className="font-mono text-xs font-bold text-foreground">{selected.lat.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-mono text-[10px] text-muted-foreground">LON</span>
                    <span className="font-mono text-xs font-bold text-foreground">{selected.lng.toFixed(4)}</span>
                  </div>
                </div>
                <div className="h-px bg-foreground/20" />
                <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground block">
                  QUALITY
                </span>
                <span className="font-mono text-sm font-bold text-accent">{selected.quality}</span>

                {selected.url && (
                  <a
                    href={selected.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block mt-2 text-center px-3 py-1.5 border border-accent text-accent font-mono text-[9px] tracking-[0.15em] uppercase hover:bg-accent hover:text-white cursor-pointer transition-none"
                  >
                    VIEW ON XENO-CANTO →
                  </a>
                )}
              </div>
            </div>

            {/* Location list */}
            {locations.length > 1 && (
              <div className="border-t border-foreground/20 pt-3">
                <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground block mb-2">
                  ALL RECORDING SITES ({locations.length})
                </span>
                <div className="max-h-[180px] overflow-y-auto space-y-1 pr-1">
                  {locations.map((loc, i) => (
                    <button
                      key={`${loc.lat}-${loc.lng}-${i}`}
                      type="button"
                      onClick={() => {
                        setSelected(loc)
                        if (leafletMap.current) {
                          leafletMap.current.setView([loc.lat, loc.lng], 8, { animate: true })
                        }
                      }}
                      className={`w-full text-left px-3 py-2 border cursor-pointer transition-none font-mono text-[10px] flex items-center justify-between gap-2 ${selected === loc
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-foreground/10 text-muted-foreground hover:border-foreground/40 hover:text-foreground"
                        }`}
                    >
                      <span className="truncate">{loc.location}, {loc.country}</span>
                      <span className="shrink-0 tracking-wider">{loc.elevation}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
