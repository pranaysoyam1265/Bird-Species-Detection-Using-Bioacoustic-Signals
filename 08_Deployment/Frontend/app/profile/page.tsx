"use client"

import { useState, useEffect, useRef } from "react"
import Image from "next/image"
import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"
import {
  User,
  Mail,
  Calendar,
  Shield,
  Bird,
  LogOut,
  Settings,
  Activity,
  Database,
  Star,
  Pencil,
  Check,
  Briefcase,
  FileText,
} from "lucide-react"
import { getDetections } from "@/lib/detection-store"
import { getAvatar, saveAvatarFromFile, clearAvatar } from "@/lib/avatar-store"

/* ── Mock stats (computed from localStorage) ── */
function useProfileStats() {
  const [stats] = useState(() => {
    try {
      const detections = getDetections()
      const totalDetections = detections.length

      // Count unique species across all detections
      const speciesSet = new Set<string>()
      detections.forEach((d) => {
        if (d.topSpecies) speciesSet.add(d.topSpecies)
        if (d.predictions) {
          d.predictions.forEach((p) => {
            if (p.species) speciesSet.add(p.species)
          })
        }
      })

      // Count favorites
      let favCount = 0
      try {
        const favs = localStorage.getItem("birdsense-favorites")
        if (favs) favCount = JSON.parse(favs).length
      } catch { /* */ }

      return {
        totalDetections,
        uniqueSpecies: speciesSet.size,
        favorites: favCount,
        memberSince: "2024",
      }
    } catch {
      return { totalDetections: 0, uniqueSpecies: 0, favorites: 0, memberSince: "2024" }
    }
  })
  return stats
}

export default function ProfilePage() {
  const { user, loading, logout } = useAuth()
  const router = useRouter()
  const stats = useProfileStats()

  /* ── Auth guard: redirect to login if not authenticated ── */
  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login")
    }
  }, [loading, user, router])

  /* ── Editing state ── */
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState("")
  const [editEmail, setEditEmail] = useState("")
  const [editRole, setEditRole] = useState("")
  const [editOrg, setEditOrg] = useState("")
  const [editBio, setEditBio] = useState("")

  /* ── Avatar state ── */
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setAvatarUrl(getAvatar())
  }, [])

  const startEditing = () => {
    const saved = getSavedProfile()
    setEditName(saved.name || user?.name || user?.email?.split("@")[0] || "Researcher")
    setEditEmail(saved.email || user?.email || "researcher@birdsense.ai")
    setEditRole(saved.role || "Researcher • Full Access")
    setEditOrg(saved.org || "BirdSense Lab")
    setEditBio(saved.bio || "Bioacoustic researcher studying avian vocalizations.")
    setEditing(true)
  }

  const saveEdits = () => {
    try {
      localStorage.setItem("birdsense-profile", JSON.stringify({
        name: editName, email: editEmail, role: editRole, org: editOrg, bio: editBio,
      }))
    } catch { /* */ }
    setEditing(false)
  }

  const handleLogout = async () => {
    await logout()
    router.replace("/")
  }

  /* Show nothing while auth is resolving or redirecting */
  if (loading || !user) {
    return (
      <div className="min-h-screen dot-grid-bg flex items-center justify-center scanline-overlay">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent animate-spin" />
          <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
            AUTHENTICATING...
          </span>
        </div>
      </div>
    )
  }

  /* ── Helper to read saved profile ── */
  function getSavedProfile(): { name?: string; email?: string; role?: string; org?: string; bio?: string } {
    try {
      const raw = localStorage.getItem("birdsense-profile")
      if (raw) return JSON.parse(raw)
    } catch { /* */ }
    return {}
  }

  /* ── Derived display values (merge localStorage overrides) ── */
  const savedProfile = getSavedProfile()
  const displayName = savedProfile.name || user?.name || user?.email?.split("@")[0] || "Researcher"
  const displayEmail = savedProfile.email || user?.email || "researcher@birdsense.ai"
  const displayRole = savedProfile.role || "Researcher • Full Access"
  const displayOrg = savedProfile.org || "BirdSense Lab"
  const displayBio = savedProfile.bio || "Bioacoustic researcher studying avian vocalizations."
  const joinDate = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
    : "January 2024"
  const initials = displayName.slice(0, 2).toUpperCase()

  return (
    <div className="min-h-screen dot-grid-bg flex flex-col relative scanline-overlay">
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />
      <Navbar />

      {/* Header */}
      <div className="px-4 lg:px-6 pt-4 lg:pt-6 flex items-center gap-3">
        <NavSidebar />
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <User size={16} className="text-accent" />
            <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
              PROFILE
            </h1>
          </div>
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            ACCOUNT & ACTIVITY
          </p>
        </div>
      </div>

      <main className="flex-1 px-4 lg:px-6 py-6 lg:py-8 relative z-10 max-w-3xl mx-auto w-full space-y-5">

        {/* ────── PROFILE CARD ────── */}
        <div className="border-2 border-foreground bg-background relative">
          {/* Corner accents */}
          <span className="absolute -top-[3px] -left-[3px] w-2.5 h-2.5 bg-accent" />
          <span className="absolute -top-[3px] -right-[3px] w-2.5 h-2.5 bg-accent" />
          <span className="absolute -bottom-[3px] -left-[3px] w-2.5 h-2.5 bg-accent" />
          <span className="absolute -bottom-[3px] -right-[3px] w-2.5 h-2.5 bg-accent" />

          {/* Edit / Save button — top right */}
          <button
            type="button"
            onClick={editing ? saveEdits : startEditing}
            className="absolute top-3 right-3 w-8 h-8 border-2 border-foreground/30 flex items-center justify-center cursor-pointer hover:border-accent hover:text-accent text-muted-foreground transition-none z-10"
            aria-label={editing ? "Save profile" : "Edit profile"}
          >
            {editing ? <Check size={14} /> : <Pencil size={14} />}
          </button>

          <div className="p-6 sm:p-8 flex flex-col sm:flex-row items-center gap-6">
            {/* Avatar */}
            <div className="relative shrink-0 group">
              <div
                className="w-24 h-24 border-2 border-foreground bg-accent/10 flex items-center justify-center overflow-hidden"
                onClick={() => editing && fileInputRef.current?.click()}
                style={{ cursor: editing ? "pointer" : "default" }}
              >
                {avatarUrl ? (
                  <Image src={avatarUrl} alt="Avatar" width={96} height={96} className="w-full h-full object-cover" />
                ) : (
                  <span className="font-mono text-2xl font-bold text-accent tracking-wider">
                    {initials}
                  </span>
                )}
                {/* Camera overlay when editing */}
                {editing && (
                  <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-hex)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" /><circle cx="12" cy="13" r="3" /></svg>
                    <span className="font-mono text-[8px] tracking-wider text-accent uppercase">Upload</span>
                  </div>
                )}
              </div>
              {/* Remove button when editing and avatar exists */}
              {editing && avatarUrl && (
                <button
                  type="button"
                  onClick={() => { clearAvatar(); setAvatarUrl(null) }}
                  className="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white flex items-center justify-center font-mono text-xs font-bold cursor-pointer border border-background z-10"
                  aria-label="Remove avatar"
                >
                  ×
                </button>
              )}
              {/* Online indicator */}
              {!editing && <span className="absolute -bottom-1 -right-1 w-4 h-4 bg-[#22c55e] border-2 border-background" />}
              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={async (e) => {
                  const f = e.target.files?.[0]
                  if (!f) return
                  try {
                    const uri = await saveAvatarFromFile(f)
                    setAvatarUrl(uri)
                  } catch { /* */ }
                  e.target.value = ""
                }}
              />
            </div>

            {/* Info */}
            <div className="flex-1 text-center sm:text-left space-y-2">
              {editing ? (
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full bg-transparent border-b-2 border-accent font-mono text-lg sm:text-xl tracking-[0.15em] uppercase text-foreground font-bold outline-none py-0.5"
                  autoFocus
                />
              ) : (
                <h2 className="font-mono text-lg sm:text-xl tracking-[0.15em] uppercase text-foreground font-bold">
                  {displayName}
                </h2>
              )}
              <div className="flex flex-col sm:flex-row gap-2 sm:gap-4 items-center sm:items-start">
                <span className="flex items-center gap-1.5 font-mono text-[11px] tracking-wider text-muted-foreground">
                  <Mail size={12} className="text-accent/60" />
                  {editing ? (
                    <input
                      type="email"
                      value={editEmail}
                      onChange={(e) => setEditEmail(e.target.value)}
                      className="bg-transparent border-b border-accent/50 font-mono text-[11px] tracking-wider text-muted-foreground outline-none py-0.5 w-48"
                    />
                  ) : (
                    displayEmail
                  )}
                </span>
                <span className="flex items-center gap-1.5 font-mono text-[11px] tracking-wider text-muted-foreground">
                  <Calendar size={12} className="text-accent/60" />
                  Joined {joinDate}
                </span>
              </div>
              <div className="flex items-center gap-1.5 justify-center sm:justify-start">
                <Shield size={12} className="text-[#22c55e]" />
                <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-[#22c55e] font-bold">
                  VERIFIED RESEARCHER
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ────── ACTIVITY STATS ────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { icon: Activity, label: "DETECTIONS", value: stats.totalDetections.toString(), color: "var(--accent-hex)" },
            { icon: Bird, label: "SPECIES FOUND", value: stats.uniqueSpecies.toString(), color: "#22c55e" },
            { icon: Star, label: "FAVORITES", value: stats.favorites.toString(), color: "#f59e0b" },
            { icon: Database, label: "MEMBER SINCE", value: stats.memberSince, color: "#6366f1" },
          ].map((stat) => (
            <div key={stat.label} className="border-2 border-foreground bg-background p-4 relative group">
              <span className="absolute top-0 left-0 w-full h-0.5" style={{ backgroundColor: stat.color, opacity: 0.4 }} />
              <div className="flex items-center gap-2 mb-2">
                <stat.icon size={14} style={{ color: stat.color }} />
                <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground">
                  {stat.label}
                </span>
              </div>
              <span className="font-mono text-xl sm:text-2xl font-bold text-foreground block">
                {stat.value}
              </span>
            </div>
          ))}
        </div>

        {/* ────── ACCOUNT DETAILS ────── */}
        <div className="border-2 border-foreground bg-background">
          <div className="border-b-2 border-foreground px-4 py-2.5 bg-muted/30">
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
              ACCOUNT DETAILS
            </span>
          </div>
          <div className="divide-y divide-foreground/10">
            {/* NAME */}
            <div className="flex items-center gap-4 px-4 py-3.5">
              <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground">
                <User size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">NAME</span>
                {editing ? (
                  <input type="text" value={editName} onChange={(e) => setEditName(e.target.value)}
                    className="w-full bg-transparent border-b border-accent/50 font-mono text-[11px] tracking-[0.1em] text-foreground outline-none mt-0.5 py-0.5" />
                ) : (
                  <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{displayName}</span>
                )}
              </div>
            </div>
            {/* EMAIL */}
            <div className="flex items-center gap-4 px-4 py-3.5">
              <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground">
                <Mail size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">EMAIL</span>
                {editing ? (
                  <input type="email" value={editEmail} onChange={(e) => setEditEmail(e.target.value)}
                    className="w-full bg-transparent border-b border-accent/50 font-mono text-[11px] tracking-[0.1em] text-foreground outline-none mt-0.5 py-0.5" />
                ) : (
                  <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{displayEmail}</span>
                )}
              </div>
            </div>
            {/* ROLE */}
            <div className="flex items-center gap-4 px-4 py-3.5">
              <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground">
                <Shield size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">ROLE</span>
                {editing ? (
                  <input type="text" value={editRole} onChange={(e) => setEditRole(e.target.value)}
                    className="w-full bg-transparent border-b border-accent/50 font-mono text-[11px] tracking-[0.1em] text-foreground outline-none mt-0.5 py-0.5" />
                ) : (
                  <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{displayRole}</span>
                )}
              </div>
            </div>
            {/* ORGANIZATION */}
            <div className="flex items-center gap-4 px-4 py-3.5">
              <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground">
                <Briefcase size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">ORGANIZATION</span>
                {editing ? (
                  <input type="text" value={editOrg} onChange={(e) => setEditOrg(e.target.value)}
                    className="w-full bg-transparent border-b border-accent/50 font-mono text-[11px] tracking-[0.1em] text-foreground outline-none mt-0.5 py-0.5" />
                ) : (
                  <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{displayOrg}</span>
                )}
              </div>
            </div>
            {/* BIO */}
            <div className="flex items-start gap-4 px-4 py-3.5">
              <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground mt-0.5">
                <FileText size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">BIO</span>
                {editing ? (
                  <textarea value={editBio} onChange={(e) => setEditBio(e.target.value)} rows={2}
                    className="w-full bg-transparent border-b border-accent/50 font-mono text-[11px] tracking-[0.1em] text-foreground outline-none mt-0.5 py-0.5 resize-none" />
                ) : (
                  <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{displayBio}</span>
                )}
              </div>
            </div>
            {/* JOINED (read-only) */}
            <div className="flex items-center gap-4 px-4 py-3.5">
              <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground">
                <Calendar size={14} />
              </div>
              <div className="flex-1 min-w-0">
                <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">JOINED</span>
                <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{joinDate}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ────── QUICK LINKS ────── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => router.push("/settings")}
            className="flex items-center gap-3 px-4 py-3.5 border-2 border-foreground bg-background hover:bg-muted cursor-pointer transition-none text-left"
          >
            <Settings size={16} className="text-accent shrink-0" />
            <div>
              <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">SETTINGS</span>
              <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground">Preferences & data management</span>
            </div>
          </button>
          <button
            type="button"
            onClick={handleLogout}
            className="flex items-center gap-3 px-4 py-3.5 border-2 border-red-500/30 bg-background hover:bg-red-500/5 cursor-pointer transition-none text-left"
          >
            <LogOut size={16} className="text-red-500 shrink-0" />
            <div>
              <span className="font-mono text-xs tracking-[0.15em] uppercase text-red-500 font-bold block">LOG OUT</span>
              <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground">End current session</span>
            </div>
          </button>
        </div>

        {/* ────── STATUS BAR ────── */}
        <div className="border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">SYS_STATUS:</span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            PROFILE LOADED • SESSION ACTIVE
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>
      </main>
    </div>
  )
}
