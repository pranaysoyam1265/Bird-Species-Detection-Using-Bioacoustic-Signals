"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import { useTheme } from "next-themes"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"
import {
  Settings,
  Sun,
  Moon,
  Trash2,
  Download,
  Upload,
  Volume2,
  VolumeX,
  Database,
  Bell,
  BellOff,
  Monitor,
  RotateCcw,
  Check,
  Palette,
  Type,
  Layers,
  Gauge,
  Mic,
  BarChart3,
  Image,
  Lock,
  UserX,
  Clock,
  HardDrive,
  Key,
  Webhook,
  Copy,
  Eye,
  EyeOff,
  Plus,
  X,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { getDetections, deleteDetections } from "@/lib/detection-store"

/* ── Preference keys ── */
const PREF_KEYS = {
  notifications: "birdsense-pref-notifications",
  audioPreview: "birdsense-pref-audio-preview",
  pageSize: "birdsense-pref-page-size",
  sortKey: "birdsense-results-sort",
  accentColor: "birdsense-pref-accent",
  fontSize: "birdsense-pref-fontsize",
  scanlines: "birdsense-pref-scanlines",
  confidenceThreshold: "birdsense-pref-confidence-threshold",
  defaultInputMode: "birdsense-pref-input-mode",
  defaultTopK: "birdsense-pref-topk",
  spectrogramColorMap: "birdsense-pref-colormap",
  autoClear: "birdsense-pref-autoclear",
  apiKeys: "birdsense-api-keys",
  webhookUrl: "birdsense-webhook-url",
} as const

const ACCENT_COLORS = [
  { name: "Electric Orange", value: "#ea580c" },
  { name: "Cyan", value: "#06b6d4" },
  { name: "Violet", value: "#8b5cf6" },
  { name: "Green", value: "#22c55e" },
  { name: "Pink", value: "#ec4899" },
  { name: "Amber", value: "#f59e0b" },
]

const FONT_SIZES = ["small", "default", "large"] as const
const COLORMAPS = ["Viridis", "Magma", "Inferno", "Grayscale"] as const
const AUTO_CLEAR_OPTIONS = [
  { label: "Never", value: 0 },
  { label: "7 days", value: 7 },
  { label: "30 days", value: 30 },
  { label: "90 days", value: 90 },
]

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const { toast } = useToast()
  const { user, logout } = useAuth()
  const router = useRouter()
  const importRef = useRef<HTMLInputElement>(null)

  // ── State ──
  const [notifications, setNotifications] = useState(true)
  const [audioPreview, setAudioPreview] = useState(true)
  const [pageSize, setPageSize] = useState(10)
  const [historyCount, setHistoryCount] = useState(0)
  const [storageSizeMB, setStorageSizeMB] = useState("0.0")
  const [confirmClear, setConfirmClear] = useState(false)

  // System Health state
  const [sysHealth, setSysHealth] = useState<{ status: string, numSpecies: number, device: string } | null>(null)

  // New settings state
  const [accentColor, setAccentColor] = useState("var(--accent-hex)")
  const [fontSize, setFontSize] = useState<typeof FONT_SIZES[number]>("default")
  const [scanlinesEnabled, setScanlinesEnabled] = useState(true)
  const [confidenceThreshold, setConfidenceThreshold] = useState(10)
  const [defaultInputMode, setDefaultInputMode] = useState<"upload" | "record">("upload")
  const [defaultTopK, setDefaultTopK] = useState(5)
  const [spectrogramColorMap, setSpectrogramColorMap] = useState<typeof COLORMAPS[number]>("Viridis")
  const [autoClearDays, setAutoClearDays] = useState(0)

  // Account
  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPasswords, setShowPasswords] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleteConfirmText, setDeleteConfirmText] = useState("")

  // API keys
  const [apiKeys, setApiKeys] = useState<{ id: string; key: string; created: string; name: string }[]>([])
  const [newKeyName, setNewKeyName] = useState("")

  // Webhook
  const [webhookUrl, setWebhookUrl] = useState("")

  // Storage breakdown
  const [storageBreakdown, setStorageBreakdown] = useState<{ label: string; bytes: number; color: string }[]>([])

  // Fetch settings from API
  useEffect(() => {
    async function loadData() {
      try {
        // Load Settings
        const sRes = await fetch("/api/settings")
        if (sRes.ok) {
          const { settings } = await sRes.json()
          if (settings) {
            setNotifications(settings.notifications ?? true)
            setAudioPreview(settings.audioPreview ?? true)
            setPageSize(settings.pageSize ?? 10)
            setAccentColor(settings.accentColor || "var(--accent-hex)")
            setFontSize(settings.fontSize || "default")
            setScanlinesEnabled(settings.scanlines ?? true)
            setConfidenceThreshold(settings.confidenceThreshold ?? 10)
            setDefaultInputMode(settings.defaultInputMode || "upload")
            setDefaultTopK(settings.defaultTopK ?? 5)
            setSpectrogramColorMap(settings.spectrogramColorMap || "Viridis")
            setAutoClearDays(settings.autoClear ?? 0)
            setWebhookUrl(settings.webhookUrl || "")
          }
        }

        // Load API Keys
        const kRes = await fetch("/api/settings/keys")
        if (kRes.ok) {
          const kData = await kRes.json()
          setApiKeys(kData.api_keys || [])
        }

        // Fetch System Health directly from FastAPI (assuming standard route structure)
        // Adjust port if necessary, typically NEXT_PUBLIC or dynamic via proxy
        const hRes = await fetch("/api/health").catch(() => null)
        if (hRes && hRes.ok) {
          const hData = await hRes.json()
          setSysHealth(hData)
        }

        // Trigger background cleanup (fire and forget)
        fetch("/api/history/cleanup", { method: "POST" }).catch(() => null)

      } catch (err) {
        toast({ title: "Failed to load settings" })
      }
      computeStorage()
    }
    loadData()
  }, [])

  const computeStorage = () => {
    const records = getDetections()
    setHistoryCount(records.length)

    // Breakdown by category
    let detectionBytes = 0
    let favoritesBytes = 0
    let avatarBytes = 0
    let prefBytes = 0
    let otherBytes = 0

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (!key) continue
      const size = ((localStorage.getItem(key) || "").length) * 2
      if (key === "birdsense-detections") detectionBytes += size
      else if (key === "birdsense-favorites") favoritesBytes += size
      else if (key === "birdsense-avatar") avatarBytes += size
      else if (key.startsWith("birdsense-pref")) prefBytes += size
      else if (key.startsWith("birdsense")) otherBytes += size
    }

    const total = detectionBytes + favoritesBytes + avatarBytes + prefBytes + otherBytes
    setStorageSizeMB((total / (1024 * 1024)).toFixed(2))
    setStorageBreakdown([
      { label: "Detections", bytes: detectionBytes, color: "var(--accent-hex)" },
      { label: "Favorites", bytes: favoritesBytes, color: "#f59e0b" },
      { label: "Avatar", bytes: avatarBytes, color: "#8b5cf6" },
      { label: "Preferences", bytes: prefBytes, color: "#06b6d4" },
      { label: "Other", bytes: otherBytes, color: "#6b7280" },
    ])
  }

  // Instead of direct localStorage writes, bounce settings object via Debounced API call
  // This simplifies code by triggering a global save object for any state change
  const buildSettingsObject = useCallback((overrides: Record<string, unknown> = {}) => {
    return {
      notifications,
      audioPreview,
      pageSize,
      accentColor,
      fontSize,
      scanlines: scanlinesEnabled,
      confidenceThreshold,
      defaultInputMode,
      defaultTopK,
      spectrogramColorMap,
      autoClear: autoClearDays,
      webhookUrl,
      ...overrides
    }
  }, [
    notifications, audioPreview, pageSize, accentColor, fontSize,
    scanlinesEnabled, confidenceThreshold, defaultInputMode,
    defaultTopK, spectrogramColorMap, autoClearDays, webhookUrl
  ])

  const savePref = async (key: string, value: unknown) => {
    try { localStorage.setItem(key, typeof value === "string" ? value : JSON.stringify(value)) } catch { /* */ }

    // Convert back from localStorage keys to properties where possible
    const propMap: Record<string, string> = {
      [PREF_KEYS.notifications]: "notifications",
      [PREF_KEYS.audioPreview]: "audioPreview",
      [PREF_KEYS.pageSize]: "pageSize",
      [PREF_KEYS.accentColor]: "accentColor",
      [PREF_KEYS.fontSize]: "fontSize",
      [PREF_KEYS.scanlines]: "scanlines",
      [PREF_KEYS.confidenceThreshold]: "confidenceThreshold",
      [PREF_KEYS.defaultInputMode]: "defaultInputMode",
      [PREF_KEYS.defaultTopK]: "defaultTopK",
      [PREF_KEYS.spectrogramColorMap]: "spectrogramColorMap",
      [PREF_KEYS.autoClear]: "autoClear",
      [PREF_KEYS.webhookUrl]: "webhookUrl"
    }

    const propName = propMap[key]
    if (propName) {
      await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSettingsObject({ [propName]: value }))
      }).catch(err => console.error(err))
    }
  }

  const toggleNotifications = () => {
    const v = !notifications
    setNotifications(v)
    savePref(PREF_KEYS.notifications, v)
    toast({ title: v ? "Notifications enabled" : "Notifications disabled" })
  }

  const toggleAudioPreview = () => {
    const v = !audioPreview
    setAudioPreview(v)
    savePref(PREF_KEYS.audioPreview, v)
    toast({ title: v ? "Audio preview enabled" : "Audio preview muted" })
  }

  const changePageSize = (size: number) => {
    setPageSize(size)
    savePref(PREF_KEYS.pageSize, size)
    toast({ title: `Page size set to ${size}` })
  }

  const handleAccentColor = (color: string) => {
    setAccentColor(color)
    savePref(PREF_KEYS.accentColor, color)
    // Convert hex to HSL for Tailwind's accent utility
    const h = color.replace("#", "")
    const r = parseInt(h.substring(0, 2), 16) / 255
    const g = parseInt(h.substring(2, 4), 16) / 255
    const b = parseInt(h.substring(4, 6), 16) / 255
    const max = Math.max(r, g, b), min = Math.min(r, g, b)
    const l = (max + min) / 2
    let s = 0, hue = 0
    if (max !== min) {
      const d = max - min
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
      switch (max) {
        case r: hue = ((g - b) / d + (g < b ? 6 : 0)); break
        case g: hue = ((b - r) / d + 2); break
        case b: hue = ((r - g) / d + 4); break
      }
      hue *= 60
    }
    const hsl = `${Math.round(hue)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`
    document.documentElement.style.setProperty("--accent", hsl)
    document.documentElement.style.setProperty("--accent-hex", color)
    toast({ title: "Accent color updated" })
  }

  const handleFontSize = (size: typeof FONT_SIZES[number]) => {
    setFontSize(size)
    savePref(PREF_KEYS.fontSize, size)
    toast({ title: `Font size: ${size}` })
  }

  const toggleScanlines = () => {
    const v = !scanlinesEnabled
    setScanlinesEnabled(v)
    savePref(PREF_KEYS.scanlines, v)
    toast({ title: v ? "Scanline overlay enabled" : "Scanline overlay disabled" })
  }

  const handleConfidenceThreshold = (val: number) => {
    setConfidenceThreshold(val)
    savePref(PREF_KEYS.confidenceThreshold, val)
  }

  const handleDefaultInputMode = (mode: "upload" | "record") => {
    setDefaultInputMode(mode)
    savePref(PREF_KEYS.defaultInputMode, mode)
    toast({ title: `Default mode: ${mode}` })
  }

  const handleDefaultTopK = (k: number) => {
    setDefaultTopK(k)
    savePref(PREF_KEYS.defaultTopK, k)
    toast({ title: `Default top-K: ${k}` })
  }

  const handleColorMap = (cm: typeof COLORMAPS[number]) => {
    setSpectrogramColorMap(cm)
    savePref(PREF_KEYS.spectrogramColorMap, cm)
    toast({ title: `Color map: ${cm}` })
  }

  const handleAutoClear = (days: number) => {
    setAutoClearDays(days)
    savePref(PREF_KEYS.autoClear, days)
    toast({ title: days === 0 ? "Auto-clear disabled" : `Auto-clear: ${days} days` })
  }

  const clearHistory = () => {
    const ids = getDetections().map((d) => d.id)
    deleteDetections(ids)
    setHistoryCount(0)
    setConfirmClear(false)
    computeStorage()
    toast({ title: "Detection history cleared", description: `${ids.length} records deleted` })
  }

  const exportAllData = () => {
    const data: Record<string, string | null> = {}
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith("birdsense")) {
        data[key] = localStorage.getItem(key)
      }
    }
    const json = JSON.stringify(data, null, 2)
    const blob = new Blob([json], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "birdsense-all-data.json"
    a.click()
    URL.revokeObjectURL(url)
    toast({ title: "All data exported" })
  }

  const importData = (file: File) => {
    const reader = new FileReader()
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string)
        let count = 0
        for (const [key, value] of Object.entries(data)) {
          if (typeof key === "string" && key.startsWith("birdsense") && typeof value === "string") {
            localStorage.setItem(key, value)
            count++
          }
        }
        computeStorage()
        toast({ title: `Imported ${count} keys`, description: "Reload the page to see updated data" })
      } catch {
        toast({ title: "Import failed", description: "Invalid JSON file" })
      }
    }
    reader.readAsText(file)
  }

  const resetPreferences = () => {
    Object.values(PREF_KEYS).forEach((key) => {
      try { localStorage.removeItem(key) } catch { /* */ }
    })
    setNotifications(true)
    setAudioPreview(true)
    setPageSize(10)
    setAccentColor("var(--accent-hex)")
    setFontSize("default")
    setScanlinesEnabled(true)
    setConfidenceThreshold(10)
    setDefaultInputMode("upload")
    setDefaultTopK(5)
    setSpectrogramColorMap("Viridis")
    setAutoClearDays(0)
    setWebhookUrl("")
    setApiKeys([])
    toast({ title: "Preferences reset to defaults" })
  }

  const handleChangePassword = async () => {
    if (!newPassword || newPassword.length < 6) {
      toast({ title: "Password too short", description: "Must be at least 6 characters" })
      return
    }
    if (newPassword !== confirmPassword) {
      toast({ title: "Passwords don't match" })
      return
    }
    try {
      const res = await fetch("/api/auth/change-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentPassword, newPassword }),
      })
      const data = await res.json()
      if (!res.ok) {
        toast({ title: "Failed to update password", description: data.error || "Unknown error" })
        return
      }
      setCurrentPassword("")
      setNewPassword("")
      setConfirmPassword("")
      toast({ title: "Password updated successfully" })
    } catch {
      toast({ title: "Failed to update password", description: "Network error" })
    }
  }

  // Delete account
  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== "DELETE") return
    // Clear all data
    const keysToRemove: string[] = []
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith("birdsense")) keysToRemove.push(key)
    }
    keysToRemove.forEach((k) => localStorage.removeItem(k))
    await logout()
    router.replace("/")
  }

  const generateApiKey = async () => {
    try {
      const res = await fetch("/api/settings/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newKeyName || "Unnamed Key" })
      })
      if (!res.ok) throw new Error("Failed to generate key")
      const data = await res.json()

      const updated = [data, ...apiKeys]
      setApiKeys(updated)
      setNewKeyName("")
      toast({ title: "API key generated", description: "Copy it now — you won't see it again" })

      // We automatically select the key to clipboard to make UX superb
      navigator.clipboard.writeText(data.key)
    } catch {
      toast({ title: "Error generating key" })
    }
  }

  const revokeApiKey = async (id: string) => {
    try {
      const res = await fetch(`/api/settings/keys?id=${id}`, { method: "DELETE" })
      if (!res.ok) throw new Error("Failed")
      const updated = apiKeys.filter((k) => k.id !== id)
      setApiKeys(updated)
      toast({ title: "API key revoked" })
    } catch {
      toast({ title: "Failed to revoke key" })
    }
  }

  const handleWebhookSave = () => {
    savePref(PREF_KEYS.webhookUrl, webhookUrl)
    toast({ title: webhookUrl ? "Webhook URL saved" : "Webhook URL cleared" })
  }

  /* ── Reusable components ── */
  const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="border-2 border-foreground bg-background">
      <div className="border-b-2 border-foreground px-4 py-2.5 bg-muted/30">
        <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">{title}</span>
      </div>
      <div className="divide-y divide-foreground/10">{children}</div>
    </div>
  )

  const Row = ({ icon, label, description, children, vertical }: { icon: React.ReactNode; label: string; description?: string; children: React.ReactNode; vertical?: boolean }) => (
    <div className={`flex ${vertical ? "flex-col gap-3" : "items-center"} gap-4 px-4 py-3.5`}>
      <div className={`flex items-center gap-4 ${vertical ? "" : "flex-1 min-w-0"}`}>
        <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0 text-muted-foreground">
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <span className="font-mono text-xs tracking-[0.15em] uppercase text-foreground font-bold block">{label}</span>
          {description && (
            <span className="font-mono text-[9px] tracking-[0.1em] text-muted-foreground block mt-0.5">{description}</span>
          )}
        </div>
      </div>
      <div className={`shrink-0 ${vertical ? "pl-12" : ""}`}>{children}</div>
    </div>
  )

  const Toggle = ({ on, onClick }: { on: boolean; onClick: () => void }) => (
    <button
      type="button"
      onClick={onClick}
      className={`w-12 h-6 border-2 relative cursor-pointer transition-none ${on ? "border-accent bg-accent" : "border-foreground/30 bg-background"}`}
    >
      <span
        className={`absolute top-0.5 w-4 h-4 transition-none ${on ? "right-0.5 bg-white" : "left-0.5 bg-foreground/30"}`}
      />
    </button>
  )

  const SegmentedControl = ({ options, value, onChange }: { options: { label: string; value: string | number }[]; value: string | number; onChange: (v: string | number) => void }) => (
    <div className="flex border-2 border-foreground">
      {options.map((opt, i) => (
        <button
          key={String(opt.value)}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`px-3 py-1.5 font-mono text-[10px] tracking-[0.1em] uppercase cursor-pointer transition-none ${i < options.length - 1 ? "border-r border-foreground" : ""} ${String(value) === String(opt.value) ? "bg-accent text-white" : "bg-background text-foreground hover:bg-muted"}`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )

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
            <Settings size={16} className="text-accent" />
            <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
              SETTINGS
            </h1>
          </div>
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            PREFERENCES &amp; DATA MANAGEMENT
          </p>
        </div>
      </div>

      <main className="flex-1 px-4 lg:px-6 py-6 lg:py-8 relative z-10 max-w-3xl mx-auto w-full space-y-5">

        {/* ────── APPEARANCE ────── */}
        <Section title="APPEARANCE">
          <Row icon={<Monitor size={14} />} label="THEME" description="Switch between light and dark mode">
            <SegmentedControl
              options={[
                { label: "☀ Light", value: "light" },
                { label: "🌙 Dark", value: "dark" },
              ]}
              value={theme || "dark"}
              onChange={(v) => setTheme(v as string)}
            />
          </Row>
          <Row icon={<Palette size={14} />} label="ACCENT COLOR" description="Choose your primary accent color">
            <div className="flex gap-2">
              {ACCENT_COLORS.map((c) => (
                <button
                  key={c.value}
                  type="button"
                  onClick={() => handleAccentColor(c.value)}
                  className={`w-7 h-7 border-2 cursor-pointer transition-none ${accentColor === c.value ? "border-foreground scale-110" : "border-transparent hover:border-foreground/30"}`}
                  style={{ backgroundColor: c.value }}
                  title={c.name}
                  aria-label={`Set accent color to ${c.name}`}
                />
              ))}
            </div>
          </Row>
          <Row icon={<Type size={14} />} label="FONT SIZE" description="Adjust interface text size">
            <SegmentedControl
              options={[
                { label: "Sm", value: "small" },
                { label: "Md", value: "default" },
                { label: "Lg", value: "large" },
              ]}
              value={fontSize}
              onChange={(v) => handleFontSize(v as typeof FONT_SIZES[number])}
            />
          </Row>
          <Row icon={<Layers size={14} />} label="SCANLINE OVERLAY" description="CRT-style scanline effect on pages">
            <Toggle on={scanlinesEnabled} onClick={toggleScanlines} />
          </Row>
        </Section>

        {/* ────── NOTIFICATIONS & AUDIO ────── */}
        <Section title="NOTIFICATIONS & AUDIO">
          <Row icon={notifications ? <Bell size={14} /> : <BellOff size={14} />} label="TOAST NOTIFICATIONS" description="Show feedback messages for actions">
            <Toggle on={notifications} onClick={toggleNotifications} />
          </Row>
          <Row icon={audioPreview ? <Volume2 size={14} /> : <VolumeX size={14} />} label="AUDIO PREVIEW" description="Enable inline audio playback on results page">
            <Toggle on={audioPreview} onClick={toggleAudioPreview} />
          </Row>
          <Row icon={<Gauge size={14} />} label="CONFIDENCE THRESHOLD" description={`Only show detections above ${confidenceThreshold}% confidence`} vertical>
            <div className="flex items-center gap-3 w-full">
              <input
                type="range"
                min={0}
                max={95}
                step={5}
                value={confidenceThreshold}
                onChange={(e) => handleConfidenceThreshold(Number(e.target.value))}
                className="flex-1 h-1.5 appearance-none bg-foreground/20 cursor-pointer accent-accent"
              />
              <span className="font-mono text-xs font-bold text-accent w-10 text-right">{confidenceThreshold}%</span>
            </div>
          </Row>
        </Section>

        {/* ────── DISPLAY & ANALYSIS ────── */}
        <Section title="DISPLAY & ANALYSIS">
          <Row icon={<Database size={14} />} label="RESULTS PER PAGE" description="Number of records shown per page">
            <SegmentedControl
              options={[
                { label: "10", value: 10 },
                { label: "25", value: 25 },
                { label: "50", value: 50 },
              ]}
              value={pageSize}
              onChange={(v) => changePageSize(v as number)}
            />
          </Row>
          <Row icon={<Mic size={14} />} label="DEFAULT INPUT MODE" description="Default mode when opening the analyze page">
            <SegmentedControl
              options={[
                { label: "Upload", value: "upload" },
                { label: "Record", value: "record" },
              ]}
              value={defaultInputMode}
              onChange={(v) => handleDefaultInputMode(v as "upload" | "record")}
            />
          </Row>
          <Row icon={<BarChart3 size={14} />} label="DEFAULT TOP-K" description="Number of predictions shown by default">
            <SegmentedControl
              options={[
                { label: "3", value: 3 },
                { label: "5", value: 5 },
                { label: "10", value: 10 },
              ]}
              value={defaultTopK}
              onChange={(v) => handleDefaultTopK(v as number)}
            />
          </Row>
          <Row icon={<Image size={14} />} label="SPECTROGRAM COLOR MAP" description="Color scheme for spectrogram visualizations">
            <SegmentedControl
              options={COLORMAPS.map((c) => ({ label: c, value: c }))}
              value={spectrogramColorMap}
              onChange={(v) => handleColorMap(v as typeof COLORMAPS[number])}
            />
          </Row>
        </Section>

        {/* ────── ACCOUNT & PRIVACY ────── */}
        <Section title="ACCOUNT & PRIVACY">
          <Row icon={<Lock size={14} />} label="CHANGE PASSWORD" description="Update your account password" vertical>
            <div className="space-y-2 w-full max-w-xs">
              <div className="relative">
                <input
                  type={showPasswords ? "text" : "password"}
                  placeholder="Current password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="w-full px-3 py-2 border-2 border-foreground/30 bg-background font-mono text-[11px] tracking-wider text-foreground outline-none focus:border-accent pr-8"
                />
              </div>
              <input
                type={showPasswords ? "text" : "password"}
                placeholder="New password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full px-3 py-2 border-2 border-foreground/30 bg-background font-mono text-[11px] tracking-wider text-foreground outline-none focus:border-accent"
              />
              <input
                type={showPasswords ? "text" : "password"}
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full px-3 py-2 border-2 border-foreground/30 bg-background font-mono text-[11px] tracking-wider text-foreground outline-none focus:border-accent"
              />
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={handleChangePassword}
                  className="px-4 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
                >
                  UPDATE
                </button>
                <button
                  type="button"
                  onClick={() => setShowPasswords(!showPasswords)}
                  className="flex items-center gap-1 px-3 py-2 border-2 border-foreground/30 font-mono text-[10px] tracking-[0.1em] uppercase text-muted-foreground hover:text-foreground cursor-pointer transition-none"
                >
                  {showPasswords ? <EyeOff size={10} /> : <Eye size={10} />}
                  {showPasswords ? "Hide" : "Show"}
                </button>
              </div>
            </div>
          </Row>
          <Row icon={<UserX size={14} />} label="DELETE ACCOUNT" description="Permanently delete your account and all data">
            {!confirmDelete ? (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                className="flex items-center gap-1.5 px-4 py-2 border-2 border-red-500/50 font-mono text-[10px] tracking-[0.15em] uppercase text-red-500 hover:bg-red-500/10 cursor-pointer transition-none"
              >
                <UserX size={10} /> DELETE
              </button>
            ) : (
              <div className="space-y-2">
                <p className="font-mono text-[9px] tracking-wider text-red-500">
                  Type DELETE to confirm. This action is irreversible.
                </p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={deleteConfirmText}
                    onChange={(e) => setDeleteConfirmText(e.target.value)}
                    placeholder='Type "DELETE"'
                    className="px-3 py-2 border-2 border-red-500/50 bg-background font-mono text-[10px] tracking-wider text-foreground outline-none focus:border-red-500 w-32"
                  />
                  <button
                    type="button"
                    onClick={handleDeleteAccount}
                    disabled={deleteConfirmText !== "DELETE"}
                    className="px-3 py-2 border-2 border-red-500 bg-red-500 text-white font-mono text-[10px] tracking-[0.15em] uppercase cursor-pointer transition-none disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    CONFIRM
                  </button>
                  <button
                    type="button"
                    onClick={() => { setConfirmDelete(false); setDeleteConfirmText("") }}
                    className="px-3 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
                  >
                    CANCEL
                  </button>
                </div>
              </div>
            )}
          </Row>
        </Section>

        {/* ────── DATA MANAGEMENT ────── */}
        <Section title="DATA MANAGEMENT">
          <Row icon={<HardDrive size={14} />} label="STORAGE USED" description={`${historyCount} detection records stored`} vertical>
            <div className="space-y-2 w-full">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-bold text-foreground">{storageSizeMB} MB</span>
              </div>
              {/* Storage breakdown bar */}
              {Number(storageSizeMB) > 0 && (
                <>
                  <div className="w-full h-3 border border-foreground/30 flex overflow-hidden">
                    {storageBreakdown.filter((s) => s.bytes > 0).map((s) => {
                      const total = storageBreakdown.reduce((a, b) => a + b.bytes, 0)
                      const pct = total > 0 ? (s.bytes / total) * 100 : 0
                      return (
                        <div
                          key={s.label}
                          style={{ width: `${pct}%`, backgroundColor: s.color }}
                          title={`${s.label}: ${(s.bytes / 1024).toFixed(1)} KB`}
                        />
                      )
                    })}
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    {storageBreakdown.filter((s) => s.bytes > 0).map((s) => (
                      <div key={s.label} className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 inline-block" style={{ backgroundColor: s.color }} />
                        <span className="font-mono text-[8px] tracking-wider text-muted-foreground uppercase">
                          {s.label} ({(s.bytes / 1024).toFixed(1)} KB)
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </Row>
          <Row icon={<Download size={14} />} label="EXPORT ALL DATA" description="Download all BirdSense data as JSON">
            <button
              type="button"
              onClick={exportAllData}
              className="flex items-center gap-1.5 px-4 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
            >
              <Download size={10} /> EXPORT
            </button>
          </Row>
          <Row icon={<Upload size={14} />} label="IMPORT DATA" description="Restore from a previously exported JSON backup">
            <button
              type="button"
              onClick={() => importRef.current?.click()}
              className="flex items-center gap-1.5 px-4 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
            >
              <Upload size={10} /> IMPORT
            </button>
            <input
              ref={importRef}
              type="file"
              accept=".json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) importData(f)
                e.target.value = ""
              }}
            />
          </Row>
          <Row icon={<Clock size={14} />} label="AUTO-CLEAR HISTORY" description="Automatically delete old detection records">
            <SegmentedControl
              options={AUTO_CLEAR_OPTIONS.map((o) => ({ label: o.label, value: o.value }))}
              value={autoClearDays}
              onChange={(v) => handleAutoClear(v as number)}
            />
          </Row>
          <Row icon={<Trash2 size={14} />} label="CLEAR DETECTION HISTORY" description="Permanently delete all saved results">
            {!confirmClear ? (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="flex items-center gap-1.5 px-4 py-2 border-2 border-red-500/50 font-mono text-[10px] tracking-[0.15em] uppercase text-red-500 hover:bg-red-500/10 cursor-pointer transition-none"
              >
                <Trash2 size={10} /> CLEAR
              </button>
            ) : (
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="px-3 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
                >
                  CANCEL
                </button>
                <button
                  type="button"
                  onClick={clearHistory}
                  className="px-3 py-2 border-2 border-red-500 bg-red-500 text-white font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-red-600 cursor-pointer transition-none"
                >
                  CONFIRM
                </button>
              </div>
            )}
          </Row>
          <Row icon={<RotateCcw size={14} />} label="RESET PREFERENCES" description="Restore all settings to defaults">
            <button
              type="button"
              onClick={resetPreferences}
              className="flex items-center gap-1.5 px-4 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none"
            >
              <RotateCcw size={10} /> RESET
            </button>
          </Row>
        </Section>

        {/* ────── INTEGRATIONS ────── */}
        <Section title="INTEGRATIONS">
          <Row icon={<Key size={14} />} label="API KEYS" description="Generate keys for programmatic access" vertical>
            <div className="space-y-3 w-full">
              {/* Existing keys */}
              {apiKeys.length > 0 && (
                <div className="space-y-2">
                  {apiKeys.map((k) => (
                    <div key={k.id} className="flex items-center gap-2 border border-foreground/20 px-3 py-2">
                      <div className="flex-1 min-w-0">
                        <span className="font-mono text-[10px] tracking-wider text-foreground font-bold block">{k.name}</span>
                        <span className="font-mono text-[8px] tracking-wider text-muted-foreground block">
                          {k.key ? `${k.key.slice(0, 12)}...${k.key.slice(-4)}` : `${k.id} (Key Hidden)`} • {new Date(k.created).toLocaleDateString()}
                        </span>
                      </div>
                      {k.key && (
                        <button
                          type="button"
                          onClick={() => { navigator.clipboard.writeText(k.key); toast({ title: "Key copied" }) }}
                          className="w-7 h-7 border border-foreground/20 flex items-center justify-center text-muted-foreground hover:text-foreground cursor-pointer transition-none"
                          aria-label="Copy key"
                        >
                          <Copy size={12} />
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => revokeApiKey(k.id)}
                        className="w-7 h-7 border border-red-500/30 flex items-center justify-center text-red-500 hover:bg-red-500/10 cursor-pointer transition-none"
                        aria-label="Revoke key"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              {/* Generate new key */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="Key name (e.g., Production)"
                  className="flex-1 px-3 py-2 border-2 border-foreground/30 bg-background font-mono text-[10px] tracking-wider text-foreground outline-none focus:border-accent"
                />
                <button
                  type="button"
                  onClick={generateApiKey}
                  className="flex items-center gap-1 px-4 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none shrink-0"
                >
                  <Plus size={10} /> GENERATE
                </button>
              </div>
            </div>
          </Row>
          <Row icon={<Webhook size={14} />} label="WEBHOOK URL" description="Receive detection results at this URL automatically" vertical>
            <div className="flex gap-2 w-full max-w-md">
              <input
                type="url"
                value={webhookUrl}
                onChange={(e) => setWebhookUrl(e.target.value)}
                placeholder="https://example.com/webhook"
                className="flex-1 px-3 py-2 border-2 border-foreground/30 bg-background font-mono text-[10px] tracking-wider text-foreground outline-none focus:border-accent"
              />
              <button
                type="button"
                onClick={handleWebhookSave}
                className="px-4 py-2 border-2 border-foreground font-mono text-[10px] tracking-[0.15em] uppercase hover:bg-muted cursor-pointer transition-none shrink-0"
              >
                SAVE
              </button>
            </div>
          </Row>
        </Section>

        {/* ────── ABOUT ────── */}
        <Section title="ABOUT">
          <Row icon={<Check size={14} />} label="SYSTEM STATUS" description="FastAPI Inference Engine">
            <span className={`font-mono text-xs font-bold ${sysHealth?.status === "ok" ? "text-[#22c55e]" : "text-amber-500"}`}>
              {sysHealth ? sysHealth.status.toUpperCase() : "FETCHING..."}
            </span>
          </Row>
          <Row icon={<Database size={14} />} label="MODEL" description={`BioacousticFSL CNN • ${sysHealth?.numSpecies || 87} species`}>
            <span className="font-mono text-xs font-bold text-foreground">
              {sysHealth ? sysHealth.device || "CPU" : "Unknown"}
            </span>
          </Row>
        </Section>

        {/* ────── KEYBOARD SHORTCUTS ────── */}
        <div className="border-2 border-foreground bg-background">
          <div className="border-b-2 border-foreground px-4 py-2.5 bg-muted/30">
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">KEYBOARD SHORTCUTS</span>
          </div>
          <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
            {[
              ["Ctrl + K", "Open command palette"],
              ["← →", "Navigate pages"],
              ["Delete", "Delete selected records"],
              ["Escape", "Close modals / stop audio"],
            ].map(([key, desc]) => (
              <div key={key} className="flex items-center gap-3">
                <kbd className="px-2 py-1 border border-foreground/30 bg-muted/30 font-mono text-[10px] tracking-wider text-foreground min-w-[72px] text-center shrink-0">
                  {key}
                </kbd>
                <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground">{desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Status */}
        <div className="border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">SYS_STATUS:</span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            SETTINGS LOADED
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>
      </main>
    </div>
  )
}
