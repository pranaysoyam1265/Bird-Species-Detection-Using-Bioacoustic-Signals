"use client"

import { useState, useEffect, useRef } from "react"
import Link from "next/link"
import Image from "next/image"
import { useRouter } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import { MorphingLogo } from "@/components/morphing-logo"
import { ThemeToggle } from "@/components/theme-toggle"
import { useAuth } from "@/contexts/auth-context"
import { getAvatar } from "@/lib/avatar-store"
import { getDetections } from "@/lib/detection-store"
import {
  User,
  Bell,
  Search,
  Wifi,
  WifiOff,
  Settings,
  UserCircle,
  LogOut,
  BarChart3,

} from "lucide-react"

/* ─── helpers ─── */
function useClickOutside(ref: React.RefObject<HTMLElement | null>, cb: () => void) {
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) cb()
    }
    document.addEventListener("click", handler)
    return () => document.removeEventListener("click", handler)
  }, [ref, cb])
}

/* ─── component ─── */
export function Navbar() {
  const { user, logout } = useAuth()
  const router = useRouter()

  /* ── avatar ── */
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null)
  useEffect(() => { setAvatarUrl(getAvatar()) }, [])

  const initials = user?.name
    ? user.name.slice(0, 2).toUpperCase()
    : user?.email
      ? user.email.slice(0, 2).toUpperCase()
      : null

  /* ── greeting ── */
  const greeting = (() => {
    const hour = new Date().getHours()
    const g = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening"
    let name = user?.name || user?.email?.split("@")[0] || "Researcher"
    try {
      const saved = localStorage.getItem("birdsense-profile")
      if (saved) { const p = JSON.parse(saved); if (p.name) name = p.name }
    } catch { /* */ }
    return `${g}, ${name}`
  })()

  /* ── quick stats (for notification count) ── */
  const [detectionCount, setDetectionCount] = useState(0)
  useEffect(() => {
    setDetectionCount(getDetections().length)
  }, [])

  /* ── notifications (mock) ── */
  const [notifOpen, setNotifOpen] = useState(false)
  const notifRef = useRef<HTMLDivElement>(null)
  useClickOutside(notifRef, () => setNotifOpen(false))
  const notifCount = detectionCount > 0 ? Math.min(detectionCount, 9) : 0
  const notifications = [
    detectionCount > 0 && { text: `${detectionCount} detection(s) in history`, time: "recent" },
    { text: "EfficientNet-B2 model loaded", time: "system" },
    { text: "87 species available for detection", time: "system" },
  ].filter(Boolean) as { text: string; time: string }[]

  /* ── avatar dropdown ── */
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  useClickOutside(dropdownRef, () => setDropdownOpen(false))

  const handleLogout = async () => {
    setDropdownOpen(false)
    await logout()
    router.replace("/")
  }

  /* ── connection status ── */
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking")
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/api/auth/me", { method: "GET", cache: "no-store" })
        setApiStatus(res.ok || res.status === 401 ? "online" : "offline")
      } catch {
        setApiStatus("offline")
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  /* ── search (triggers existing command palette via Ctrl+K) ── */
  const handleSearch = () => {
    const ev = new KeyboardEvent("keydown", { key: "k", metaKey: true, ctrlKey: true, bubbles: true })
    document.dispatchEvent(ev)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="w-full px-4 pt-4 lg:px-6 lg:pt-6 relative z-50"
    >
      <nav className="w-full border border-foreground/20 bg-background/80 backdrop-blur-sm px-4 py-2.5 lg:px-6">
        <div className="flex items-center justify-between">
          {/* ── Left: Logo + Connection ── */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.4 }}
            className="flex items-center gap-3"
          >
            <Link href="/" className="flex items-center gap-2.5">
              <MorphingLogo size={32} />
              <span className="text-base font-mono tracking-[0.15em] uppercase font-bold hidden sm:inline">
                BirdSense
              </span>
            </Link>

            {/* Connection status */}
            <div className="hidden md:flex items-center gap-2 border border-foreground/15 px-3 py-1">
              {apiStatus === "online" ? (
                <Wifi size={13} className="text-[#22c55e]" />
              ) : apiStatus === "offline" ? (
                <WifiOff size={13} className="text-red-400" />
              ) : (
                <Wifi size={13} className="text-muted-foreground animate-pulse" />
              )}
              <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
                {apiStatus === "online" ? "API:ONLINE" : apiStatus === "offline" ? "API:OFFLINE" : "CHECKING..."}
              </span>
            </div>


          </motion.div>

          {/* ── Right: greeting + controls ── */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.4 }}
            className="flex items-center gap-2.5 sm:gap-3.5"
          >
            {/* Greeting */}
            {user && (
              <span className="hidden lg:block text-xs font-mono tracking-widest uppercase text-muted-foreground">
                {greeting}
              </span>
            )}

            <ThemeToggle />

            {/* Search button */}
            <button
              onClick={handleSearch}
              aria-label="Search"
              className="w-10 h-10 border border-foreground/20 flex items-center justify-center text-muted-foreground hover:text-accent hover:border-accent/40 cursor-pointer transition-none"
            >
              <Search size={18} />
            </button>

            {/* Notification bell */}
            {user && (
              <div ref={notifRef} className="relative">
                <button
                  onClick={() => setNotifOpen(!notifOpen)}
                  aria-label="Notifications"
                  className="w-10 h-10 border border-foreground/20 flex items-center justify-center text-muted-foreground hover:text-accent hover:border-accent/40 cursor-pointer transition-none relative"
                >
                  <Bell size={18} />
                  {notifCount > 0 && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-accent text-white text-[9px] font-bold flex items-center justify-center font-mono">
                      {notifCount}
                    </span>
                  )}
                </button>

                {/* Notification dropdown */}
                <AnimatePresence>
                  {notifOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -8, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -8, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className="absolute right-0 top-12 w-72 border-2 border-foreground bg-background shadow-xl z-50"
                    >
                      <div className="border-b border-foreground/20 px-4 py-2.5">
                        <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-accent font-bold">
                          Notifications
                        </span>
                      </div>
                      {notifications.map((n, i) => (
                        <div key={i} className="px-4 py-2.5 border-b border-foreground/10 last:border-b-0">
                          <p className="font-mono text-xs tracking-wider text-foreground">{n.text}</p>
                          <p className="font-mono text-[9px] tracking-wider text-muted-foreground uppercase mt-0.5">
                            {n.time}
                          </p>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}

            {/* Log in link for guests */}
            {!user && (
              <Link
                href="/login"
                className="hidden sm:block text-xs font-mono tracking-widest uppercase text-muted-foreground hover:text-foreground transition-colors duration-200"
              >
                Log In
              </Link>
            )}

            {/* Avatar with dropdown */}
            <div ref={dropdownRef} className="relative">
              {/* Avatar trigger — NOT a button wrapping the dropdown */}
              <div
                role="button"
                tabIndex={0}
                onClick={(e) => {
                  e.stopPropagation()
                  if (user) setDropdownOpen((prev) => !prev)
                  else router.push("/login")
                }}
                onKeyDown={(e) => { if (e.key === "Enter") { user ? setDropdownOpen((prev) => !prev) : router.push("/login") } }}
                aria-label={user ? "Open profile menu" : "Log in"}
                className="cursor-pointer"
              >
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-11 h-11 border-2 border-foreground/30 bg-background flex items-center justify-center hover:border-accent transition-colors duration-200 relative overflow-hidden"
                >
                  {/* Avatar Image/Initials */}
                  <div className="w-full h-full flex items-center justify-center">
                    {avatarUrl ? (
                      <Image src={avatarUrl} alt="Avatar" width={44} height={44} className="w-full h-full object-cover" />
                    ) : initials ? (
                      <span className="font-mono text-sm font-bold text-foreground tracking-wider">
                        {initials}
                      </span>
                    ) : (
                      <User size={20} className="text-foreground/60" />
                    )}
                  </div>

                  {/* Status dot */}
                  {user && (
                    <div className={`absolute bottom-0.5 right-0.5 w-2 h-2 rounded-full ring-2 ring-background ${apiStatus === "online"
                        ? "bg-[#22c55e]"
                        : apiStatus === "offline"
                          ? "bg-red-400"
                          : "bg-yellow-400 animate-pulse"
                      }`} />
                  )}
                </motion.div>
              </div>

              {/* Dropdown menu */}
              <AnimatePresence>
                {dropdownOpen && user && (
                  <motion.div
                    initial={{ opacity: 0, y: -8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -8, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-[52px] w-56 border-2 border-foreground bg-background shadow-xl z-[60] overflow-hidden pointer-events-auto"
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                  >
                    {/* User info */}
                    <div className="border-b border-foreground/20 px-4 py-3 bg-muted/5">
                      <p className="font-mono text-xs tracking-wider text-foreground font-bold truncate">
                        {(() => {
                          try {
                            const s = localStorage.getItem("birdsense-profile")
                            if (s) { const p = JSON.parse(s); if (p.name) return p.name }
                          } catch { /* */ }
                          return user.name || user.email
                        })()}
                      </p>
                      <p className="font-mono text-[10px] tracking-wider text-muted-foreground truncate mt-0.5">
                        {user.email}
                      </p>
                    </div>

                    {/* Menu items */}
                    <Link
                      href="/profile"
                      onClick={() => setDropdownOpen(false)}
                      className="w-full flex items-center gap-3 px-4 py-2.5 font-mono text-xs tracking-[0.15em] uppercase text-foreground hover:bg-accent/10 hover:text-accent cursor-pointer transition-none border-b border-foreground/10 no-underline"
                    >
                      <UserCircle size={16} />
                      Profile
                    </Link>
                    <Link
                      href="/settings"
                      onClick={() => setDropdownOpen(false)}
                      className="w-full flex items-center gap-3 px-4 py-2.5 font-mono text-xs tracking-[0.15em] uppercase text-foreground hover:bg-accent/10 hover:text-accent cursor-pointer transition-none border-b border-foreground/10 no-underline"
                    >
                      <Settings size={16} />
                      Settings
                    </Link>
                    <Link
                      href="/models"
                      onClick={() => setDropdownOpen(false)}
                      className="w-full flex items-center gap-3 px-4 py-2.5 font-mono text-xs tracking-[0.15em] uppercase text-foreground hover:bg-accent/10 hover:text-accent cursor-pointer transition-none border-b border-foreground/10 no-underline"
                    >
                      <BarChart3 size={16} />
                      Models
                    </Link>

                    <button
                      type="button"
                      onClick={() => {
                        setDropdownOpen(false)
                        handleLogout()
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 font-mono text-xs tracking-[0.15em] uppercase text-red-500 hover:bg-red-500/10 cursor-pointer transition-none text-left"
                    >
                      <LogOut size={16} />
                      Log Out
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>
      </nav>
    </motion.div>
  )
}
