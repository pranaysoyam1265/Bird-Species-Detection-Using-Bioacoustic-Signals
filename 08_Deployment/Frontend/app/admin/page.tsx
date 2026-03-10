"use client"

import { useState, useEffect } from "react"
import { Navbar } from "@/components/navbar"
import { NavSidebar } from "@/components/nav-sidebar"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"
import {
  Shield,
  Users,
  Bird,
  Activity,
  Trash2,
  ChevronUp,
  ChevronDown,
  Crown,
  UserMinus,
} from "lucide-react"
import { useToast } from "@/hooks/use-toast"

type PlatformUser = {
  id: number
  email: string
  name: string | null
  role: string
  detection_count: number
  created_at: string
}

type PlatformStats = {
  user_count: number
  detection_count: number
  top_species: string
  recent_detections: {
    id: string
    filename: string
    top_species: string
    top_confidence: number
    date: string
    time: string
    user_email: string
  }[]
}

export default function AdminPage() {
  const { user, loading, isAdmin } = useAuth()
  const router = useRouter()
  const { toast } = useToast()

  const [stats, setStats] = useState<PlatformStats | null>(null)
  const [users, setUsers] = useState<PlatformUser[]>([])
  const [loadingData, setLoadingData] = useState(true)

  // Auth guard: only admins allowed
  useEffect(() => {
    if (!loading && (!user || !isAdmin)) {
      router.replace("/")
    }
  }, [loading, user, isAdmin, router])

  // Fetch data
  useEffect(() => {
    if (!user || !isAdmin) return

    async function fetchData() {
      try {
        const [statsRes, usersRes] = await Promise.all([
          fetch("/api/admin?endpoint=stats"),
          fetch("/api/admin?endpoint=users"),
        ])

        if (statsRes.ok) {
          setStats(await statsRes.json())
        }
        if (usersRes.ok) {
          const data = await usersRes.json()
          setUsers(data.users || [])
        }
      } catch (err) {
        toast({ title: "Failed to load admin data" })
      } finally {
        setLoadingData(false)
      }
    }

    fetchData()
  }, [user, isAdmin, toast])

  const handleRoleChange = async (targetId: number, newRole: string) => {
    try {
      const res = await fetch("/api/admin", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targetId, role: newRole }),
      })
      if (!res.ok) {
        const err = await res.json()
        toast({ title: "Failed", description: err.error || "Could not update role" })
        return
      }
      setUsers((prev) =>
        prev.map((u) => (u.id === targetId ? { ...u, role: newRole } : u))
      )
      toast({ title: `User role updated to ${newRole}` })
    } catch {
      toast({ title: "Network error" })
    }
  }

  const handleDeleteUser = async (targetId: number, email: string) => {
    if (!confirm(`Permanently delete user "${email}"? This cannot be undone.`)) return
    try {
      const res = await fetch(`/api/admin?targetId=${targetId}`, { method: "DELETE" })
      if (!res.ok) {
        const err = await res.json()
        toast({ title: "Failed", description: err.error || "Could not delete user" })
        return
      }
      setUsers((prev) => prev.filter((u) => u.id !== targetId))
      toast({ title: `User "${email}" deleted` })
    } catch {
      toast({ title: "Network error" })
    }
  }

  if (loading || !user || !isAdmin) {
    return (
      <div className="min-h-screen dot-grid-bg flex items-center justify-center scanline-overlay">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent animate-spin" />
          <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground">
            VERIFYING ACCESS...
          </span>
        </div>
      </div>
    )
  }

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
            <Shield size={16} className="text-accent" />
            <h1 className="font-mono text-base sm:text-lg tracking-[0.2em] uppercase text-foreground font-bold">
              ADMIN DASHBOARD
            </h1>
          </div>
          <p className="font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground">
            PLATFORM MANAGEMENT & USER CONTROL
          </p>
        </div>
      </div>

      <main className="flex-1 px-4 lg:px-6 py-6 lg:py-8 relative z-10 max-w-5xl mx-auto w-full space-y-6">

        {/* ────── PLATFORM STATS ────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            {
              icon: Users,
              label: "TOTAL USERS",
              value: stats?.user_count?.toString() ?? "—",
              color: "var(--accent-hex)",
            },
            {
              icon: Activity,
              label: "TOTAL DETECTIONS",
              value: stats?.detection_count?.toString() ?? "—",
              color: "#22c55e",
            },
            {
              icon: Bird,
              label: "TOP SPECIES",
              value: stats?.top_species ?? "—",
              color: "#f59e0b",
              small: true,
            },
            {
              icon: Crown,
              label: "YOUR ROLE",
              value: "ADMIN",
              color: "#8b5cf6",
            },
          ].map((stat) => (
            <div key={stat.label} className="border-2 border-foreground bg-background p-4 relative">
              <span
                className="absolute top-0 left-0 w-full h-0.5"
                style={{ backgroundColor: stat.color, opacity: 0.4 }}
              />
              <div className="flex items-center gap-2 mb-2">
                <stat.icon size={14} style={{ color: stat.color }} />
                <span className="font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground">
                  {stat.label}
                </span>
              </div>
              <span
                className={`font-mono font-bold text-foreground block ${stat.small ? "text-xs sm:text-sm truncate" : "text-xl sm:text-2xl"
                  }`}
              >
                {stat.value}
              </span>
            </div>
          ))}
        </div>

        {/* ────── USER MANAGEMENT ────── */}
        <div className="border-2 border-foreground bg-background">
          <div className="border-b-2 border-foreground px-4 py-2.5 bg-muted/30 flex items-center justify-between">
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
              USER MANAGEMENT
            </span>
            <span className="font-mono text-[9px] tracking-wider text-muted-foreground">
              {users.length} registered
            </span>
          </div>

          {loadingData ? (
            <div className="p-8 flex justify-center">
              <div className="w-6 h-6 border-2 border-accent border-t-transparent animate-spin" />
            </div>
          ) : users.length === 0 ? (
            <div className="p-8 text-center">
              <span className="font-mono text-xs text-muted-foreground tracking-wider">
                No users found
              </span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-foreground/20">
                    <th className="text-left px-4 py-2.5 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground font-bold">
                      User
                    </th>
                    <th className="text-left px-4 py-2.5 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground font-bold">
                      Role
                    </th>
                    <th className="text-left px-4 py-2.5 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground font-bold">
                      Detections
                    </th>
                    <th className="text-left px-4 py-2.5 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground font-bold">
                      Joined
                    </th>
                    <th className="text-right px-4 py-2.5 font-mono text-[9px] tracking-[0.2em] uppercase text-muted-foreground font-bold">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr
                      key={u.id}
                      className="border-b border-foreground/10 last:border-b-0 hover:bg-muted/20"
                    >
                      <td className="px-4 py-3">
                        <span className="font-mono text-xs font-bold text-foreground block">
                          {u.name || u.email.split("@")[0]}
                        </span>
                        <span className="font-mono text-[9px] text-muted-foreground block">
                          {u.email}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 font-mono text-[9px] tracking-wider uppercase font-bold ${u.role === "admin"
                              ? "bg-accent/15 text-accent border border-accent/30"
                              : "bg-muted text-muted-foreground border border-foreground/20"
                            }`}
                        >
                          {u.role === "admin" && <Crown size={9} />}
                          {u.role}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-mono text-sm font-bold text-foreground">
                          {u.detection_count}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-mono text-[10px] text-muted-foreground">
                          {u.created_at
                            ? new Date(u.created_at).toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                              year: "numeric",
                            })
                            : "—"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        {u.id !== user.id && (
                          <div className="flex items-center gap-2 justify-end">
                            {u.role === "user" ? (
                              <button
                                type="button"
                                onClick={() => handleRoleChange(u.id, "admin")}
                                className="flex items-center gap-1 px-2 py-1 border border-accent/40 font-mono text-[8px] tracking-wider uppercase text-accent hover:bg-accent/10 cursor-pointer transition-none"
                                title="Promote to Admin"
                              >
                                <ChevronUp size={10} />
                                Promote
                              </button>
                            ) : (
                              <button
                                type="button"
                                onClick={() => handleRoleChange(u.id, "user")}
                                className="flex items-center gap-1 px-2 py-1 border border-foreground/30 font-mono text-[8px] tracking-wider uppercase text-muted-foreground hover:bg-muted cursor-pointer transition-none"
                                title="Demote to User"
                              >
                                <ChevronDown size={10} />
                                Demote
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => handleDeleteUser(u.id, u.email)}
                              className="flex items-center gap-1 px-2 py-1 border border-red-500/30 font-mono text-[8px] tracking-wider uppercase text-red-500 hover:bg-red-500/10 cursor-pointer transition-none"
                              title="Delete User"
                            >
                              <Trash2 size={10} />
                            </button>
                          </div>
                        )}
                        {u.id === user.id && (
                          <span className="font-mono text-[8px] tracking-wider uppercase text-accent/50">
                            YOU
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ────── RECENT ACTIVITY ────── */}
        <div className="border-2 border-foreground bg-background">
          <div className="border-b-2 border-foreground px-4 py-2.5 bg-muted/30">
            <span className="font-mono text-[10px] tracking-[0.25em] uppercase text-accent font-bold">
              RECENT PLATFORM ACTIVITY
            </span>
          </div>
          <div className="divide-y divide-foreground/10">
            {!stats?.recent_detections?.length ? (
              <div className="p-6 text-center">
                <span className="font-mono text-xs text-muted-foreground tracking-wider">
                  No detections yet across the platform
                </span>
              </div>
            ) : (
              stats.recent_detections.map((d) => (
                <div key={d.id} className="flex items-center gap-4 px-4 py-3">
                  <div className="w-8 h-8 border border-foreground/20 flex items-center justify-center shrink-0">
                    <Bird size={14} className="text-accent" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className="font-mono text-xs font-bold text-foreground block truncate">
                      {d.top_species}
                    </span>
                    <span className="font-mono text-[9px] text-muted-foreground block">
                      {d.filename} — {d.user_email} — {d.date}
                    </span>
                  </div>
                  <span className="font-mono text-xs font-bold text-accent shrink-0">
                    {(d.top_confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* ────── STATUS BAR ────── */}
        <div className="border border-foreground/30 bg-muted/30 px-4 py-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-accent/60">
            SYS_STATUS:
          </span>
          <span className="font-mono text-[10px] tracking-[0.15em] uppercase text-foreground font-bold">
            ADMIN PANEL ACTIVE • {users.length} USERS • {stats?.detection_count ?? 0} DETECTIONS
          </span>
          <span className="inline-block w-1.5 h-3 bg-accent animate-blink" />
        </div>

      </main>
    </div>
  )
}
