"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { usePathname } from "next/navigation"
import Link from "next/link"
import {
  Menu,
  X,
  Home,
  Mic,
  History,
  Bird,
  User,
  Settings,
  ChevronRight,
} from "lucide-react"

interface NavLink {
  href: string
  label: string
  description: string
  icon: React.ReactNode
}

const NAV_LINKS: NavLink[] = [
  {
    href: "/",
    label: "HOME",
    description: "Landing page",
    icon: <Home size={16} />,
  },
  {
    href: "/analyze",
    label: "ANALYZE",
    description: "Detect bird species",
    icon: <Mic size={16} />,
  },
  {
    href: "/results",
    label: "RESULTS",
    description: "Detection history",
    icon: <History size={16} />,
  },
  {
    href: "/species",
    label: "SPECIES",
    description: "Browse 87 species",
    icon: <Bird size={16} />,
  },
  {
    href: "/settings",
    label: "SETTINGS",
    description: "Preferences",
    icon: <Settings size={16} />,
  },
]

interface NavSidebarProps {
  className?: string
}

export function NavSidebar({ className }: NavSidebarProps) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const pathname = usePathname()

  // Close on route change
  useEffect(() => {
    setOpen(false)
  }, [pathname])

  // Close on ESC
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false)
    }
    if (open) {
      document.addEventListener("keydown", onKey)
      return () => document.removeEventListener("keydown", onKey)
    }
  }, [open])

  // Close on click outside
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener("mousedown", onClick)
      return () => document.removeEventListener("mousedown", onClick)
    }
  }, [open])

  const toggle = useCallback(() => setOpen((prev) => !prev), [])

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/"
    return pathname.startsWith(href)
  }

  return (
    <div className={`relative z-[999] ${className ?? ""}`}>
      {/* Hamburger Button */}
      <button
        ref={buttonRef}
        type="button"
        onClick={toggle}
        aria-label={open ? "Close navigation" : "Open navigation"}
        aria-expanded={open}
        className={`
          w-10 h-10 border-2 border-foreground flex items-center justify-center
          font-mono cursor-pointer transition-none
          ${open
            ? "border-accent text-accent"
            : "text-muted-foreground hover:border-accent hover:text-accent"
          }
        `}
      >
        {open ? <X size={20} strokeWidth={2.5} /> : <Menu size={20} strokeWidth={2.5} />}
      </button>

      {/* Backdrop overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-black/40 z-40"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Dropdown menu */}
      {open && (
        <div
          ref={menuRef}
          className="
            absolute top-[calc(100%+8px)] left-0
            w-[320px] max-w-[calc(100vw-2rem)]
            border-2 border-foreground bg-background
            shadow-[6px_6px_0px_0px_rgba(234,88,12,0.25)]
            z-50 flex flex-col
            animate-in slide-in-from-top-2 duration-150
          "
        >
          {/* Header */}
          <div className="border-b-2 border-foreground px-4 py-3 flex items-center justify-between">
            <span className="font-mono text-xs tracking-[0.25em] uppercase text-accent font-bold">
              NAVIGATION
            </span>
            <span className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
              v1.0
            </span>
          </div>

          {/* Links */}
          <nav className="flex flex-col">
            {NAV_LINKS.map((link) => {
              const active = isActive(link.href)
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`
                    group flex items-center gap-3 px-4 py-3.5
                    border-b border-foreground/10
                    transition-none cursor-pointer relative
                    ${active
                      ? "bg-accent/10 border-l-4 border-l-accent"
                      : "border-l-4 border-l-transparent hover:bg-muted/50 hover:border-l-accent/50"
                    }
                  `}
                >
                  {/* Icon */}
                  <div
                    className={`
                      w-8 h-8 border-2 flex items-center justify-center shrink-0
                      ${active
                        ? "border-accent text-accent bg-accent/10"
                        : "border-foreground/30 text-foreground/60 group-hover:border-accent/60 group-hover:text-accent"
                      }
                    `}
                  >
                    {link.icon}
                  </div>

                  {/* Label + description */}
                  <div className="flex-1 min-w-0">
                    <span
                      className={`
                        font-mono text-xs tracking-[0.2em] uppercase block font-bold
                        ${active ? "text-accent" : "text-foreground group-hover:text-accent"}
                      `}
                    >
                      {link.label}
                    </span>
                    <span className="font-mono text-[10px] tracking-[0.1em] text-muted-foreground block mt-0.5">
                      {link.description}
                    </span>
                  </div>

                  {/* Arrow indicator */}
                  <ChevronRight
                    size={14}
                    className={`
                      shrink-0 transition-none
                      ${active
                        ? "text-accent"
                        : "text-foreground/20 group-hover:text-accent/60"
                      }
                    `}
                  />

                  {/* Active dot */}
                  {active && (
                    <span className="absolute right-3 top-3 w-1.5 h-1.5 bg-accent rounded-full" />
                  )}
                </Link>
              )
            })}
          </nav>

          {/* Footer — System Status */}
          <div className="border-t-2 border-foreground px-4 py-3 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#22c55e] rounded-full animate-pulse" />
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-foreground font-bold">
                SYSTEM ONLINE
              </span>
            </div>
            <p className="font-mono text-[9px] tracking-[0.15em] uppercase text-muted-foreground">
              87 SPECIES • 96.06% ACC • MODEL v3.2
            </p>
          </div>

          {/* Corner accents */}
          <span className="absolute -bottom-[3px] -left-[3px] w-2 h-2 bg-accent" />
          <span className="absolute -bottom-[3px] -right-[3px] w-2 h-2 bg-accent" />
        </div>
      )}
    </div>
  )
}
