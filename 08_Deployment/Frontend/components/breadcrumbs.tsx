"use client"

import Link from "next/link"
import { ChevronRight } from "lucide-react"

interface Crumb {
  label: string
  href?: string
}

interface BreadcrumbsProps {
  items: Crumb[]
}

export function Breadcrumbs({ items }: BreadcrumbsProps) {
  return (
    <nav className="flex items-center gap-1 font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground overflow-x-auto">
      {items.map((crumb, i) => (
        <span key={i} className="flex items-center gap-1 shrink-0">
          {i > 0 && <ChevronRight size={10} className="text-foreground/20" />}
          {crumb.href ? (
            <Link
              href={crumb.href}
              className="hover:text-accent cursor-pointer transition-none"
            >
              {crumb.label}
            </Link>
          ) : (
            <span className="text-foreground font-bold">{crumb.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}
