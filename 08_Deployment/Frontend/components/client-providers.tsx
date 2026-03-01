"use client"

import { CommandPalette } from "@/components/command-palette"
import { KeyboardShortcutsToast } from "@/components/keyboard-shortcuts-toast"
import { AccentColorProvider } from "@/components/accent-color-provider"

export function ClientProviders({ children }: { children: React.ReactNode }) {
  return (
    <>
      <AccentColorProvider />
      {children}
      <CommandPalette />
      <KeyboardShortcutsToast />
    </>
  )
}
