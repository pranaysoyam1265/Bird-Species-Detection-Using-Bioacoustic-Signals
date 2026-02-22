"use client"

import Link from "next/link"
import { Navbar } from "@/components/navbar"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"

export default function LoginPage() {
  return (
    <div className="min-h-screen dot-grid-bg">
      <Navbar />

      <main className="w-full px-4 py-12 lg:px-12 lg:py-20">
        <div className="mx-auto max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-start">

          {/* ─── LEFT COLUMN: IDENTITY / MARKETING ─── */}
          <div className="flex flex-col gap-8 pt-4 lg:pt-10">
            {/* Title */}
            <div>
              <h1 className="font-pixel text-4xl sm:text-5xl lg:text-6xl tracking-tight text-foreground uppercase select-none">
                BIRDSENSE
              </h1>
              <p className="mt-3 text-xs sm:text-sm font-mono tracking-[0.2em] uppercase text-muted-foreground">
                {"// ACCESS_PORTAL // AUTH_REQUIRED"}
              </p>
            </div>

            {/* System notes terminal box */}
            <div className="border-2 border-foreground bg-background p-5 font-mono text-xs sm:text-sm space-y-3">
              <div className="border-b border-foreground/30 pb-2 mb-3 text-[10px] sm:text-xs tracking-[0.25em] uppercase text-muted-foreground">
                SYS_NOTES
              </div>
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground select-none">&gt;</span>
                <span>
                  SUPPORTED: <span className="text-foreground font-bold">EMAIL + PASSWORD</span>
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground select-none">&gt;</span>
                <span>
                  MODE: <span className="text-foreground font-bold">DETECTION_DASHBOARD</span>
                </span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground select-none">&gt;</span>
                <span>
                  STATUS: <span className="text-accent font-bold">READY</span>
                </span>
              </div>
            </div>

            {/* Decorative separator */}
            <div className="hidden lg:block font-mono text-muted-foreground/40 text-[10px] tracking-[0.5em] select-none">
              ─────────────────────────────
            </div>
          </div>

          {/* ─── RIGHT COLUMN: LOGIN FORM ─── */}
          <div className="w-full">
            <div className="border-2 sm:border-4 border-foreground bg-background shadow-[6px_6px_0px_0px_hsl(var(--foreground))]">

              {/* Header bar */}
              <div className="border-b-2 border-foreground px-5 py-3 flex items-center justify-between">
                <span className="font-mono text-xs sm:text-sm tracking-[0.2em] uppercase font-bold">
                  LOGIN_TERMINAL
                </span>
                <div className="flex gap-1.5">
                  <span className="w-2.5 h-2.5 border border-foreground" />
                  <span className="w-2.5 h-2.5 border border-foreground bg-foreground" />
                  <span className="w-2.5 h-2.5 border border-foreground" />
                </div>
              </div>

              {/* Form body */}
              <div className="p-5 sm:p-8 space-y-6">

                {/* EMAIL field */}
                <div className="space-y-2">
                  <Label className="font-mono text-xs tracking-[0.2em] uppercase font-bold">
                    EMAIL
                  </Label>
                  <Input
                    type="email"
                    placeholder="researcher@birdsense.ai"
                    className="rounded-none border-2 border-foreground bg-background font-mono text-sm h-11 placeholder:text-muted-foreground/50 focus-visible:ring-accent focus-visible:ring-offset-0"
                  />
                </div>

                {/* PASSWORD field */}
                <div className="space-y-2">
                  <Label className="font-mono text-xs tracking-[0.2em] uppercase font-bold">
                    PASSWORD
                  </Label>
                  <Input
                    type="password"
                    placeholder="••••••••••••"
                    className="rounded-none border-2 border-foreground bg-background font-mono text-sm h-11 placeholder:text-muted-foreground/50 focus-visible:ring-accent focus-visible:ring-offset-0"
                  />
                </div>

                {/* Remember me + Forgot password row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="remember"
                      className="rounded-none border-2 border-foreground h-4 w-4 data-[state=checked]:bg-foreground data-[state=checked]:text-background"
                    />
                    <Label
                      htmlFor="remember"
                      className="font-mono text-[10px] sm:text-xs tracking-[0.15em] uppercase text-muted-foreground cursor-pointer select-none"
                    >
                      REMEMBER_ME
                    </Label>
                  </div>
                  <Link
                    href="#"
                    className="font-mono text-[10px] sm:text-xs tracking-[0.1em] uppercase text-muted-foreground hover:text-foreground underline underline-offset-4 decoration-1"
                  >
                    [FORGOT_PASSWORD?]
                  </Link>
                </div>

                {/* Submit button */}
                <button
                  type="button"
                  className="w-full border-2 border-foreground bg-foreground text-background font-mono text-sm sm:text-base tracking-[0.2em] uppercase font-bold py-3.5 shadow-[4px_4px_0px_0px_hsl(var(--foreground)/0.3)] hover:bg-background hover:text-foreground active:shadow-none active:translate-x-[4px] active:translate-y-[4px] transition-none cursor-pointer select-none"
                >
                  &gt;&gt;&gt; AUTHENTICATE &lt;&lt;&lt;
                </button>

                {/* Back to home */}
                <div className="text-center">
                  <Link
                    href="/"
                    className="inline-block font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground transition-none"
                  >
                    [← BACK_TO_HOME]
                  </Link>
                </div>

                {/* AUTH_STATUS terminal box */}
                <div className="border border-foreground/30 bg-muted/30 px-4 py-3 font-mono text-[10px] sm:text-xs tracking-[0.15em] uppercase text-muted-foreground">
                  <span className="mr-2 select-none">AUTH_STATUS:</span>
                  <span className="inline-flex items-center gap-1">
                    [ ] WAITING_FOR_INPUT
                    <span className="inline-block w-1.5 h-3.5 bg-foreground/60 animate-blink" />
                  </span>
                </div>

              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}
