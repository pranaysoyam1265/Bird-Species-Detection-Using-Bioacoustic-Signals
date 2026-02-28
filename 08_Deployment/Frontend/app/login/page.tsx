"use client"

import { useState, useEffect, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import Link from "next/link"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { Navbar } from "@/components/navbar"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { useAuth } from "@/contexts/auth-context"
import { loginSchema, signupSchema, type LoginInput, type SignupInput } from "@/lib/validations"

type AuthMode = "login" | "signup"

function LoginPageContent() {
  const [mode, setMode] = useState<AuthMode>("login")
  const isLogin = mode === "login"

  const router = useRouter()
  const searchParams = useSearchParams()
  const { login, signup, loading, error, clearError, user } = useAuth()
  // ── Login form ── (must be before any early returns)
  const loginForm = useForm<LoginInput>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: "", password: "" },
  })

  // ── Signup form ──
  const signupForm = useForm<SignupInput>({
    resolver: zodResolver(signupSchema),
    defaultValues: { email: "", password: "", confirmPassword: "", name: "" },
  })

  const [authChecked, setAuthChecked] = useState(false)

  // Wait for auth check then redirect if already logged in
  useEffect(() => {
    if (!loading) {
      setAuthChecked(true)
      if (user) {
        const redirect = searchParams.get("redirect") || "/"
        router.replace(redirect)
      }
    }
  }, [user, loading, router, searchParams])

  // Show nothing while auth is being checked to prevent flash
  if (!authChecked) return null

  const switchMode = (next: AuthMode) => {
    setMode(next)
    clearError()
    loginForm.reset()
    signupForm.reset()
  }

  const onLogin = loginForm.handleSubmit(async ({ email, password }) => {
    const ok = await login(email, password)
    if (ok) router.replace(searchParams.get("redirect") || "/")
  })

  const onSignup = signupForm.handleSubmit(async ({ email, password, confirmPassword, name }) => {
    const ok = await signup(email, password, confirmPassword, name)
    if (ok) router.replace("/")
  })

  const submitHandler = isLogin ? onLogin : onSignup
  const formErrors = isLogin ? loginForm.formState.errors : signupForm.formState.errors

  return (
    <div className="min-h-screen dot-grid-bg relative scanline-overlay">
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.25) 100%)" }}
      />
      <Navbar />

      <main className="w-full px-4 py-12 lg:px-12 lg:py-20">
        <div className="mx-auto max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-start">

          {/* ─── LEFT COLUMN ─── */}
          <div className="flex flex-col gap-8 pt-4 lg:pt-10">
            <div>
              <h1 className="font-pixel text-4xl sm:text-5xl lg:text-6xl tracking-tight text-foreground uppercase select-none">
                BIRDSENSE
              </h1>
              <p className="mt-3 text-xs sm:text-sm font-mono tracking-[0.2em] uppercase text-muted-foreground">
                {isLogin ? "// ACCESS_PORTAL // AUTH_REQUIRED" : "// REGISTRATION // NEW_ACCOUNT"}
              </p>
            </div>

            <div className="border-2 border-foreground bg-background p-5 font-mono text-xs sm:text-sm space-y-3">
              <div className="border-b border-foreground/30 pb-2 mb-3 text-[10px] sm:text-xs tracking-[0.25em] uppercase text-muted-foreground flex items-center gap-2">
                <span className="h-1.5 w-1.5 bg-accent" />
                SYS_NOTES
              </div>
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground select-none">&gt;</span>
                <span>SUPPORTED: <span className="text-foreground font-bold">EMAIL + PASSWORD</span></span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground select-none">&gt;</span>
                <span>MODE: <span className="text-foreground font-bold">{isLogin ? "DETECTION_DASHBOARD" : "ACCOUNT_CREATION"}</span></span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-muted-foreground select-none">&gt;</span>
                <span>STATUS: <span className="text-accent font-bold">READY</span></span>
              </div>
            </div>

            <div className="hidden lg:block font-mono text-muted-foreground/40 text-[10px] tracking-[0.5em] select-none">
              ─────────────────────────────
            </div>
          </div>

          {/* ─── RIGHT COLUMN: FORM ─── */}
          <div className="w-full">
            <div className="relative border-2 sm:border-4 border-foreground bg-background shadow-[6px_6px_0px_0px_hsl(var(--foreground))]">
              {/* Orange corner dots */}
              <span className="absolute -top-[3px] -left-[3px] w-2 h-2 bg-accent" />
              <span className="absolute -top-[3px] -right-[3px] w-2 h-2 bg-accent" />
              <span className="absolute -bottom-[3px] -left-[3px] w-2 h-2 bg-accent" />
              <span className="absolute -bottom-[3px] -right-[3px] w-2 h-2 bg-accent" />

              {/* Tabs */}
              <div className="border-b-2 border-foreground flex items-stretch justify-between">
                <div className="flex flex-1">
                  <button
                    type="button"
                    onClick={() => switchMode("login")}
                    className={`flex-1 px-5 py-3 font-mono text-xs sm:text-sm tracking-[0.2em] uppercase font-bold transition-none cursor-pointer select-none border-r-2 border-foreground ${isLogin ? "bg-accent text-white" : "bg-background text-foreground hover:bg-muted"
                      }`}
                  >
                    LOGIN
                  </button>
                  <button
                    type="button"
                    onClick={() => switchMode("signup")}
                    className={`flex-1 px-5 py-3 font-mono text-xs sm:text-sm tracking-[0.2em] uppercase font-bold transition-none cursor-pointer select-none ${!isLogin ? "bg-accent text-white" : "bg-background text-foreground hover:bg-muted"
                      }`}
                  >
                    SIGN UP
                  </button>
                </div>
                <div className="flex items-center gap-1.5 px-4 border-l-2 border-foreground">
                  <span className="w-2.5 h-2.5 border border-foreground" />
                  <span className="w-2.5 h-2.5 border border-foreground bg-accent" />
                  <span className="w-2.5 h-2.5 border border-foreground" />
                </div>
              </div>

              {/* Form */}
              <form onSubmit={submitHandler} className="p-5 sm:p-8 space-y-6">

                {/* API-level error banner */}
                {error && (
                  <div className="border-2 border-accent bg-accent/10 px-4 py-3 font-mono text-xs tracking-[0.15em] uppercase text-accent flex items-center gap-2">
                    <span className="select-none">[!]</span> {error}
                  </div>
                )}

                {/* NAME (Sign Up only) */}
                {!isLogin && (
                  <div className="space-y-2">
                    <Label className="font-mono text-xs tracking-[0.2em] uppercase font-bold">NAME</Label>
                    <Input
                      type="text"
                      placeholder="dr_ornithologist"
                      {...signupForm.register("name")}
                      className="rounded-none border-2 border-foreground bg-background font-mono text-sm h-11 placeholder:text-muted-foreground/50 focus-visible:ring-accent focus-visible:ring-offset-0"
                    />
                    {signupForm.formState.errors.name && (
                      <p className="text-[10px] font-mono text-accent tracking-[0.1em] uppercase">
                        {signupForm.formState.errors.name.message}
                      </p>
                    )}
                  </div>
                )}

                {/* EMAIL */}
                <div className="space-y-2">
                  <Label className="font-mono text-xs tracking-[0.2em] uppercase font-bold">EMAIL</Label>
                  <Input
                    type="email"
                    placeholder="researcher@birdsense.ai"
                    {...(isLogin ? loginForm.register("email") : signupForm.register("email"))}
                    className="rounded-none border-2 border-foreground bg-background font-mono text-sm h-11 placeholder:text-muted-foreground/50 focus-visible:ring-accent focus-visible:ring-offset-0"
                  />
                  {(formErrors as Record<string, { message?: string }>).email && (
                    <p className="text-[10px] font-mono text-accent tracking-[0.1em] uppercase">
                      {(formErrors as Record<string, { message?: string }>).email?.message}
                    </p>
                  )}
                </div>

                {/* PASSWORD */}
                <div className="space-y-2">
                  <Label className="font-mono text-xs tracking-[0.2em] uppercase font-bold">PASSWORD</Label>
                  <Input
                    type="password"
                    placeholder="••••••••••••"
                    {...(isLogin ? loginForm.register("password") : signupForm.register("password"))}
                    className="rounded-none border-2 border-foreground bg-background font-mono text-sm h-11 placeholder:text-muted-foreground/50 focus-visible:ring-accent focus-visible:ring-offset-0"
                  />
                  {(formErrors as Record<string, { message?: string }>).password && (
                    <p className="text-[10px] font-mono text-accent tracking-[0.1em] uppercase">
                      {(formErrors as Record<string, { message?: string }>).password?.message}
                    </p>
                  )}
                </div>

                {/* CONFIRM PASSWORD (Sign Up only) */}
                {!isLogin && (
                  <div className="space-y-2">
                    <Label className="font-mono text-xs tracking-[0.2em] uppercase font-bold">CONFIRM_PASSWORD</Label>
                    <Input
                      type="password"
                      placeholder="••••••••••••"
                      {...signupForm.register("confirmPassword")}
                      className="rounded-none border-2 border-foreground bg-background font-mono text-sm h-11 placeholder:text-muted-foreground/50 focus-visible:ring-accent focus-visible:ring-offset-0"
                    />
                    {signupForm.formState.errors.confirmPassword && (
                      <p className="text-[10px] font-mono text-accent tracking-[0.1em] uppercase">
                        {signupForm.formState.errors.confirmPassword.message}
                      </p>
                    )}
                  </div>
                )}

                {/* Remember me + Forgot (Login only) */}
                {isLogin && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="remember"
                        className="rounded-none border-2 border-foreground h-4 w-4 data-[state=checked]:bg-foreground data-[state=checked]:text-background"
                      />
                      <Label htmlFor="remember" className="font-mono text-[10px] sm:text-xs tracking-[0.15em] uppercase text-muted-foreground cursor-pointer select-none">
                        REMEMBER_ME
                      </Label>
                    </div>
                    <Link href="#" className="font-mono text-[10px] sm:text-xs tracking-[0.1em] uppercase text-muted-foreground hover:text-foreground underline underline-offset-4 decoration-1">
                      [FORGOT_PASSWORD?]
                    </Link>
                  </div>
                )}

                {/* Submit */}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full border-2 border-accent bg-accent text-white font-mono text-sm sm:text-base tracking-[0.2em] uppercase font-bold py-3.5 shadow-[4px_4px_0px_0px_rgba(234,88,12,0.3)] hover:bg-background hover:text-accent active:shadow-none active:translate-x-[4px] active:translate-y-[4px] transition-none cursor-pointer select-none disabled:opacity-60 disabled:cursor-not-allowed disabled:active:translate-x-0 disabled:active:translate-y-0"
                >
                  {loading
                    ? ">>> PROCESSING... <<<"
                    : isLogin
                      ? ">>> AUTHENTICATE <<<"
                      : ">>> CREATE_ACCOUNT <<<"}
                </button>

                {/* Mode toggle */}
                <div className="text-center font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground">
                  {isLogin ? (
                    <>NO_ACCOUNT?{" "}
                      <button type="button" onClick={() => switchMode("signup")} className="text-accent hover:text-foreground underline underline-offset-4 decoration-1 font-bold cursor-pointer">
                        [SIGN_UP]
                      </button>
                    </>
                  ) : (
                    <>HAVE_ACCOUNT?{" "}
                      <button type="button" onClick={() => switchMode("login")} className="text-accent hover:text-foreground underline underline-offset-4 decoration-1 font-bold cursor-pointer">
                        [LOGIN]
                      </button>
                    </>
                  )}
                </div>

                {/* Back to home */}
                <div className="text-center">
                  <Link href="/" className="inline-block font-mono text-xs tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground transition-none">
                    [← BACK_TO_HOME]
                  </Link>
                </div>

                {/* Status bar */}
                <div className="border border-foreground/30 bg-muted/30 px-4 py-3 font-mono text-[10px] sm:text-xs tracking-[0.15em] uppercase text-muted-foreground">
                  <span className="mr-2 select-none">AUTH_STATUS:</span>
                  <span className="inline-flex items-center gap-1">
                    {loading
                      ? "[ ] PROCESSING..."
                      : isLogin
                        ? "[ ] WAITING_FOR_INPUT"
                        : "[ ] REGISTRATION_MODE"}
                    <span className="inline-block w-1.5 h-3.5 bg-accent animate-blink" />
                  </span>
                </div>

              </form>
            </div>
          </div>

        </div>
      </main>
    </div>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-background flex items-center justify-center">
        <span className="font-mono text-xs tracking-[0.2em] uppercase text-muted-foreground animate-pulse">
          LOADING...
        </span>
      </div>
    }>
      <LoginPageContent />
    </Suspense>
  )
}
