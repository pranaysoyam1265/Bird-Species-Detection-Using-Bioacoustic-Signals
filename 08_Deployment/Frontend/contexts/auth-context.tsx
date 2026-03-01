"use client"

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react"

type User = {
  id: number
  email: string
  name: string | null
  created_at: string
  updated_at: string
}

type AuthContextType = {
  user: User | null
  loading: boolean
  error: string | null
  login: (email: string, password: string) => Promise<boolean>
  signup: (email: string, password: string, confirmPassword: string, name?: string) => Promise<boolean>
  logout: () => Promise<void>
  clearError: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Check session on mount
  useEffect(() => {
    fetch("/api/auth/me")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.user) setUser(data.user)
      })
      .catch(() => { })
      .finally(() => setLoading(false))
  }, [])

  const login = useCallback(async (email: string, password: string): Promise<boolean> => {
    setError(null)
    setLoading(true)
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "LOGIN_FAILED")
        return false
      }
      setUser(data.user)
      return true
    } catch {
      setError("NETWORK_ERROR")
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  const signup = useCallback(
    async (email: string, password: string, confirmPassword: string, name?: string): Promise<boolean> => {
      setError(null)
      setLoading(true)
      try {
        const res = await fetch("/api/auth/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password, confirmPassword, name }),
        })
        const data = await res.json()
        if (!res.ok) {
          if (data.details) {
            // Flatten field errors into a single message
            const msgs = Object.values(data.details).flat() as string[]
            setError(msgs[0] || data.error)
          } else {
            setError(data.error || "SIGNUP_FAILED")
          }
          return false
        }
        setUser(data.user)
        return true
      } catch {
        setError("NETWORK_ERROR")
        return false
      } finally {
        setLoading(false)
      }
    },
    []
  )

  const logout = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" })
    setUser(null)
  }, [])

  const clearError = useCallback(() => setError(null), [])

  return (
    <AuthContext.Provider value={{ user, loading, error, login, signup, logout, clearError }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
