import bcrypt from "bcryptjs"
import jwt from "jsonwebtoken"
import { cookies } from "next/headers"
import { findUserById, type SafeUser } from "./db"

const JWT_SECRET = process.env.JWT_SECRET || "birdsense-dev-secret"
const TOKEN_EXPIRY = "7d"
const COOKIE_NAME = "birdsense_token"

// ── Runtime guard: warn loudly if default secret is used in production ──
if (
  process.env.NODE_ENV === "production" &&
  (JWT_SECRET === "birdsense-dev-secret" || JWT_SECRET.length < 16)
) {
  console.error(
    "\n⚠️  CRITICAL SECURITY WARNING ⚠️\n" +
    "JWT_SECRET is using the default dev value or is too short.\n" +
    "Set a strong, random JWT_SECRET (≥32 chars) in your environment.\n"
  )
}

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 12)
}

export async function comparePassword(
  password: string,
  hash: string
): Promise<boolean> {
  return bcrypt.compare(password, hash)
}

export interface TokenPayload {
  userId: number;
  email: string;
  name?: string | null;
}

export function signToken(user: { id: number; email: string; name?: string | null }): string {
  return jwt.sign(
    { userId: user.id, email: user.email, name: user.name },
    JWT_SECRET,
    { expiresIn: TOKEN_EXPIRY }
  )
}

export function verifyToken(token: string): TokenPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as TokenPayload
  } catch {
    return null
  }
}

export async function setAuthCookie(user: { id: number; email: string; name?: string | null }): Promise<void> {
  const token = signToken(user)
  const cookieStore = await cookies()
  cookieStore.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 7, // 7 days
  })
}

export async function clearAuthCookie(): Promise<void> {
  const cookieStore = await cookies()
  cookieStore.delete(COOKIE_NAME)
}

export async function getSession(): Promise<SafeUser | null> {
  const cookieStore = await cookies()
  const token = cookieStore.get(COOKIE_NAME)?.value
  if (!token) return null

  const payload = verifyToken(token)
  if (!payload) return null

  // In split architecture, we trust the JWT payload to avoid DB lookups on the frontend
  return {
    id: payload.userId,
    email: payload.email,
    name: payload.name || null,
    created_at: new Date().toISOString(), // Mocked as we don't have it in token
    updated_at: new Date().toISOString(),
  }
}
