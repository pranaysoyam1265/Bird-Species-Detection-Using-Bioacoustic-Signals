import Database from "better-sqlite3"
import path from "path"

const DB_PATH = path.join(process.cwd(), "birdsense.db")

let db: Database.Database | null = null

export function getDb(): Database.Database {
  if (db) return db

  // ── Vercel / Serverless Safety ──
  // better-sqlite3 cannot write to the read-only filesystem on Vercel.
  // In a split architecture, we rely on the backend for persistence.
  if (process.env.VERCEL || process.env.NODE_ENV === "production") {
    console.warn("[db] SQLite is not available in production serverless environments. Use the Backend API.")
    // Return a dummy object or throw a helpful error if hit
    throw new Error("Local SQLite is disabled in production. Use Backend API.")
  }

  if (!db) {
    db = new Database(DB_PATH)
    db.pragma("journal_mode = WAL")
    db.pragma("foreign_keys = ON")

    // ── Users table ──
    db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // ── Detections table (replaces localStorage) ──
    db.exec(`
      CREATE TABLE IF NOT EXISTS detections (
        id              TEXT PRIMARY KEY,
        user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        filename        TEXT NOT NULL,
        date            TEXT NOT NULL,
        time            TEXT NOT NULL,
        duration        INTEGER NOT NULL,
        top_species     TEXT NOT NULL,
        top_scientific  TEXT NOT NULL,
        top_confidence  REAL NOT NULL,
        predictions     TEXT NOT NULL,
        segments        TEXT NOT NULL,
        audio_url       TEXT,
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // ── User settings table ──
    db.exec(`
      CREATE TABLE IF NOT EXISTS user_settings (
        user_id       INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        settings_json TEXT NOT NULL DEFAULT '{}',
        updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)

    // ── API Keys table ──
    db.exec(`
      CREATE TABLE IF NOT EXISTS api_keys (
        id          TEXT PRIMARY KEY,
        user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        name        TEXT NOT NULL,
        key_hash    TEXT NOT NULL,
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `)
  }
  return db
}

// ── User types & queries ──────────────────────────────────────

export type User = {
  id: number
  email: string
  password: string
  name: string | null
  created_at: string
  updated_at: string
}

export type SafeUser = Omit<User, "password">

export function findUserByEmail(email: string): User | undefined {
  return getDb().prepare("SELECT * FROM users WHERE email = ?").get(email) as User | undefined
}

export function findUserById(id: number): SafeUser | undefined {
  return getDb()
    .prepare("SELECT id, email, name, created_at, updated_at FROM users WHERE id = ?")
    .get(id) as SafeUser | undefined
}

export function findUserByIdFull(id: number): User | undefined {
  return getDb().prepare("SELECT * FROM users WHERE id = ?").get(id) as User | undefined
}

export function updateUserPassword(id: number, hashedPassword: string): void {
  getDb().prepare("UPDATE users SET password = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?").run(hashedPassword, id)
}

export function createUser(email: string, hashedPassword: string, name?: string): SafeUser {
  const stmt = getDb().prepare(
    "INSERT INTO users (email, password, name) VALUES (?, ?, ?)"
  )
  const result = stmt.run(email, hashedPassword, name || null)
  return findUserById(result.lastInsertRowid as number)!
}

// ── Detection types & queries ─────────────────────────────────

export type DetectionRow = {
  id: string
  user_id: number
  filename: string
  date: string
  time: string
  duration: number
  top_species: string
  top_scientific: string
  top_confidence: number
  predictions: string   // JSON string
  segments: string      // JSON string
  audio_url: string | null
  created_at: string
}

export function getDetectionsByUser(
  userId: number,
  limit = 50,
  offset = 0,
): DetectionRow[] {
  return getDb()
    .prepare(
      `SELECT * FROM detections WHERE user_id = ?
       ORDER BY date DESC, time DESC LIMIT ? OFFSET ?`
    )
    .all(userId, limit, offset) as DetectionRow[]
}

export function getDetectionById(userId: number, id: string): DetectionRow | undefined {
  return getDb()
    .prepare("SELECT * FROM detections WHERE id = ? AND user_id = ?")
    .get(id, userId) as DetectionRow | undefined
}

export function insertDetection(row: Omit<DetectionRow, "created_at">): void {
  getDb()
    .prepare(
      `INSERT INTO detections
         (id, user_id, filename, date, time, duration,
          top_species, top_scientific, top_confidence,
          predictions, segments, audio_url)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      row.id, row.user_id, row.filename, row.date, row.time, row.duration,
      row.top_species, row.top_scientific, row.top_confidence,
      row.predictions, row.segments, row.audio_url ?? null,
    )
}

export function deleteDetectionById(userId: number, id: string): boolean {
  const result = getDb()
    .prepare("DELETE FROM detections WHERE id = ? AND user_id = ?")
    .run(id, userId)
  return result.changes > 0
}

export function deleteDetectionsByUser(userId: number, ids: string[]): number {
  const placeholders = ids.map(() => "?").join(",")
  const result = getDb()
    .prepare(`DELETE FROM detections WHERE user_id = ? AND id IN (${placeholders})`)
    .run(userId, ...ids)
  return result.changes
}

export function countDetectionsByUser(userId: number): number {
  const row = getDb()
    .prepare("SELECT COUNT(*) as count FROM detections WHERE user_id = ?")
    .get(userId) as { count: number }
  return row.count
}

// ── Settings queries ──────────────────────────────────────────

export function getUserSettings(userId: number): Record<string, unknown> {
  const row = getDb()
    .prepare("SELECT settings_json FROM user_settings WHERE user_id = ?")
    .get(userId) as { settings_json: string } | undefined
  return row ? JSON.parse(row.settings_json) : {}
}

export function upsertUserSettings(userId: number, settings: Record<string, unknown>): void {
  getDb()
    .prepare(
      `INSERT INTO user_settings (user_id, settings_json, updated_at)
       VALUES (?, ?, CURRENT_TIMESTAMP)
       ON CONFLICT(user_id) DO UPDATE SET
         settings_json = excluded.settings_json,
         updated_at = CURRENT_TIMESTAMP`
    )
    .run(userId, JSON.stringify(settings))
}

// ── API Key types & queries ───────────────────────────────────

export type ApiKeyRow = {
  id: string
  user_id: number
  name: string
  key_hash: string
  created_at: string
}

export function getApiKeysByUser(userId: number): Omit<ApiKeyRow, "key_hash">[] {
  return getDb()
    .prepare("SELECT id, user_id, name, created_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC")
    .all(userId) as Omit<ApiKeyRow, "key_hash">[]
}

export function insertApiKey(userId: number, id: string, name: string, keyHash: string): void {
  getDb()
    .prepare(
      "INSERT INTO api_keys (id, user_id, name, key_hash) VALUES (?, ?, ?, ?)"
    )
    .run(id, userId, name, keyHash)
}

export function deleteApiKeyById(userId: number, id: string): boolean {
  const result = getDb()
    .prepare("DELETE FROM api_keys WHERE id = ? AND user_id = ?")
    .run(id, userId)
  return result.changes > 0
}

export function getAllApiKeys(): ApiKeyRow[] {
  return getDb()
    .prepare("SELECT * FROM api_keys")
    .all() as ApiKeyRow[]
}
