import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import bcrypt

DB_PATH = Path(__file__).parent / "birdsense.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add role column if missing (existing DBs)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        except Exception:
            pass  # column already exists
        # Detections table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id              TEXT PRIMARY KEY,
                user_id         INTEGER NOT NULL,
                filename        TEXT NOT NULL,
                date            TEXT NOT NULL,
                time            TEXT NOT NULL,
                duration        REAL NOT NULL,
                top_species     TEXT NOT NULL,
                top_scientific  TEXT NOT NULL,
                top_confidence  REAL NOT NULL,
                predictions     TEXT NOT NULL,
                segments        TEXT NOT NULL,
                audio_url       TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        # User settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id       INTEGER PRIMARY KEY,
                settings_json TEXT NOT NULL DEFAULT '{}',
                updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        conn.commit()

# --- Auth Helpers ---
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_user_by_email(email: str):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

def create_user(email: str, password_hash: str, name: Optional[str] = None, role: str = 'user'):
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password, name, role) VALUES (?, ?, ?, ?)",
            (email, password_hash, name, role)
        )
        return cursor.lastrowid

def count_users() -> int:
    with get_db_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()
        return row["count"]

def get_all_users():
    with get_db_connection() as conn:
        return conn.execute("""
            SELECT u.id, u.email, u.name, u.role, u.created_at, u.updated_at,
                   COUNT(d.id) as detection_count
            FROM users u
            LEFT JOIN detections d ON d.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """).fetchall()

def get_user_by_id(user_id: int):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

def update_user_role(user_id: int, role: str):
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET role = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (role, user_id))
        conn.commit()

def update_user_password(user_id: int, password_hash: str):
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET password = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (password_hash, user_id))
        conn.commit()

def delete_user_by_id(user_id: int):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

def get_platform_stats():
    with get_db_connection() as conn:
        user_count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
        detection_count = conn.execute("SELECT COUNT(*) as c FROM detections").fetchone()["c"]
        top_species_row = conn.execute("""
            SELECT top_species, COUNT(*) as c FROM detections
            GROUP BY top_species ORDER BY c DESC LIMIT 1
        """).fetchone()
        top_species = top_species_row["top_species"] if top_species_row else "N/A"
        recent_detections = conn.execute("""
            SELECT d.id, d.filename, d.top_species, d.top_confidence, d.date, d.time, u.email as user_email
            FROM detections d
            JOIN users u ON u.id = d.user_id
            ORDER BY d.created_at DESC LIMIT 10
        """).fetchall()
        return {
            "user_count": user_count,
            "detection_count": detection_count,
            "top_species": top_species,
            "recent_detections": [dict(r) for r in recent_detections]
        }

# --- Detection Helpers ---
def insert_detection(data: Dict[str, Any]):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO detections 
            (id, user_id, filename, date, time, duration, top_species, top_scientific, top_confidence, predictions, segments, audio_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"], data["user_id"], data["filename"], data["date"], data["time"],
            data["duration"], data["top_species"], data["top_scientific"], 
            data["top_confidence"], json.dumps(data["predictions"]), 
            json.dumps(data["segments"]), data.get("audio_url")
        ))
        conn.commit()

def get_detections_by_user(user_id: int, limit: int = 50, offset: int = 0):
    with get_db_connection() as conn:
        return conn.execute("""
            SELECT * FROM detections WHERE user_id = ? 
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        """, (user_id, limit, offset)).fetchall()

def count_detections_by_user(user_id: int) -> int:
    with get_db_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as count FROM detections WHERE user_id = ?", (user_id,)).fetchone()
        return row["count"]

# --- Settings Helpers ---
def get_user_settings(user_id: int) -> Dict[str, Any]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT settings_json FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
        return json.loads(row["settings_json"]) if row else {}

def upsert_user_settings(user_id: int, settings: Dict[str, Any]):
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO user_settings (user_id, settings_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                settings_json = excluded.settings_json,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, json.dumps(settings)))
        conn.commit()
