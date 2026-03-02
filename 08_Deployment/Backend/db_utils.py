import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext

DB_PATH = Path(__file__).parent / "birdsense.db"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
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
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user_by_email(email: str):
    with get_db_connection() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

def create_user(email: str, password_hash: str, name: Optional[str] = None):
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (email, password, name) VALUES (?, ?, ?)",
            (email, password_hash, name)
        )
        return cursor.lastrowid

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
