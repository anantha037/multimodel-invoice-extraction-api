import sqlite3
import json
import os
from datetime import datetime
from src.core.logger import get_logger

logger = get_logger(__name__)
DB_PATH = os.getenv("DB_PATH", "invoices.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT UNIQUE,
                timestamp TEXT,
                filename TEXT,
                prediction_json TEXT,
                confidence_score REAL,
                validation_status TEXT,
                corrected_json TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error("Database initialization failed", extra={"exception": str(e)})

def save_extraction(request_id: str, filename: str, prediction: dict, confidence: float, status: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO extractions (request_id, timestamp, filename, prediction_json, confidence_score, validation_status) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (request_id, datetime.utcnow().isoformat(), filename, json.dumps(prediction), confidence, status)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to save extraction to DB", extra={"request_id": request_id, "exception": str(e)})

def save_correction(request_id: str, corrected_json: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE extractions SET corrected_json = ? WHERE request_id = ?",
        (json.dumps(corrected_json), request_id)
    )
    conn.commit()
    conn.close()
