"""SQLite access for bills and their activity log."""
import json
import logging
import sqlite3
import time
from typing import Optional

from app.config import DB_PATH, UPLOAD_DIR

log = logging.getLogger("bill-scanner")

_COLUMNS = {"stage": "TEXT", "log": "TEXT", "updated": "REAL", "ocr_engine": "TEXT", "text_model": "TEXT"}


def query(sql: str, args=()):
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, args).fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()


def init_db() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    query("""CREATE TABLE IF NOT EXISTS bills(
        id TEXT PRIMARY KEY, filename TEXT, path TEXT, handwritten INT,
        status TEXT, method TEXT, data TEXT, warnings TEXT, error TEXT,
        stage TEXT, log TEXT, updated REAL, ocr_engine TEXT, text_model TEXT)""")
    have = {r["name"] for r in query("PRAGMA table_info(bills)")}
    for col, kind in _COLUMNS.items():
        if col not in have:
            query(f"ALTER TABLE bills ADD COLUMN {col} {kind}")


def get_bill(bill_id: str):
    rows = query("SELECT * FROM bills WHERE id=?", (bill_id,))
    return rows[0] if rows else None


def update_bill(bill_id: str, **fields) -> None:
    """Every update stamps `updated`, which tells the page to redraw that card."""
    fields["updated"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    query(f"UPDATE bills SET {cols} WHERE id=?", (*fields.values(), bill_id))


def append_log(bill_id: str, ok: bool, message: str, stage: Optional[str] = None) -> None:
    row = get_bill(bill_id)
    entries = json.loads(row["log"] or "[]") if row else []
    entries.append({"ok": ok, "m": message})
    update_bill(bill_id, log=json.dumps(entries), stage=stage)
    log.info("[%s] %s %s", bill_id, "ok" if ok else "FAIL", message)


def set_stage(bill_id: str, stage: str) -> None:
    update_bill(bill_id, stage=stage)
    log.info("[%s] ... %s", bill_id, stage)
