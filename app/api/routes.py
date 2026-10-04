"""HTTP API used by the web page."""
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from app import categories as category_store
from app.config import (DEFAULT_OCR, DEFAULT_TEXT_MODEL, DERIVED_SUFFIXES, OCR_ENGINES, SYNC_TARGET,
                        TEXT_MODEL_PRESETS, UPLOAD_DIR, VISION_MODEL)
from app.database import get_bill, query, update_bill
from app.pipeline import llm
from app.pipeline.preprocess import preprocess  # noqa: F401  (registers HEIC support)
from app.schemas import Bill, CategoryIn, ScanSettings
from app.sinks import sync_bill
from app.worker import enqueue

router = APIRouter(prefix="/api")
BROWSER_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _resolve(settings: Optional[ScanSettings]) -> tuple[str, str]:
    settings = settings or ScanSettings()
    engine = settings.ocr_engine or DEFAULT_OCR
    if engine not in OCR_ENGINES:
        raise HTTPException(422, f"Unknown OCR engine '{engine}'.")
    return engine, settings.text_model or DEFAULT_TEXT_MODEL


def _require(bill_id: str):
    row = get_bill(bill_id)
    if row is None:
        raise HTTPException(404)
    return row


@router.get("/config")
def config():
    state = {"ollama_ok": True, "installed": [], "ollama_error": None}
    try:
        state["installed"] = sorted(llm.installed_models())
    except Exception as e:
        state.update(ollama_ok=False, ollama_error=llm.describe_error(e))
    return {
        "categories": category_store.load(),
        "ocr_engines": [{"id": k, "label": v} for k, v in OCR_ENGINES.items()],
        "default_ocr": DEFAULT_OCR,
        "text_model_presets": TEXT_MODEL_PRESETS,
        "default_text_model": DEFAULT_TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "sync": SYNC_TARGET,
        **state,
    }


@router.get("/categories")
def list_categories():
    return category_store.load()


@router.post("/categories")
def add_category(body: CategoryIn):
    try:
        return category_store.add(body.name)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.delete("/categories/{name}")
def delete_category(name: str):
    try:
        return category_store.remove(name)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.post("/upload")
async def upload(files: list[UploadFile] = File(...), handwritten: bool = Form(False)):
    """Save the files as 'staged'. Nothing is read until /api/scan."""
    ids = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        bill_id = uuid.uuid4().hex[:10]
        path = UPLOAD_DIR / f"{bill_id}{Path(f.filename or '').suffix.lower() or '.jpg'}"
        path.write_bytes(data)
        query("INSERT INTO bills(id,filename,path,handwritten,status,log,updated) VALUES(?,?,?,?,'staged','[]',?)",
              (bill_id, f.filename or path.name, str(path), int(handwritten), time.time()))
        ids.append(bill_id)
    return {"ids": ids}


@router.post("/scan")
def scan(settings: Optional[ScanSettings] = None):
    """Queue every staged bill, oldest first, with the chosen OCR engine and text model."""
    engine, model = _resolve(settings)
    ids = [r["id"] for r in query("SELECT id FROM bills WHERE status='staged' ORDER BY rowid")]
    if not ids:
        raise HTTPException(409, "There are no bills to scan. Add some first.")
    for bill_id in ids:
        update_bill(bill_id, status="queued", stage=None, error=None, log="[]", ocr_engine=engine, text_model=model)
        enqueue(bill_id)
    return {"queued": len(ids)}


@router.post("/bills/{bill_id}/flag")
def flag(bill_id: str, handwritten: bool):
    row = _require(bill_id)
    if row["status"] not in ("staged", "failed", "review"):
        raise HTTPException(409, "This bill cannot be changed right now.")
    update_bill(bill_id, handwritten=int(handwritten))
    return {"ok": True}


@router.post("/staged/flag")
def flag_all(handwritten: bool):
    ids = [r["id"] for r in query("SELECT id FROM bills WHERE status='staged'")]
    for bill_id in ids:
        update_bill(bill_id, handwritten=int(handwritten))
    return {"updated": len(ids)}


@router.delete("/bills/{bill_id}")
def remove(bill_id: str):
    row = _require(bill_id)
    if row["status"] in ("queued", "processing"):
        raise HTTPException(409, "This bill is being read. Wait until it finishes.")
    if row["status"] == "synced":
        raise HTTPException(409, "This bill was already sent and cannot be removed here.")
    query("DELETE FROM bills WHERE id=?", (bill_id,))
    src = Path(row["path"])
    for p in (src, *(src.with_suffix(s) for s in DERIVED_SUFFIXES)):
        p.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/bills")
def bills():
    return [{"id": r["id"], "seq": r["rowid"], "filename": r["filename"], "status": r["status"],
             "stage": r["stage"], "handwritten": bool(r["handwritten"]), "method": r["method"],
             "updated": r["updated"], "data": json.loads(r["data"]) if r["data"] else None,
             "warnings": json.loads(r["warnings"] or "[]"), "log": json.loads(r["log"] or "[]"),
             "error": r["error"]}
            for r in query("SELECT rowid, * FROM bills ORDER BY rowid")]


@router.get("/image/{bill_id}")
def image(bill_id: str):
    path = Path(_require(bill_id)["path"])
    if path.suffix.lower() in BROWSER_FORMATS:
        return FileResponse(path)
    preview = path.with_suffix(".preview.jpg")  # HEIC and TIFF cannot be shown by browsers
    if not preview.exists():
        ImageOps.exif_transpose(Image.open(path)).convert("RGB").save(preview, quality=85)
    return FileResponse(preview)


@router.post("/bills/{bill_id}/approve")
def approve(bill_id: str, bill: Bill):
    row = _require(bill_id)
    if row["status"] == "synced":
        raise HTTPException(409, "This bill was already sent.")
    if bill.date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", bill.date):
        raise HTTPException(422, "Date must look like YYYY-MM-DD, for example 2026-03-27.")
    update_bill(bill_id, data=bill.model_dump_json(), status="approved")
    try:
        sync_bill(bill_id, row["filename"], bill)
        update_bill(bill_id, status="synced", error=None)
    except Exception as e:
        update_bill(bill_id, error=str(e))
        raise HTTPException(502, f"Saved locally, but the sync failed: {e}")
    return {"ok": True}


@router.post("/bills/{bill_id}/retry")
def retry(bill_id: str, vision_model: bool = False, settings: Optional[ScanSettings] = None):
    """Read a bill again. vision_model=true skips OCR. Optional settings replace the stored ones."""
    row = _require(bill_id)
    if row["status"] in ("queued", "processing"):
        raise HTTPException(409, "This bill is already being read.")
    if row["status"] == "synced":
        raise HTTPException(409, "This bill was already sent.")
    fields = {}
    if settings and (settings.ocr_engine or settings.text_model):
        engine, model = _resolve(settings)
        fields = {"ocr_engine": engine, "text_model": model}
    update_bill(bill_id, status="queued", stage=None, error=None, log="[]", **fields)
    enqueue(bill_id, vision_model)
    return {"ok": True}
