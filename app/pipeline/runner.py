"""Reads one bill end to end.

    image -> OCR -> text -> text model -> JSON      (normal path)
    image -> vision model -> JSON                   (fallback)

The vision model runs when OCR fails, when the bill is marked handwritten, when the text model
finds no items, or when the person asks for it. If the text model itself fails, the bill stops as failed.
"""
import json
import logging
import time
from pathlib import Path

from app import categories as category_store
from app.config import DEFAULT_OCR, DEFAULT_TEXT_MODEL, OCR_ENGINES, OCR_WARN_CONF, VISION_MODEL
from app.database import append_log, get_bill, set_stage, update_bill
from app.pipeline import llm
from app.pipeline.extract import read_image_bill, read_text_bill
from app.pipeline.ocr import get_engine
from app.pipeline.preprocess import preprocess
from app.pipeline.validation import ocr_failure, review_warnings
from app.schemas import Bill

log = logging.getLogger("bill-scanner")
Result = tuple[Bill, list[str], str]


class OCRFallback(Exception):
    """OCR could not read the image. The reason is already logged and the vision model takes over."""


class Reported(Exception):
    """A step failed and is already logged. The bill stops as failed."""


def _read_with_ocr(bill_id: str, src: Path, engine_id: str, model: str, categories: list[str]) -> Result:
    engine_name = OCR_ENGINES[engine_id]
    set_stage(bill_id, f"Reading the text with {engine_name}")
    started = time.perf_counter()
    try:
        result = get_engine(engine_id).read(preprocess(src))
        text, confidence = result.text, result.confidence
        reason = ocr_failure(text, confidence)
    except Exception as e:
        text, confidence, reason = "", 0.0, describe_ocr_error(e)
    if reason:
        append_log(bill_id, False, f"OCR failed: {reason}", stage=f"OCR failed, so {VISION_MODEL} will read the image instead")
        raise OCRFallback(reason)
    src.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
    append_log(bill_id, True,
               f"{engine_name} read {len(text.splitlines())} lines, confidence {confidence:.0%} ({time.perf_counter() - started:.1f}s)",
               stage=f"{model} is turning the text into JSON")

    started = time.perf_counter()
    try:
        bill, notes = read_text_bill(text, model, categories)
    except Exception as e:
        reason = llm.describe_error(e)
        append_log(bill_id, False, f"{model} failed: {reason}")
        raise Reported(f"{model} failed: {reason}") from e
    append_log(bill_id, True, f"{model} returned the JSON ({time.perf_counter() - started:.1f}s)")

    warnings = review_warnings(bill, notes, categories)
    if confidence < OCR_WARN_CONF:
        warnings.append(f"OCR confidence is low ({confidence:.0%}). Check the extracted values carefully.")
    return bill, warnings, f"{engine_name} + {model}"


def describe_ocr_error(e: Exception) -> str:
    return llm.describe_error(e) if "ollama" in type(e).__module__ else f"{type(e).__name__}: {str(e)[:150]}"


def _read_with_vision(bill_id: str, src: Path, categories: list[str], reason: str = "") -> Result:
    set_stage(bill_id, f"{VISION_MODEL} is reading the image (the first run loads the model and can take a minute)")
    started = time.perf_counter()
    bill, notes = read_image_bill(src, VISION_MODEL, categories)
    append_log(bill_id, True, f"{VISION_MODEL} returned the JSON ({time.perf_counter() - started:.0f}s)")
    warnings = ["Read from the image by the vision model only. Check every value against the bill."]
    warnings += review_warnings(bill, notes, categories)
    return bill, warnings, f"{VISION_MODEL} (vision{', ' + reason if reason else ''})"


def _retry_empty_with_vision(bill_id: str, src: Path, ocr_result: Result, categories: list[str]) -> Result:
    """The text model found no items. Keep the OCR result if the vision model fails or also finds none."""
    ocr_bill, ocr_warnings, ocr_method = ocr_result
    try:
        bill, warnings, method = _read_with_vision(bill_id, src, categories, "after OCR found no items")
    except Exception as e:
        message = llm.describe_error(e)
        append_log(bill_id, False, f"{VISION_MODEL} failed: {message}")
        return ocr_bill, ocr_warnings + [f"The vision model could not help: {message}"], ocr_method
    if not bill.items:
        append_log(bill_id, False, f"{VISION_MODEL} also found no items")
        return ocr_bill, ocr_warnings + [f"{VISION_MODEL} also found no items. This is the OCR result."], ocr_method
    if ocr_bill.total is not None and bill.total is not None and abs(ocr_bill.total - bill.total) > 0.005:
        warnings.append(f"Total differs: the vision model read {bill.total:.2f}, the OCR text says {ocr_bill.total:.2f}.")
    return bill, warnings, method


def process_bill(bill_id: str, force_vision: bool = False) -> None:
    """Never leaves a bill stuck: it ends as 'review' or 'failed'."""
    row = get_bill(bill_id)
    if row is None:
        return
    src = Path(row["path"])
    engine_id = row["ocr_engine"] if row["ocr_engine"] in OCR_ENGINES else DEFAULT_OCR
    model = row["text_model"] or DEFAULT_TEXT_MODEL
    categories = category_store.load()
    update_bill(bill_id, status="processing", stage="Starting", log="[]", error=None, data=None, warnings="[]", method=None)

    try:
        result = None
        if force_vision:
            append_log(bill_id, True, "Vision model requested, OCR skipped", stage=f"{VISION_MODEL} is reading the image")
        elif row["handwritten"]:
            append_log(bill_id, True, "Marked as handwritten or poor photo, OCR skipped", stage=f"{VISION_MODEL} is reading the image")
        else:
            try:
                ocr_result = _read_with_ocr(bill_id, src, engine_id, model, categories)
            except OCRFallback:
                ocr_result = None
            if ocr_result is not None and not ocr_result[0].items:
                append_log(bill_id, False, f"{model} found no items in the text", stage=f"{VISION_MODEL} is reading the image")
                result = _retry_empty_with_vision(bill_id, src, ocr_result, categories)
            elif ocr_result is not None:
                result = ocr_result
        if result is None:
            after_ocr = not (force_vision or row["handwritten"])
            result = _read_with_vision(bill_id, src, categories, "after OCR failed" if after_ocr else "")
        bill, warnings, method = result
        update_bill(bill_id, status="review", stage=None, method=method,
                    data=bill.model_dump_json(), warnings=json.dumps(warnings))
    except Reported as e:
        update_bill(bill_id, status="failed", stage=None, error=str(e))
    except Exception as e:
        message = llm.describe_error(e)
        append_log(bill_id, False, message)
        update_bill(bill_id, status="failed", stage=None, error=message)
