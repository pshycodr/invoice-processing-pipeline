"""Model extraction followed by rule-based fixes. Shared by the app and the evaluation scripts."""
from pathlib import Path

from app.config import USE_RULES
from app.pipeline import llm
from app.pipeline.rules import clean_items, fix_bill
from app.pipeline.validation import sanitize
from app.schemas import Bill


def read_text_bill(text: str, model: str, categories: list[str]) -> tuple[Bill, list[str]]:
    """Text model proposes the JSON, rules correct total, date and store. Returns (bill, notes)."""
    bill = llm.extract_from_text(text, model, categories)
    notes: list[str] = []
    if USE_RULES:
        bill, notes = fix_bill(bill, text)
    bill, date_notes = sanitize(bill)
    bill, item_notes = clean_items(bill)
    return bill, notes + date_notes + item_notes


def read_image_bill(image_path: Path, model: str, categories: list[str]) -> tuple[Bill, list[str]]:
    bill = llm.extract_from_image(image_path, model, categories)
    bill, date_notes = sanitize(bill)
    bill, item_notes = clean_items(bill)
    return bill, date_notes + item_notes
