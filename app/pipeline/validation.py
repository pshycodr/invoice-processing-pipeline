"""Sanity checks on OCR output and extracted bills."""
import re
from datetime import date as _date
from typing import Optional

from app.config import MIN_OCR_CHARS, MIN_OCR_WORDS, MIN_WORD_RATIO, OCR_MIN_CONF
from app.schemas import Bill


def readable_words(text: str) -> tuple[int, int]:
    """(tokens made only of letters and 3+ long, all tokens). Garbage like 'isape1Bay0' scores zero."""
    tokens = [re.sub(r"^[\W_]+|[\W_]+$", "", t) for t in text.split()]
    tokens = [t for t in tokens if t]
    return sum(t.isalpha() and len(t) >= 3 for t in tokens), len(tokens)


def ocr_failure(text: str, confidence: float) -> Optional[str]:
    """Why the OCR result is unusable, or None if it is good enough for the text model.
    Confidence alone is not enough: a 71% confident read can still be pure garbage."""
    text = text.strip()
    if not text:
        return "no text was found in the image"
    if len(text) < MIN_OCR_CHARS:
        return f"only {len(text)} characters were read"
    if confidence < OCR_MIN_CONF:
        return f"confidence is too low ({confidence:.0%})"
    words, total = readable_words(text)
    if words < MIN_OCR_WORDS or words / max(1, total) < MIN_WORD_RATIO:
        return f"the text is mostly unreadable ({words} readable words out of {total})"
    return None


def sanitize(bill: Bill) -> tuple[Bill, list[str]]:
    """The date must be a real YYYY-MM-DD, otherwise it is dropped and flagged."""
    if bill.date:
        try:
            _date.fromisoformat(bill.date.strip())
            if bill.date != bill.date.strip():
                bill = bill.model_copy(update={"date": bill.date.strip()})
        except ValueError:
            return bill.model_copy(update={"date": None}), [f"The date '{bill.date}' was not a valid date and was removed"]
    return bill, []


def check(bill: Bill, categories: list[str]) -> list[str]:
    warnings = []
    if bill.total is None:
        warnings.append("Total is missing")
    if not bill.date:
        warnings.append("Date is missing")
    if bill.category not in categories:
        warnings.append("Category is not in your list")
    prices = [i.price for i in bill.items if i.price is not None]
    if prices and bill.total:
        s, tolerance = sum(prices), 0.01 * bill.total
        if abs(s - bill.total) > tolerance and abs(s + (bill.tax or 0) - bill.total) > tolerance:
            warnings.append(f"Items add up to {s:.2f} but total is {bill.total:.2f}")
    return warnings


def review_warnings(bill: Bill, notes: list[str], categories: list[str]) -> list[str]:
    warnings = check(bill, categories) + notes
    if not bill.items:
        warnings.append("No items were found")
    return warnings
