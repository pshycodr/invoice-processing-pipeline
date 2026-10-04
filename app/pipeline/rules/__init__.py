"""Deterministic extraction of total, date and store. A small LLM is unreliable at picking exact
numbers, so these rules override it and the model keeps items, category and payment method."""
from .date import find_date
from .items import clean_items
from .store import pick_store, store_in_text
from .total import find_total

__all__ = ["clean_items", "find_date", "find_total", "fix_bill", "pick_store"]


def fix_bill(bill, text: str):
    """Override model output with rule-based values where the rules found something. Returns (bill, notes)."""
    updates, notes = {}, []

    total = find_total(text)
    if total is not None:
        if bill.total is not None and abs(bill.total - total) > 0.005:
            notes.append(f"Total corrected from {bill.total:.2f} to {total:.2f} using the bill text")
        updates["total"] = total

    date = find_date(text)
    if date:
        updates["date"] = date

    guess, strong = pick_store(text)
    name = (bill.store_name or "").strip()
    model_name_ok = bool(name) and name.lower() != "unknown" and store_in_text(name, text)
    extends = bool(guess) and bool(name) and name.lower() in guess.lower() and len(guess) > len(name)
    if guess and (strong or not model_name_ok or extends):
        updates["store_name"] = guess

    return bill.model_copy(update=updates), notes
