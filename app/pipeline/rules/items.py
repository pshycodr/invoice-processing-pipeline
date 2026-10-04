"""Filter out lines a small model wrongly lists as products."""
import re

JUNK_ITEM = re.compile(
    r"^(visa|master\s*card|mastercard|debit|credit|cash|change|tender(ed)?|paid|payment|card|round(ing|ed)?|"
    r"total\w*|sub\s*-?\s*total|grand\s*total|net\s*total|amount\w*|balance|tax\w*|gst\w*|sst|vat|cgst|sgst|cess|"
    r"points?\b.*|invoice\b.*|receipt\b.*|thank\b.*|pump\b.*|(a\s*)?(rm|inr|rs|usd|myr)\.?\s*\d*)$"
    r"|loyalty",
    re.I)


def clean_items(bill):
    """Drop entries that cannot be products. Returns (bill, notes)."""
    keep, dropped = [], []
    for item in bill.items:
        name = re.sub(r"^[\W_]+|[\W_]+$", "", item.name or "")
        letters = len(re.sub(r"[^A-Za-z]", "", name))
        digits = len(re.sub(r"\D", "", name))
        if letters < 2 or digits > letters * 2 or JUNK_ITEM.search(name):
            dropped.append(item.name)
        else:
            keep.append(item)
    if not dropped:
        return bill, []
    shown = ", ".join(f"'{d}'" for d in dropped[:4]) + (" and more" if len(dropped) > 4 else "")
    return bill.model_copy(update={"items": keep}), [f"Removed {len(dropped)} line(s) that are not products: {shown}"]
