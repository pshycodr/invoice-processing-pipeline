"""Bill total detection."""
import re
from typing import Optional

# At most 7 integer digits and no leading zero, so an HSN code glued to a price is rejected.
AMOUNT = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|[1-9]\d{0,6}|0)\.\d{2}(?!\d)")
EXCLUDE = re.compile(
    r"sub\s*-?\s*total|total\s*(tax|gst|qty|quantity|item|disc|saving|point|includes)|excl|"
    r"before\s*(gst|tax)|(gst|tax)\s*(amt|amount)?\s*payable|includes\s*\d")
SUMMARY = re.compile(r"gst\s*summary|tax\s*summary|tax\s*code|summary")
PRIORITY = [
    re.compile(r"(grand|nett?|rounded?)\s*total|total\s*(amt|amount)?\s*(due|payable)|"
               r"amount\s*(due|payable)|payable|total\s*rounded"),
    re.compile(r"total\s*(sales\s*)?\(?\s*incl|total\s*(amt|amount)|total\s*\(?\s*rm"),
    re.compile(r"\btotal\b"),  # generic: trusted only when the line has exactly one amount
]
NOT_TOTAL_FALLBACK = re.compile(r"cash|change|tender|paid|payment|round|card|visa|master")


def amounts_in_line(line: str) -> list[float]:
    return [float(m.group().replace(",", "")) for m in AMOUNT.finditer(line)]


def find_total(text: str) -> Optional[float]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    low = [l.lower() for l in lines]
    best = None  # (priority, -index, value)
    in_summary = False
    for i, line in enumerate(low):
        if SUMMARY.search(line):
            in_summary = True
        if EXCLUDE.search(line):
            continue
        priority = next((p for p, rx in enumerate(PRIORITY) if rx.search(line)), None)
        if priority is None:
            continue
        amounts = amounts_in_line(line)
        if not amounts and len(line.split()) <= 3:  # bare "TOTAL": the value sits on a following line
            for j in range(i + 1, min(i + 3, len(lines))):
                if any(rx.search(low[j]) for rx in PRIORITY):
                    break
                amounts = amounts_in_line(low[j])
                if amounts:
                    break
        if not amounts:
            continue
        if priority == 2 and (in_summary or len(amounts) != 1):  # GST summary rows like "Total 7.50 0.45"
            continue
        candidate = (priority, -i, amounts[-1])
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best:
        return best[2]

    # No total line: among the three largest distinct amounts, prefer the one printed most often.
    pool = [a for l in low if not NOT_TOTAL_FALLBACK.search(l) for a in amounts_in_line(l) if a > 0]
    if not pool:
        return None
    top = sorted(set(pool), reverse=True)[:3]
    return max(top, key=lambda v: (pool.count(v), v))
