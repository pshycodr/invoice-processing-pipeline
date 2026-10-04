"""Store name detection."""
import re
from difflib import SequenceMatcher
from typing import Optional

JUNK = re.compile(r"^\(?[a-z]{0,3}[\d\-]{5,}|\btax invoice\b|\binvoice\b|\breceipt\b|^tel\b|^fax\b|\bgst\b|"
                  r"\bcash bill\b|^no[.:\s]|\bjalan\b|^lot\b|\bdate\b|\bcopy\b", re.I)
COMPANY = re.compile(
    r"sdn\.?\s*b[a-z]{1,2}\b|\bs/b\b|\bsb\b|\bbhd\b|enterprise|perniagaan|syarikat|kedai|trading|restaurant|"
    r"restoran|marketing|engineering|gallery|industries|supermarket|mart\b|store|stationery|hardware|cafe|"
    r"bistro|pharmacy|shop|\bco\b|station|services|supplies", re.I)
CONTINUATION = re.compile(r"^\(?\s*(?:sdn\.?|bhd\.?|s/b|sb|co\.?\s*\(m\))(?![A-Za-z])", re.I)
REGISTRATION = re.compile(r"\s*[\(\[]\s*[A-Za-z]{0,3}\d[\d\-]{3,}[\-A-Za-z]{0,2}\s*[\)\]]\s*$")

SUFFIXES = ("SDN", "BHD", "ENTERPRISES", "ENTERPRISE", "TRADING", "MARKETING", "RESTAURANTS",
            "RESTAURANT", "STATIONERY", "HARDWARE", "GALLERY", "NETWORK", "ENGINEERING",
            "INDUSTRIES", "SUPERMARKET")
_SUFFIX = re.compile(r"^(.{2,}?)(" + "|".join(SUFFIXES) + r")([.,]*)$", re.I)


def unglue(s: str) -> str:
    """'OJCMARKETINGSDNBHD' -> 'OJC MARKETING SDN BHD'. Only long space-less tokens are touched."""
    out = []
    for token in s.split():
        parts = []
        while len(token) >= 8 and (m := _SUFFIX.match(token)):
            parts.insert(0, m[2] + m[3])
            token = m[1]
        out.append(" ".join([token, *parts]))
    return " ".join(out)


def _finish(s: str) -> str:
    return REGISTRATION.sub("", unglue(s)).strip()


def _texty(line: str) -> bool:
    line = REGISTRATION.sub("", line)
    letters = len(re.sub(r"[^A-Za-z]", "", line))
    return letters >= 4 and letters / max(1, len(re.sub(r"\s", "", line))) >= 0.6


def pick_store(text: str) -> tuple[Optional[str], bool]:
    """Return (store line, is_strong). Strong means the line has a business keyword such as SDN BHD."""
    lines = [l.strip() for l in text.splitlines() if l.strip()][:10]
    candidates = [(i, l) for i, l in enumerate(lines) if _texty(l) and not JUNK.search(l)]
    candidate_idx = {i for i, _ in candidates}
    for i, line in candidates:
        if COMPANY.search(line):
            if CONTINUATION.match(line) and (i - 1) in candidate_idx:
                line = lines[i - 1] + " " + line
            elif i + 1 < len(lines) and CONTINUATION.match(lines[i + 1]):
                line = line + " " + lines[i + 1]
            return _finish(line), True
    if candidates:
        i, name = candidates[0]
        if i > 0 and re.fullmatch(r"[A-Z]{2,5}", lines[i - 1]) and not re.search(r"\d", name):
            name = lines[i - 1] + " " + name
        return _finish(name), False
    return (lines[0] if lines else None), False


def store_in_text(store: str, text: str, threshold=0.75) -> bool:
    """True if the model's store name resembles a real OCR line. Guards against invented names."""
    s = re.sub(r"[^A-Z0-9 ]", "", store.upper()).strip()
    lines = [re.sub(r"[^A-Z0-9 ]", "", l.upper()).strip() for l in text.splitlines()[:12]]
    candidates = lines + [a + " " + b for a, b in zip(lines, lines[1:])]
    return bool(s) and any(SequenceMatcher(None, s, c).ratio() >= threshold for c in candidates if c)
