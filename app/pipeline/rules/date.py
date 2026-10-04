"""Bill date detection. Day-first is assumed for ambiguous numeric dates."""
import calendar
import re
from datetime import date as _date
from typing import Optional

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MIN_YEAR = 2015


def _iso(y, m, d):
    y = 2000 + y if y < 100 else y
    try:
        if MIN_YEAR <= y <= _date.today().year + 1:
            return _date(y, m, d).isoformat()
    except ValueError:
        pass
    return None


def find_date(text: str) -> Optional[str]:
    """Return YYYY-MM-DD, preferring dates on a line that mentions 'date'.
    OCR often glues the time onto the year ('15/01/201911:05:16AM'), so 4-digit years need no trailing boundary."""
    found = []  # (is_on_date_line, line position, iso)
    for pos, line in enumerate(text.splitlines()):
        low = line.lower()
        flag = "date" in low or " dt" in low
        for m in re.finditer(r"(?<!\d)(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", low):
            v = _iso(int(m[1]), int(m[2]), int(m[3]))
            if v:
                found.append((flag, pos, v))
        # The lookahead allows overlapping matches, so '#19-18/01/2018' still yields 18/01/2018.
        numeric = (r"(?<!\d)(?<!\d\.)(?=(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*"
                   r"(20\d{2}|\d{2}(?!\d)|\d{2}(?=\d{4,})))")
        for m in re.finditer(numeric, low):
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            v = _iso(y, b, a) or _iso(y, a, b)
            if v:
                found.append((flag, pos, v))
        for m in re.finditer(r"(?<!\d)(\d{1,2})\s*([a-z]{3})[a-z]*\.?,?\s*(20\d{2}|\d{2}(?!\d))", low):
            if m[2] in MONTHS:
                v = _iso(int(m[3]), MONTHS[m[2]], int(m[1]))
                if v:
                    found.append((flag, pos, v))
    if not found:
        return None
    found.sort(key=lambda f: (not f[0], f[1]))
    return found[0][2]
