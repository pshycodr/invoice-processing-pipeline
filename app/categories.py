"""User-editable category list, persisted as JSON."""
import json
import threading

from app.config import CATEGORIES_PATH

DEFAULTS = ["Groceries", "Food & Dining", "Electronics", "Clothing", "Medical",
            "Household", "Transport", "Stationery", "Other"]
FALLBACK = "Other"
_lock = threading.Lock()


def load() -> list[str]:
    try:
        data = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(c, str) for c in data):
            return data if FALLBACK in data else [*data, FALLBACK]
    except (OSError, ValueError):
        pass
    return list(DEFAULTS)


def _save(items: list[str]) -> list[str]:
    CATEGORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATEGORIES_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items


def add(name: str) -> list[str]:
    name = " ".join(name.split())
    if not 1 <= len(name) <= 40:
        raise ValueError("A category name needs 1 to 40 characters.")
    with _lock:
        items = load()
        if name.lower() in {c.lower() for c in items}:
            raise ValueError(f"'{name}' is already in the list.")
        insert_at = items.index(FALLBACK)
        items.insert(insert_at, name)
        return _save(items)


def remove(name: str) -> list[str]:
    if name == FALLBACK:
        raise ValueError(f"'{FALLBACK}' is the fallback category and cannot be removed.")
    with _lock:
        items = [c for c in load() if c != name]
        return _save(items)
