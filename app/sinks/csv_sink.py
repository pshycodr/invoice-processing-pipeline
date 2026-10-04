"""Append approved bills to local CSV files."""
import csv
import threading
from pathlib import Path

from app.config import CSV_BILLS_PATH, CSV_ITEMS_PATH
from app.schemas import Bill

BILLS_HEADER = ["bill_id", "date", "store", "category", "payment", "tax", "total", "file"]
ITEMS_HEADER = ["bill_id", "date", "store", "item", "qty", "price"]
_lock = threading.Lock()


def _cell(value):
    return "" if value is None else value


def _append(path: Path, header: list[str], rows: list[list]) -> None:
    with _lock:
        is_new = not path.exists() or path.stat().st_size == 0
        # utf-8-sig on creation so Excel opens the file correctly
        with open(path, "a", newline="", encoding="utf-8-sig" if is_new else "utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(header)
            writer.writerows(rows)


def save(bill_id: str, filename: str, bill: Bill) -> None:
    _append(CSV_BILLS_PATH, BILLS_HEADER,
            [[bill_id, _cell(bill.date), bill.store_name, bill.category, _cell(bill.payment_method),
              _cell(bill.tax), _cell(bill.total), filename]])
    if bill.items:
        _append(CSV_ITEMS_PATH, ITEMS_HEADER,
                [[bill_id, _cell(bill.date), bill.store_name, i.name, _cell(i.qty), _cell(i.price)] for i in bill.items])
