from app.config import SYNC_TARGET
from app.schemas import Bill


def sync_bill(bill_id: str, filename: str, bill: Bill) -> None:
    if SYNC_TARGET == "sheet":
        from app.sinks import sheets
        sheets.save(bill_id, filename, bill)
    else:
        from app.sinks import csv_sink
        csv_sink.save(bill_id, filename, bill)
