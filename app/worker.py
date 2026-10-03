"""Single-worker queue: bills are read one at a time, in the order they were scanned."""
import asyncio
import logging
from typing import Optional

from app.database import query, update_bill
from app.pipeline.runner import process_bill

log = logging.getLogger("bill-scanner")
_jobs: Optional[asyncio.Queue] = None


def enqueue(bill_id: str, force_vision: bool = False) -> None:
    _jobs.put_nowait((bill_id, force_vision))


async def _work() -> None:
    while True:
        bill_id, force_vision = await _jobs.get()
        try:
            await asyncio.to_thread(process_bill, bill_id, force_vision)
        except Exception:
            log.exception("worker error")
        finally:
            _jobs.task_done()


def start() -> asyncio.Task:
    """Create the queue, requeue anything a restart interrupted, and start the worker."""
    global _jobs
    _jobs = asyncio.Queue()
    for row in query("SELECT id FROM bills WHERE status IN ('queued','processing') ORDER BY rowid"):
        update_bill(row["id"], status="queued", stage=None)
        enqueue(row["id"])
    return asyncio.create_task(_work())
