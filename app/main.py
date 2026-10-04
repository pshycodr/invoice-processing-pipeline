"""FastAPI entrypoint. Run with: uvicorn app.main:app --host 127.0.0.1 --port 8000"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import worker
from app.api.routes import router
from app.config import STATIC_DIR
from app.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    task = worker.start()
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Bill Scanner", lifespan=lifespan)
app.include_router(router)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
