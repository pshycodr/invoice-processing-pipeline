"""Runtime settings, read once from the environment."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "bills.db"
CATEGORIES_PATH = DATA_DIR / "categories.json"
CSV_BILLS_PATH = DATA_DIR / "bills.csv"
CSV_ITEMS_PATH = DATA_DIR / "items.csv"
STATIC_DIR = Path(__file__).parent / "static"
DERIVED_SUFFIXES = (".proc.jpg", ".vlm.jpg", ".preview.jpg", ".ocr.txt")

OCR_ENGINES = {"glm": "GLM-OCR", "rapid": "RapidOCR"}
DEFAULT_OCR = os.getenv("OCR_ENGINE", "glm")
if DEFAULT_OCR not in OCR_ENGINES:
    DEFAULT_OCR = "glm"

TEXT_MODEL_PRESETS = ["gemma3:1b", "gemma3:4b", "llama3.2:3b", "phi4-mini", "qwen2.5:3b"]
DEFAULT_TEXT_MODEL = os.getenv("TEXT_MODEL", "gemma3:1b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:4b-instruct")
GLM_OCR_MODEL = os.getenv("GLM_OCR_MODEL", "glm-ocr")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "3072"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
VISION_MAX_SIDE = int(os.getenv("VISION_MAX_SIDE", "1344"))

OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "1800"))
OCR_CONTRAST = float(os.getenv("OCR_CONTRAST", "1.5"))
OCR_SHARPNESS = float(os.getenv("OCR_SHARPNESS", "1.7"))

OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.50"))
OCR_WARN_CONF = 0.60
MIN_OCR_CHARS = int(os.getenv("MIN_OCR_CHARS", "30"))
MIN_OCR_WORDS = int(os.getenv("MIN_OCR_WORDS", "4"))
MIN_WORD_RATIO = float(os.getenv("MIN_WORD_RATIO", "0.15"))
USE_RULES = os.getenv("USE_RULES", "1") == "1"

SYNC_TARGET = os.getenv("SYNC_TARGET", "csv")  # csv | sheet
SHEET_NAME = os.getenv("SHEET_NAME", "Bills")
GOOGLE_CREDS = os.getenv("GOOGLE_CREDS", "creds.json")
