"""OCR engines. Each returns text lines plus an average confidence."""
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import ollama

from app.config import GLM_OCR_MODEL, OCR_ENGINES, OLLAMA_HOST, OLLAMA_TIMEOUT


@dataclass
class OCRResult:
    text: str
    confidence: float


def group_lines(result) -> list[str]:
    """Join boxes that sit on the same row into one text line, left to right."""
    boxes = []
    for item in result or []:
        if len(item) < 3 or not str(item[1]).strip():
            continue
        ys = [p[1] for p in item[0]]
        xs = [p[0] for p in item[0]]
        boxes.append({"t": str(item[1]).strip(), "cy": (min(ys) + max(ys)) / 2,
                      "h": max(ys) - min(ys), "x": min(xs)})
    if not boxes:
        return []
    threshold = 0.6 * median(b["h"] for b in boxes)
    boxes.sort(key=lambda b: b["cy"])
    rows, current = [], [boxes[0]]
    for b in boxes[1:]:
        if abs(b["cy"] - sum(c["cy"] for c in current) / len(current)) <= threshold:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    return ["  ".join(c["t"] for c in sorted(r, key=lambda c: c["x"])) for r in rows]


class OCREngine(ABC):
    id = ""

    @abstractmethod
    def _raw(self, path: Path) -> list:
        """Return [(box, text, score), ...]."""

    def read(self, path: Path) -> OCRResult:
        raw = self._raw(path)
        if not raw:
            return OCRResult("", 0.0)
        scores = [s for s in (_as_float(item[2]) for item in raw) if s is not None]
        confidence = sum(scores) / len(scores) if scores else 0.0
        return OCRResult("\n".join(group_lines(raw)), confidence)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RapidOCREngine(OCREngine):
    id = "rapid"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._engine = RapidOCR()
        self._lock = threading.Lock()  # the ONNX session is not guaranteed thread-safe

    def _raw(self, path):
        with self._lock:
            result, _ = self._engine(str(path))
        return [(item[0], item[1], float(item[2])) for item in result or [] if item and len(item) >= 3]


class GLMOCREngine(OCREngine):
    id = "glm"
    PROMPT = ("Perform structured text recognition on this receipt. Output only plain text or clean "
              "Markdown lines. Do not use HTML or table tags. List the items, quantities and prices line by line.")
    PLACEHOLDER_BOX = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def __init__(self):
        self._client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)

    @staticmethod
    def _strip_markup(text: str) -> str:
        """Flatten any HTML table the model returns into 'cell | cell' lines and drop stray tags."""
        if "<table" in text.lower():
            lines = []
            for row in re.findall(r"<tr>(.*?)</tr>", text, re.DOTALL | re.IGNORECASE):
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
                cells = [c for c in cells if c]
                if cells:
                    lines.append(" | ".join(cells))
            if lines:
                return "\n".join(lines)
        text = re.sub(r"<[^>]+>", "", text)
        return text.replace("&amp;", "&").replace("&nbsp;", " ").strip()

    def _raw(self, path):
        response = self._client.chat(
            model=GLM_OCR_MODEL,
            messages=[{"role": "user", "content": self.PROMPT, "images": [Path(path).read_bytes()]}],
            options={"temperature": 0.0},
        )
        text = self._strip_markup(response["message"]["content"])
        return [(self.PLACEHOLDER_BOX, text, 0.95)] if text else []


_ENGINE_CLASSES = {"glm": GLMOCREngine, "rapid": RapidOCREngine}
_cache: dict[str, OCREngine] = {}
_cache_lock = threading.Lock()


def get_engine(engine_id: str) -> OCREngine:
    if engine_id not in OCR_ENGINES:
        raise ValueError(f"Unknown OCR engine '{engine_id}'. Options: {', '.join(OCR_ENGINES)}")
    with _cache_lock:
        if engine_id not in _cache:
            _cache[engine_id] = _ENGINE_CLASSES[engine_id]()
        return _cache[engine_id]
