"""Ollama client: OCR text or a bill image in, validated Bill out."""
from pathlib import Path
from typing import Optional

import httpx
import ollama

from app.config import (MAX_OUTPUT_TOKENS, NUM_CTX, OLLAMA_HOST, OLLAMA_TIMEOUT, VISION_MAX_SIDE)
from app.pipeline.prompts import VISION_HINT, text_prompt, vision_prompt
from app.schemas import Bill

_client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)


def installed_models() -> set[str]:
    """Names of the models Ollama has pulled. Raises when Ollama is not running."""
    names = set()
    for m in _client.list().models:
        name = getattr(m, "model", None) or (m.get("name") if isinstance(m, dict) else None)
        if name:
            names.add(name)
    return names


def is_installed(model: str, installed: set[str]) -> bool:
    return model in installed or f"{model}:latest" in installed


def describe_error(e: Exception) -> str:
    """Turn the usual Ollama failures into a sentence a person can act on."""
    if isinstance(e, ConnectionError):
        return "Cannot reach Ollama. Start it (run 'ollama serve' or open the Ollama app) and try again."
    if isinstance(e, httpx.TimeoutException):
        return f"Ollama did not answer within {OLLAMA_TIMEOUT:.0f} seconds. The model may still be loading. Try again."
    if isinstance(e, ollama.ResponseError):
        if getattr(e, "status_code", None) == 404:
            return f"{e.error}. Download it with: ollama pull <model name>"
        return f"Ollama error: {e.error}"
    return f"{type(e).__name__}: {str(e)[:200]}"


def _chat(model: str, content: str, images: Optional[list[str]] = None) -> Bill:
    message = {"role": "user", "content": content}
    if images:
        message["images"] = images
    response = _client.chat(
        model=model,
        messages=[message],
        format=Bill.model_json_schema(),
        options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": MAX_OUTPUT_TOKENS},
    )
    return Bill.model_validate_json(response.message.content)


def extract_from_text(text: str, model: str, categories: list[str]) -> Bill:
    if not (text or "").strip():
        raise ValueError(f"{model} needs text to read")
    return _chat(model, f"{text_prompt(categories)}\n\nBILL TEXT:\n{text}")


def _upright_copy(path: Path) -> Path:
    """EXIF-rotated, downsized JPEG copy so the vision model never sees the bill sideways."""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE))
    out = path.with_suffix(".vlm.jpg")
    img.save(out, quality=92)
    return out


def extract_from_image(image_path: Path, model: str, categories: list[str], ocr_hint: str = "") -> Bill:
    prompt = vision_prompt(categories) + (VISION_HINT + ocr_hint if ocr_hint.strip() else "")
    return _chat(model, prompt, [str(_upright_copy(Path(image_path)))])
