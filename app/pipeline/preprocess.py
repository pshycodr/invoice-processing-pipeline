"""Photo cleanup before OCR."""
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

from app.config import OCR_CONTRAST, OCR_MAX_SIDE, OCR_SHARPNESS

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def preprocess(src: Path) -> Path:
    """Apply EXIF rotation, shrink, and boost contrast and sharpness. Returns the processed copy."""
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Contrast(img).enhance(OCR_CONTRAST)
    img = ImageEnhance.Sharpness(img).enhance(OCR_SHARPNESS)
    out = src.with_suffix(".proc.jpg")
    img.save(out, quality=95, optimize=True)
    return out
