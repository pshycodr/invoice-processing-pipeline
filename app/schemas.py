"""Shared data models."""
import re
from typing import Optional

from pydantic import BaseModel, field_validator


class Item(BaseModel):
    name: str
    qty: Optional[float] = None
    price: Optional[float] = None  # line total


class Bill(BaseModel):
    store_name: str
    date: Optional[str] = None  # YYYY-MM-DD
    items: list[Item] = []
    tax: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None
    category: str


class ScanSettings(BaseModel):
    ocr_engine: Optional[str] = None
    text_model: Optional[str] = None

    @field_validator("text_model")
    @classmethod
    def _model_name(cls, v):
        v = (v or "").strip()
        if v and not re.fullmatch(r"[\w][\w.\-/:]{0,99}", v):
            raise ValueError("Model names may only contain letters, digits and . _ - / :")
        return v or None


class CategoryIn(BaseModel):
    name: str
