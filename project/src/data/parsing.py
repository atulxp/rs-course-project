"""Low-level parsing helpers for the raw 99acres/Housing-style export.

The raw CSVs store several fields as human-formatted strings (e.g.
PRICE="1.1 Cr", AREA="518-623  sq.ft.") or as Python-repr'd dicts
(location="{'CITY_NAME': 'Gurgaon', ...}", MAP_DETAILS="{'LATITUDE': ...}").
These helpers turn them into clean numeric / structured values.
"""
from __future__ import annotations

import ast
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_CRORE = 1e7
_LAKH = 1e5
_THOUSAND = 1e3

_PRICE_UNIT_RE = re.compile(r"(cr|crore|l|lac|lakh|k)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"[\d,]+\.?\d*")


def _to_float(token: str) -> Optional[float]:
    token = token.replace(",", "").strip()
    if token == "":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_price_token(token: str) -> Optional[float]:
    """Parse a single price token like '1.1 Cr', '69.25 L', '85,000' -> INR float."""
    if token is None:
        return None
    token = token.strip()
    if token == "":
        return None
    low = token.lower()
    multiplier = 1.0
    if "cr" in low:
        multiplier = _CRORE
    elif "lac" in low or "lakh" in low or re.search(r"\bl\b", low):
        multiplier = _LAKH
    elif re.search(r"\bk\b", low):
        multiplier = _THOUSAND
    num_match = _NUM_RE.search(token)
    if not num_match:
        return None
    val = _to_float(num_match.group())
    if val is None:
        return None
    return val * multiplier


def parse_price(raw: object) -> Tuple[Optional[float], Optional[float]]:
    """Parse a raw PRICE string into (min_price, max_price) in INR.

    Handles:
      - "Price on Request"           -> (None, None)
      - "3.25 Cr"                    -> (32500000, 32500000)
      - "69.3  - 83.7 L"              -> (6930000, 8370000)
      - "19.1 L Onwards"              -> (1910000, None)  (open-ended upper bound)
      - "85,000" / "20,000/Bedroom"   -> per-unit rent style values (kept as-is;
                                          these are typically per-bedroom / per-bed
                                          rentals, flagged separately by the caller)
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, None
    text = str(raw).strip()
    if text == "" or "request" in text.lower():
        return None, None

    onwards = "onwards" in text.lower()
    text = re.sub(r"onwards", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/\s*(bedroom|bed|month|sqft|sq\.?ft\.?)", "", text, flags=re.IGNORECASE)

    # The unit (Cr/L/K) is often stated only once for a whole range, e.g.
    # "69.3 - 83.7 L" -- both numbers share the trailing "L". Detect the
    # single trailing unit for the whole string up front and apply it to
    # both range endpoints if a given token has no unit of its own.
    whole_unit_match = _PRICE_UNIT_RE.search(text)
    whole_unit = whole_unit_match.group(1).lower() if whole_unit_match else None

    if "-" in text:
        parts = [p.strip() for p in text.split("-") if p.strip() != ""]
        if len(parts) == 2:
            lo = parse_price_token(parts[0]) if _PRICE_UNIT_RE.search(parts[0]) else _apply_unit(parts[0], whole_unit)
            hi = parse_price_token(parts[1]) if _PRICE_UNIT_RE.search(parts[1]) else _apply_unit(parts[1], whole_unit)
            return lo, hi
    val = parse_price_token(text)
    if onwards:
        return val, None
    return val, val


def _apply_unit(token: str, unit: Optional[str]) -> Optional[float]:
    num_match = _NUM_RE.search(token)
    if not num_match:
        return None
    val = _to_float(num_match.group())
    if val is None:
        return None
    multiplier = 1.0
    if unit:
        if unit == "cr" or unit == "crore":
            multiplier = _CRORE
        elif unit in ("l", "lac", "lakh"):
            multiplier = _LAKH
        elif unit == "k":
            multiplier = _THOUSAND
    return val * multiplier


def parse_area(raw: object) -> Tuple[Optional[float], Optional[float]]:
    """Parse a raw AREA string ('518-623  sq.ft.', '1089 sq.ft.') -> (min, max) sqft."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, None
    text = str(raw).lower().replace("sq.ft.", "").replace("sq ft", "").strip()
    text = text.replace(",", "")
    if text == "":
        return None, None
    if "-" in text:
        parts = [p.strip() for p in text.split("-") if p.strip() != ""]
        if len(parts) == 2:
            lo = _to_float(parts[0])
            hi = _to_float(parts[1])
            return lo, hi
    val = _to_float(text)
    return val, val


def parse_pydict(raw: object) -> dict:
    """Parse a python-repr'd dict column (location, MAP_DETAILS) safely."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return {}
    try:
        val = ast.literal_eval(text)
        if isinstance(val, dict):
            return val
    except (ValueError, SyntaxError):
        pass
    return {}


def parse_id_list(raw: object) -> list:
    """Parse a comma-separated id list column (FEATURES, AMENITIES) -> list[str]."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    text = str(raw).strip()
    if text == "" or text.upper() in {"N", "P", "NAN"}:
        return []
    ids = [t.strip() for t in text.split(",") if t.strip() != ""]
    return [i for i in ids if i.replace(".", "", 1).isdigit()]


def parse_floor_num(raw: object) -> Optional[int]:
    """Parse FLOOR_NUM: numeric floors kept as-is; 'G'/'Ground' -> 0; 'B'/'Basement' -> -1."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if text == "":
        return None
    low = text.lower()
    if low.startswith("g"):
        return 0
    if low.startswith("b") or low.startswith("l"):  # basement / lower ground
        return -1
    try:
        return int(float(text))
    except ValueError:
        return None
