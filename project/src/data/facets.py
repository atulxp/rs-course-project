"""Loads the `data/raw/facets/facets/*.csv` id->label lookup tables.

These facet files are 99acres/Housing-style dropdown-option dumps (mostly
covering the Gurgaon export in full and only partially covering the ids that
appear in the other three city files -- confirmed during dataset inspection,
see docs/data_dictionary.md). We use them as best-effort decoders: any id not
present in the facet table is decoded as "Unknown" / "Other" rather than
silently dropped, and this is documented as a known dataset limitation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.utils.config import DATA_RAW_DIR

FACETS_DIR = DATA_RAW_DIR / "facets" / "facets"


def _load_facet(name: str) -> Dict[str, str]:
    path = FACETS_DIR / f"{name}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["id"].astype(str).str.strip(), df["label"].astype(str).str.strip()))


def load_all_facets() -> Dict[str, Dict[str, str]]:
    names = [
        "AGE",
        "AMENITIES",
        "BATHROOM_NUM",
        "BEDROOM_NUM",
        "FACING_DIRECTION",
        "FEATURES",
        "FURNISH",
        "OWNERSHIP_TYPE",
        "PROPERTY_TYPE",
        "SUB_AVAILABILITY",
        "TOTAL_FLOOR",
    ]
    return {n: _load_facet(n) for n in names}


def decode_id_list(ids: list, facet: Dict[str, str]) -> list:
    """Decode a list of raw ids into labels, falling back to 'Other:<id>'."""
    out = []
    for i in ids:
        label = facet.get(str(i))
        out.append(label if label else f"Other:{i}")
    return out


# Manually curated fallback decoders for small enum-like codes that are only
# consistently present as *raw integer codes* (0/1/2/4 etc.) across all four
# city files, rather than string ids matching the facets/ CSVs one-to-one.
# Derived by cross-checking value distributions against the FURNISH.csv /
# AGE.csv facet files during dataset inspection (see docs/data_dictionary.md).
FURNISH_CODE_MAP = {
    "0": "Not Specified",
    "1": "Furnished",
    "2": "Unfurnished",
    "4": "Semifurnished",
}

AGE_CODE_MAP = {
    "0": "Not Specified",
    "1": "1-5 Year Old Property",
    "2": "5-10 Year Old Property",
    "3": "10+ Year Old Property",
    "5": "Under Construction",
    "6": "Not Specified",
}

# IMPORTANT (see docs/data_dictionary.md "TRANSACT_TYPE" entry): the raw
# TRANSACT_TYPE code does NOT encode Buy vs Rent. Cross-checking the price
# magnitude and SECONDARY_TAGS/LISTING columns against TRANSACT_TYPE during
# EDA showed that BOTH code 1 and code 2 rows are purchase (Buy) listings --
# 1 = Resale, 2 = New/Primary Booking (confirmed by SECONDARY_TAGS containing
# 'RESALE' vs 'NEW BOOKING'/'NEW LAUNCH'). Genuine rental listings in this
# dataset instead have a *missing* TRANSACT_TYPE code combined with a price
# in the monthly-rent numeric range -- handled by
# src/data/prepare.impute_transaction_type(). This mapping is therefore used
# only to populate the auxiliary `listing_status` column, not the buy/rent
# `transaction_type` column.
LISTING_STATUS_MAP = {
    "1.0": "Resale",
    "1": "Resale",
    "2.0": "New Booking",
    "2": "New Booking",
}
