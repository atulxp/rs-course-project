"""Helpers for presenting recorded user-property interactions."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


PROPERTY_COLUMNS = [
    "property_id", "prop_heading", "building_name", "property_type", "city",
    "locality", "price", "transaction_type", "bhk", "area_sqft", "furnish",
]


def build_interaction_history(
    interactions: pd.DataFrame,
    properties: pd.DataFrame,
    property_columns: Iterable[str] = PROPERTY_COLUMNS,
) -> pd.DataFrame:
    """Join event rows to property details without collapsing repeated events."""
    if interactions.empty:
        return interactions.copy()

    columns = [column for column in property_columns if column in properties.columns]
    history = interactions.merge(
        properties[columns], on="property_id", how="left", suffixes=("", "_property")
    )
    history["_sequence"] = pd.to_numeric(history.get("sequence"), errors="coerce").fillna(-1)
    tie_breaker = history["interaction_id"] if "interaction_id" in history else pd.Series(range(len(history)))
    latest = (
        history.assign(_tie_breaker=tie_breaker)
        .sort_values(["property_id", "_sequence", "_tie_breaker"])
        .groupby("property_id")["interaction_type"]
        .last()
        .to_dict()
    )
    history["most_recent_interaction"] = history["property_id"].map(latest)
    return history