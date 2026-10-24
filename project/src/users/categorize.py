"""User category classification (paper Section 2; thresholds in config.yaml).

See config.yaml `user_category_thresholds` docstring for why we use
interaction COUNTS (recent vs. total) instead of the paper's literal
wall-clock minute windows: our simulated interactions have no real
timestamps, only an explicit `is_recent` flag per interaction.
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import load_config


def categorize_user(user_id: str, interactions: pd.DataFrame, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    th = cfg["user_category_thresholds"]

    user_inter = interactions[interactions["user_id"] == user_id]
    total = len(user_inter)
    recent = int(user_inter["is_recent"].sum()) if total else 0

    if total <= th["cold_start"]["max_total_interactions"]:
        return "cold_start"

    is_long_term = total >= th["long_term"]["min_total_interactions"]
    is_short_term_recent = recent >= th["short_long_term"]["min_recent_interactions"]

    if is_long_term and is_short_term_recent:
        return "short_long_term"
    if is_long_term:
        return "long_term"
    if recent >= th["short_term"]["min_recent_interactions"]:
        return "short_term"
    return "cold_start"


def categorize_all_users(user_ids: pd.Series, interactions: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    cfg = cfg or load_config()
    counts = interactions.groupby("user_id").agg(total=("interaction_id", "count"), recent=("is_recent", "sum"))
    th = cfg["user_category_thresholds"]

    def _cat(uid):
        if uid not in counts.index:
            return "cold_start"
        total = counts.loc[uid, "total"]
        recent = counts.loc[uid, "recent"]
        is_long_term = total >= th["long_term"]["min_total_interactions"]
        is_short_term_recent = recent >= th["short_long_term"]["min_recent_interactions"]
        if is_long_term and is_short_term_recent:
            return "short_long_term"
        if is_long_term:
            return "long_term"
        if recent >= th["short_term"]["min_recent_interactions"]:
            return "short_term"
        return "cold_start"

    return user_ids.apply(_cat)
