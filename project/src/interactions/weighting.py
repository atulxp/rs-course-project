"""Interaction -> numeric score weighting schemes.

The paper (Section 3.0.4, Table 3) compares three weighting schemes for
training the collaborative-filtering (ALS) model on real Housing.com click
data: linear weighting (their production choice), exponential decay, and
TF-IDF weighting. We reproduce the same three schemes here so the
paper-inspired ablation experiment (Section 20 of the brief) can be run on
our simulated interactions, while being explicit this is a *simulated*
evaluation environment, not a reproduction of the paper's numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config


def linear_weighting(interactions: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """weight = interaction_weights[type] (paper's production scheme)."""
    cfg = cfg or load_config()
    weights_cfg = cfg["interaction_weights"]
    return interactions["interaction_type"].map(weights_cfg).astype(float)


def exponential_decay_weighting(interactions: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """weight = base_weight * 0.5**(rank_from_most_recent / half_life).

    Our simulated interactions do not carry real timestamps, so "recency" is
    approximated using each interaction's `sequence` position within its
    user's interaction history (lower sequence number = generated earlier =
    "older"), with half-life expressed in *interaction-count* units instead
    of days (documented adaptation of the paper's literal 3-day half-life,
    since we have no wall-clock click timestamps to decay against).
    """
    cfg = cfg or load_config()
    base = interactions["interaction_type"].map(cfg["interaction_weights"]).astype(float)
    half_life = cfg["weighting_schemes"]["exponential_decay"]["half_life_days"]

    def _decay_group(g: pd.DataFrame) -> pd.Series:
        max_seq = g["sequence"].max()
        age = max_seq - g["sequence"]
        return base.loc[g.index] * (0.5 ** (age / max(half_life, 1)))

    return interactions.groupby("user_id", group_keys=False).apply(_decay_group)


def tfidf_weighting(interactions: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """weight = base_weight * log(total_interactions / interactions_on_property)."""
    cfg = cfg or load_config()
    base = interactions["interaction_type"].map(cfg["interaction_weights"]).astype(float)
    total = len(interactions)
    counts_per_property = interactions.groupby("property_id")["property_id"].transform("count")
    idf = np.log(total / counts_per_property.clip(lower=1))
    return base * idf


WEIGHTING_FUNCS = {
    "linear": linear_weighting,
    "exponential_decay": exponential_decay_weighting,
    "tfidf": tfidf_weighting,
}


def compute_weights(interactions: pd.DataFrame, scheme: str, cfg: dict | None = None) -> pd.Series:
    if scheme not in WEIGHTING_FUNCS:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return WEIGHTING_FUNCS[scheme](interactions, cfg)
