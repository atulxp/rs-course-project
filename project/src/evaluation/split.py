"""Train/test split for evaluation (Section 18 of the brief).

For every user with enough interactions, the most-recently-generated
`test_fraction_per_user` share of their interactions (by `sequence`) is held
out as the TEST set; everything else is TRAIN. Models used during
evaluation (content profile vectors, the ALS matrix, rule-based cohort
popularity) are built ONLY from the TRAIN split, so test interactions never
leak into training -- this is verified by
`tests/test_evaluation.py::test_no_leakage_between_train_and_test`.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.utils.config import load_config


def split_interactions(interactions: pd.DataFrame, cfg: dict | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    ev = cfg["evaluation"]
    min_n = ev["min_interactions_for_eval"]
    frac = ev["test_fraction_per_user"]

    train_parts, test_parts = [], []
    for uid, grp in interactions.groupby("user_id"):
        grp = grp.sort_values("sequence")
        n = len(grp)
        if n < min_n:
            train_parts.append(grp)
            continue
        n_test = max(1, int(round(n * frac)))
        n_test = min(n_test, n - 1)  # always leave >=1 train interaction
        train_parts.append(grp.iloc[: n - n_test])
        test_parts.append(grp.iloc[n - n_test :])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0]
    return train_df, test_df
