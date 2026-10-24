"""Model 2 -- Content-based recommender for short-term users (paper Section 2.0.2).

User profile vector = weighted sum of the property feature vectors the user
has interacted with, where the weight of each interaction is
`interaction_weights[interaction_type]` from config.yaml (the paper's
Table 1 conversion-rate-derived weighting scheme, adapted -- see
config.yaml comments). The profile vector lives in exactly the same feature
space as the property vectors (src/features/property_features.py), so
cosine similarity between the two is meaningful.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.utils.config import load_config


def build_user_profile_vector(
    user_interactions: pd.DataFrame,
    property_id_to_row: dict,
    feature_matrix: sp.csr_matrix,
    cfg: dict | None = None,
    only_recent: bool = False,
) -> Optional[sp.csr_matrix]:
    """Weighted sum of interacted-property feature vectors -> 1 x D profile vector."""
    cfg = cfg or load_config()
    weights_cfg = cfg["interaction_weights"]

    df = user_interactions
    if only_recent:
        df = df[df["is_recent"]]
    if df.empty:
        return None

    rows = []
    weights = []
    for _, r in df.iterrows():
        row_idx = property_id_to_row.get(r["property_id"])
        if row_idx is None:
            continue
        rows.append(row_idx)
        weights.append(weights_cfg.get(r["interaction_type"], 1))

    if not rows:
        return None

    sub = feature_matrix[rows]
    w = np.array(weights, dtype=float).reshape(-1, 1)
    profile = sub.multiply(w).sum(axis=0)
    profile = sp.csr_matrix(profile)
    return normalize(profile)


class ContentBasedRecommender:
    def __init__(self, properties: pd.DataFrame, feature_matrix: sp.csr_matrix, property_ids: List[str], cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.properties = properties.reset_index(drop=True)
        self.feature_matrix = feature_matrix
        self.property_ids = property_ids
        self.property_id_to_row = {pid: i for i, pid in enumerate(property_ids)}
        self.row_to_property_id = {i: pid for i, pid in enumerate(property_ids)}

    def recommend_for_interactions(
        self,
        user_interactions: pd.DataFrame,
        city: str,
        transaction_type: str,
        top_k: int = 10,
        only_recent: bool = False,
        exclude_property_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        profile = build_user_profile_vector(
            user_interactions, self.property_id_to_row, self.feature_matrix, self.cfg, only_recent=only_recent
        )
        candidates = self.properties[
            (self.properties["city"] == city) & (self.properties["transaction_type"] == transaction_type)
        ]
        if exclude_property_ids:
            candidates = candidates[~candidates["property_id"].isin(set(exclude_property_ids))]
        if candidates.empty or profile is None:
            return pd.DataFrame()

        cand_rows = [self.property_id_to_row[p] for p in candidates["property_id"] if p in self.property_id_to_row]
        if not cand_rows:
            return pd.DataFrame()
        cand_matrix = self.feature_matrix[cand_rows]
        sims = cosine_similarity(profile, cand_matrix).flatten()

        result = candidates[candidates["property_id"].isin([self.row_to_property_id[r] for r in cand_rows])].copy()
        # re-align order: candidates filtered above may not be in cand_rows order
        result = result.set_index("property_id").loc[[self.row_to_property_id[r] for r in cand_rows]].reset_index()
        result["score"] = sims
        result["method"] = "content_based"
        return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    def similarity_scores_for_candidates(
        self, user_interactions: pd.DataFrame, candidate_property_ids: List[str], only_recent: bool = False
    ) -> pd.Series:
        """Return a Series[property_id -> cosine similarity] for hybrid combination."""
        profile = build_user_profile_vector(
            user_interactions, self.property_id_to_row, self.feature_matrix, self.cfg, only_recent=only_recent
        )
        if profile is None:
            return pd.Series(dtype=float)
        rows = [self.property_id_to_row[p] for p in candidate_property_ids if p in self.property_id_to_row]
        if not rows:
            return pd.Series(dtype=float)
        sims = cosine_similarity(profile, self.feature_matrix[rows]).flatten()
        pids = [self.row_to_property_id[r] for r in rows]
        return pd.Series(sims, index=pids)
