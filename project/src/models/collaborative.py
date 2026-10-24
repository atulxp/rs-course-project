"""Model 3 -- Collaborative filtering for long-term users (paper Section 2.0.3).

Uses Alternating Least Squares (ALS) matrix factorization exactly as the
paper does, but via the `implicit` library (a well-maintained, single-machine
ALS implementation) instead of Apache Spark ML, since Spark is unnecessary
infrastructure for a laptop-scale academic project (this dataset has ~38K
properties and ~1-2K simulated users, vs. the paper's 3M/1.1M -- Spark's
distributed-cluster design buys us nothing at this scale and would violate
the brief's "do not over-engineer" principle).

The user-item interaction matrix is built with entries = interaction weight
(paper: "if a user has submitted the CRF form then we fill the score with
value of 10 ... if user have just scrolled the property page then ... 1").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.interactions.weighting import compute_weights
from src.utils.config import MODELS_DIR, load_config

COLLAB_MODEL_DIR = MODELS_DIR / "collaborative"


def build_interaction_matrix(
    interactions: pd.DataFrame, weighting_scheme: str = "linear", cfg: dict | None = None
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[str, int]]:
    cfg = cfg or load_config()
    weights = compute_weights(interactions, weighting_scheme, cfg)

    user_ids = sorted(interactions["user_id"].unique())
    item_ids = sorted(interactions["property_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {p: i for i, p in enumerate(item_ids)}

    rows = interactions["user_id"].map(user_to_idx).values
    cols = interactions["property_id"].map(item_to_idx).values
    data = weights.values

    # duplicate (user, property) interactions are summed -- multiple
    # simulated interactions on the same property increase confidence.
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids))).tocsr()
    matrix.sum_duplicates()
    return matrix, user_to_idx, item_to_idx


class CollaborativeRecommender:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.model: AlternatingLeastSquares | None = None
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_item_matrix: sp.csr_matrix | None = None

    def fit(self, interactions: pd.DataFrame, weighting_scheme: str = "linear") -> "CollaborativeRecommender":
        matrix, user_to_idx, item_to_idx = build_interaction_matrix(interactions, weighting_scheme, self.cfg)
        cc = self.cfg["collaborative"]
        model = AlternatingLeastSquares(
            factors=cc["factors"],
            regularization=cc["regularization"],
            iterations=cc["iterations"],
            random_state=cc["random_state"],
        )
        confidence_matrix = (matrix * cc["alpha"]).astype(np.float32)
        model.fit(confidence_matrix)

        self.model = model
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}
        self.user_item_matrix = confidence_matrix
        return self

    def can_recommend_for(self, user_id: str) -> bool:
        return self.model is not None and user_id in self.user_to_idx

    def recommend(self, user_id: str, top_k: int = 10, filter_already_interacted: bool = True) -> pd.DataFrame:
        if not self.can_recommend_for(user_id):
            return pd.DataFrame(columns=["property_id", "score"])
        uidx = self.user_to_idx[user_id]
        user_row = self.user_item_matrix[uidx]
        item_idxs, scores = self.model.recommend(
            uidx, user_row, N=top_k, filter_already_liked_items=filter_already_interacted
        )
        pids = [self.idx_to_item[i] for i in item_idxs]
        return pd.DataFrame({"property_id": pids, "score": scores, "method": "collaborative"})

    def scores_for_candidates(self, user_id: str, candidate_property_ids: List[str]) -> pd.Series:
        """Raw dot-product scores (user_factor . item_factor) for a fixed candidate
        set, used by the hybrid model. Not filtered/ranked."""
        if not self.can_recommend_for(user_id) or self.model is None:
            return pd.Series(dtype=float)
        uidx = self.user_to_idx[user_id]
        user_vec = self.model.user_factors[uidx]
        item_vecs = self.model.item_factors
        idxs = [self.item_to_idx[p] for p in candidate_property_ids if p in self.item_to_idx]
        if not idxs:
            return pd.Series(dtype=float)
        scores = item_vecs[idxs] @ user_vec
        pids = [self.idx_to_item[i] for i in idxs]
        s = pd.Series(scores, index=pids)
        # min-max normalize to [0, 1] for fair combination with cosine-similarity content scores
        if s.max() > s.min():
            s = (s - s.min()) / (s.max() - s.min())
        else:
            s = s * 0.0
        return s

    def save(self, out_dir: Path = COLLAB_MODEL_DIR) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_dir / "als_model.joblib")
        with open(out_dir / "user_to_idx.json", "w") as f:
            json.dump(self.user_to_idx, f)
        with open(out_dir / "item_to_idx.json", "w") as f:
            json.dump(self.item_to_idx, f)
        sp.save_npz(out_dir / "user_item_matrix.npz", self.user_item_matrix)

    @classmethod
    def load(cls, in_dir: Path = COLLAB_MODEL_DIR, cfg: dict | None = None) -> "CollaborativeRecommender":
        rec = cls(cfg)
        rec.model = joblib.load(in_dir / "als_model.joblib")
        with open(in_dir / "user_to_idx.json") as f:
            rec.user_to_idx = json.load(f)
        with open(in_dir / "item_to_idx.json") as f:
            rec.item_to_idx = json.load(f)
        rec.idx_to_item = {v: k for k, v in rec.item_to_idx.items()}
        rec.user_item_matrix = sp.load_npz(in_dir / "user_item_matrix.npz")
        return rec
