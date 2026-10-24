"""Model 4 -- Hybrid recommender for long-term / short-long-term users
(paper Sections 2.0.3/2.0.4): final_score = alpha*content + (1-alpha)*collaborative.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.utils.config import load_config


class HybridRecommender:
    def __init__(self, content_model: ContentBasedRecommender, collab_model: CollaborativeRecommender, cfg: dict | None = None):
        self.content_model = content_model
        self.collab_model = collab_model
        self.cfg = cfg or load_config()

    def recommend(
        self,
        user_id: str,
        user_interactions: pd.DataFrame,
        city: str,
        transaction_type: str,
        alpha: Optional[float] = None,
        top_k: int = 10,
        only_recent_content: bool = False,
        exclude_property_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        alpha = self.cfg["hybrid"]["default_alpha"] if alpha is None else alpha

        candidates = self.content_model.properties[
            (self.content_model.properties["city"] == city)
            & (self.content_model.properties["transaction_type"] == transaction_type)
        ]
        if exclude_property_ids:
            candidates = candidates[~candidates["property_id"].isin(set(exclude_property_ids))]
        if candidates.empty:
            return pd.DataFrame()
        candidate_ids = candidates["property_id"].tolist()

        content_scores = self.content_model.similarity_scores_for_candidates(
            user_interactions, candidate_ids, only_recent=only_recent_content
        )
        collab_scores = self.collab_model.scores_for_candidates(user_id, candidate_ids)

        all_ids = sorted(set(content_scores.index) | set(collab_scores.index))
        if not all_ids:
            return pd.DataFrame()

        content_aligned = content_scores.reindex(all_ids).fillna(0.0)
        collab_aligned = collab_scores.reindex(all_ids).fillna(0.0)
        final_score = alpha * content_aligned + (1 - alpha) * collab_aligned

        result = candidates.set_index("property_id").loc[all_ids].reset_index()
        result["content_score"] = content_aligned.values
        result["collaborative_score"] = collab_aligned.values
        result["score"] = final_score.values
        result["method"] = "hybrid"
        result["alpha"] = alpha
        return result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
