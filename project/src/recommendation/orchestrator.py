"""Central recommendation orchestrator (Section 16 of the brief).

Given a user_id (+ optional explicit preferences), determines the user's
category (cold_start / short_term / long_term / short_long_term) from their
simulated interaction history and routes to the appropriate model, exactly
following the paper's pipeline (Figure 1):

    cold_start          -> rule-based
    short_term          -> content-based (recent interactions only)
    long_term           -> hybrid (content + collaborative, paper deploys
                            "content+collaborative model for long-term users")
    short_long_term      -> hybrid (content + collaborative, full history)

Returns a list of recommendations, each with property details, a score, the
method used, and a human-readable explanation (Section 17).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.rule_based import RuleBasedRecommender, UserPreference, explain_match, parse_user_row
from src.users.categorize import categorize_user
from src.utils.config import load_config

DISPLAY_COLUMNS = [
    "property_id",
    "city",
    "locality",
    "property_type",
    "bhk",
    "price",
    "area_sqft",
    "furnish",
    "transaction_type",
    "building_name",
    "prop_heading",
]


@dataclass
class RecommendationResult:
    property_id: str
    rank: int
    score: float
    method: str
    explanation: List[str]
    details: dict


class RecommendationOrchestrator:
    def __init__(
        self,
        properties: pd.DataFrame,
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        rule_based: RuleBasedRecommender,
        content_based: ContentBasedRecommender,
        collaborative: CollaborativeRecommender,
        hybrid: HybridRecommender,
        cfg: dict | None = None,
    ):
        self.properties = properties
        self.interactions = interactions
        self.users = users
        self.rule_based = rule_based
        self.content_based = content_based
        self.collaborative = collaborative
        self.hybrid = hybrid
        self.cfg = cfg or load_config()

    # ------------------------------------------------------------------
    def _user_row(self, user_id: str) -> Optional[pd.Series]:
        matches = self.users[self.users["user_id"] == user_id]
        return matches.iloc[0] if len(matches) else None

    def get_user_category(self, user_id: str) -> str:
        return categorize_user(user_id, self.interactions, self.cfg)

    def _already_interacted(self, user_id: str) -> List[str]:
        return self.interactions[self.interactions["user_id"] == user_id]["property_id"].tolist()

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: str,
        city: Optional[str] = None,
        transaction_type: Optional[str] = None,
        explicit_preferences: Optional[dict] = None,
        top_k: int = 10,
        alpha: Optional[float] = None,
    ) -> List[RecommendationResult]:
        user_row = self._user_row(user_id)
        if user_row is None and explicit_preferences is None:
            raise ValueError(f"Unknown user_id '{user_id}' and no explicit_preferences supplied")

        if explicit_preferences:
            base = user_row.to_dict() if user_row is not None else {}
            base.update(explicit_preferences)
            user_pref = parse_user_row(base)
        else:
            user_pref = parse_user_row(user_row)

        city = city or user_pref.city
        transaction_type = transaction_type or user_pref.transaction_type
        user_pref.city = city
        user_pref.transaction_type = transaction_type

        category = self.get_user_category(user_id)
        user_interactions = self.interactions[self.interactions["user_id"] == user_id]
        exclude = self._already_interacted(user_id)

        if category == "cold_start":
            df = self.rule_based.recommend(user_pref, top_k=top_k)
            if df.empty:
                return []
            return self._to_results(df, df["method"].iloc[0], df["explanation"].tolist())

        if category == "short_term":
            df = self.content_based.recommend_for_interactions(
                user_interactions, city, transaction_type, top_k=top_k, only_recent=True, exclude_property_ids=exclude
            )
            if df.empty:
                # fall back to rule-based if the user's recent history doesn't
                # overlap with any candidate property feature (cold-start-like edge case)
                df = self.rule_based.recommend(user_pref, top_k=top_k)
                if df.empty:
                    return []
                return self._to_results(df, "cold_start_rule_based_fallback", df["explanation"].tolist())
            explanations = [
                self._content_explanation(user_pref, row, user_interactions)
                for _, row in df.iterrows()
            ]
            return self._to_results(df, "content_based", explanations)

        if category in ("long_term", "short_long_term"):
            only_recent = category == "short_long_term"
            df = self.hybrid.recommend(
                user_id,
                user_interactions,
                city,
                transaction_type,
                alpha=alpha,
                top_k=top_k,
                only_recent_content=False,  # long-term hybrid uses full history for content too
                exclude_property_ids=exclude,
            )
            if df.empty:
                df = self.rule_based.recommend(user_pref, top_k=top_k)
                if df.empty:
                    return []
                return self._to_results(df, "cold_start_rule_based_fallback", df["explanation"].tolist())
            explanations = [
                self._hybrid_explanation(user_pref, row, user_interactions)
                for _, row in df.iterrows()
            ]
            return self._to_results(df, "hybrid", explanations)

        raise RuntimeError(f"Unknown category: {category}")

    # ------------------------------------------------------------------
    def _content_explanation(
        self, user_pref: UserPreference, row: pd.Series, user_interactions: pd.DataFrame
    ) -> List[str]:
        base = explain_match(user_pref, row)
        if float(row.get("score", 0.0)) > 0 and not user_interactions.empty:
            types = set(user_interactions["interaction_type"].dropna().astype(str))
            labels = []
            if "detail_view" in types:
                labels.append("viewed")
            if "shortlist" in types:
                labels.append("shortlisted")
            if "lead" in types:
                labels.append("generated a lead for")
            if labels:
                return base + [
                    "Similar to properties you " + ", ".join(labels) + ""
                ]
            return base + ["Similar to properties in your interaction history"]
        return base

    def _hybrid_explanation(
        self, user_pref: UserPreference, row: pd.Series, user_interactions: pd.DataFrame
    ) -> List[str]:
        base = explain_match(user_pref, row)
        reasons = list(base)
        if float(row.get("content_score", 0.0)) > 0 and not user_interactions.empty:
            reasons.append("Its features are similar to properties in your interaction history")
        if float(row.get("collaborative_score", 0.0)) > 0:
            reasons.append("Supported by patterns learned from similar users")
        if len(reasons) == len(base):
            reasons.append("Ranked by the hybrid recommendation strategy")
        return reasons

    def _to_results(self, df: pd.DataFrame, method: str, explanations: List[List[str]]) -> List[RecommendationResult]:
        results = []
        for rank, (i, row) in enumerate(df.iterrows(), start=1):
            details = {c: row.get(c) for c in DISPLAY_COLUMNS if c in row}
            results.append(
                RecommendationResult(
                    property_id=row["property_id"],
                    rank=rank,
                    score=float(row.get("score", 0.0)),
                    method=method if isinstance(method, str) else row.get("method", method),
                    explanation=explanations[rank - 1] if rank - 1 < len(explanations) else [],
                    details=details,
                )
            )
        return results
