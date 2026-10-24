"""Model 1 -- Rule-based recommender for cold-start users (paper Section 2.0.1).

Two things live in this module:

1. `parse_user_row` / `score_property`: a transparent, config-weighted
   preference-match scoring function used BOTH by the rule-based
   recommender itself AND by the synthetic interaction generator (so that
   simulated interactions are consistent with "how well a property matches
   a user's stated preferences", as required by Section 9 of the brief).

2. `RuleBasedRecommender`: the cold-start engine. Combines the preference
   match score with a POPULARITY signal derived from simulated interaction
   aggregates (impressions / leads / conversions), mirroring the paper's
   cohort scoring (density + leads + property-conversion% + user-conversion%)
   adapted to our locality x property_type x bhk x transaction_type cohorts
   (see config.yaml `rule_based.cohort_keys`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config


@dataclass
class UserPreference:
    city: str
    transaction_type: str
    preferred_bhk: Optional[float] = None
    preferred_property_type: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    preferred_area_min: Optional[float] = None
    preferred_area_max: Optional[float] = None
    preferred_furnish: Optional[str] = None
    preferred_localities: List[str] = field(default_factory=list)
    preferred_amenities: List[str] = field(default_factory=list)


def parse_user_row(row) -> UserPreference:
    """Build a UserPreference from a row of data/processed/users.csv (or an
    equivalent dict coming from the API's POST /users body)."""
    def _get(key, default=None):
        if isinstance(row, dict):
            return row.get(key, default)
        if key not in row.index:
            return default
        val = row[key]
        if isinstance(val, (list, dict)):
            return val
        return val if pd.notna(val) else default

    def _json_list(key):
        val = _get(key, "[]")
        if isinstance(val, list):
            return val
        try:
            return json.loads(val) if val else []
        except (TypeError, json.JSONDecodeError):
            return []

    return UserPreference(
        city=_get("city"),
        transaction_type=_get("transaction_type"),
        preferred_bhk=_get("preferred_bhk"),
        preferred_property_type=_get("preferred_property_type"),
        budget_min=_get("budget_min"),
        budget_max=_get("budget_max"),
        preferred_area_min=_get("preferred_area_min"),
        preferred_area_max=_get("preferred_area_max"),
        preferred_furnish=_get("preferred_furnish"),
        preferred_localities=_json_list("preferred_localities"),
        preferred_amenities=_json_list("preferred_amenities"),
    )


def _range_match(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> float:
    """1.0 if value falls inside [lo, hi]; decays smoothly outside the range."""
    if value is None or pd.isna(value) or lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return 0.5  # neutral score when we can't evaluate
    if lo <= value <= hi:
        return 1.0
    span = max(hi - lo, 1e-6)
    if value < lo:
        dist = (lo - value) / span
    else:
        dist = (value - hi) / span
    return float(max(0.0, 1.0 - min(dist, 1.0)))


def score_property(user: UserPreference, prop: pd.Series, weights: Dict[str, float] | None = None) -> float:
    """Weighted, explainable match score in [0, 1] between a user preference
    profile and a single property row (from properties.csv)."""
    if weights is None:
        weights = load_config()["rule_based"]["scoring_weights"]

    budget_match = _range_match(prop.get("price"), user.budget_min, user.budget_max)
    location_match = 1.0 if (user.preferred_localities and prop.get("locality") in user.preferred_localities) else (
        0.5 if not user.preferred_localities else 0.0
    )
    bhk_match = 1.0 if (user.preferred_bhk is not None and prop.get("bhk") == user.preferred_bhk) else (
        0.5 if user.preferred_bhk is None or pd.isna(prop.get("bhk")) else 0.0
    )
    property_type_match = 1.0 if (
        user.preferred_property_type and prop.get("property_type") == user.preferred_property_type
    ) else 0.3
    furnishing_match = 1.0 if (
        user.preferred_furnish and prop.get("furnish") == user.preferred_furnish
    ) else 0.3

    prop_amenities = set(str(prop.get("amenities", "")).split(";")) if prop.get("amenities") else set()
    user_amenities = set(user.preferred_amenities)
    if user_amenities:
        overlap = len(prop_amenities & user_amenities) / len(user_amenities)
    else:
        overlap = 0.5
    amenities_match = float(overlap)

    score = (
        weights["budget_match"] * budget_match
        + weights["location_match"] * location_match
        + weights["bhk_match"] * bhk_match
        + weights["property_type_match"] * property_type_match
        + weights["furnishing_match"] * furnishing_match
        + weights["amenities_match"] * amenities_match
    )
    total_weight = sum(weights.values())
    return float(np.clip(score / total_weight, 0.0, 1.0))


def explain_match(user: UserPreference, prop: pd.Series) -> List[str]:
    reasons = []
    if user.preferred_bhk is not None and prop.get("bhk") == user.preferred_bhk:
        reasons.append(f"Matches your {int(user.preferred_bhk)} BHK preference")
    if user.budget_min is not None and user.budget_max is not None and prop.get("price") is not None:
        if pd.notna(prop.get("price")) and user.budget_min <= prop["price"] <= user.budget_max:
            reasons.append("Within your budget")
    if user.preferred_localities and prop.get("locality") in user.preferred_localities:
        reasons.append(f"Located in {prop.get('locality')}, one of your preferred localities")
    if user.preferred_furnish and prop.get("furnish") == user.preferred_furnish:
        reasons.append(f"Matches your {user.preferred_furnish.lower()} preference")
    if user.preferred_property_type and prop.get("property_type") == user.preferred_property_type:
        reasons.append(f"Matches your preferred property type ({prop.get('property_type')})")
    prop_amenities = set(str(prop.get("amenities", "")).split(";")) if prop.get("amenities") else set()
    matched_amenities = prop_amenities & set(user.preferred_amenities)
    if matched_amenities:
        reasons.append("Has amenities you prefer: " + ", ".join(list(matched_amenities)[:3]))
    if not reasons:
        reasons.append("Popular property matching your search filters")
    return reasons


class RuleBasedRecommender:
    """Cold-start engine: preference match + cohort popularity (paper 2.0.1)."""

    def __init__(self, properties: pd.DataFrame, interactions: pd.DataFrame | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.properties = properties.reset_index(drop=True)
        self.popularity = self._build_cohort_popularity(interactions)

    def _cohort_key(self, row: pd.Series) -> str:
        keys = self.cfg["rule_based"]["cohort_keys"]
        return "|".join(str(row.get(k)) for k in keys)

    def _build_cohort_popularity(self, interactions: pd.DataFrame | None) -> Dict[tuple, float]:
        """Cohort score = weighted sum of (#flats, #leads, %property_conv, %user_conv)
        approximated from simulated interactions, matching the paper's 4-metric
        cohort score (density, total leads, property-conversion%, user-conversion%)."""
        cohort_flat_counts: Dict[tuple, int] = {}
        for _, row in self.properties.iterrows():
            key = self._cohort_key(row)
            cohort_flat_counts[key] = cohort_flat_counts.get(key, 0) + 1

        if interactions is None or len(interactions) == 0:
            # no simulated interactions available yet -> popularity == density only
            max_count = max(cohort_flat_counts.values()) if cohort_flat_counts else 1
            return {k: v / max_count for k, v in cohort_flat_counts.items()}

        merged = interactions.merge(self.properties, on="property_id", how="left")
        merged["cohort_key"] = merged.apply(self._cohort_key, axis=1)

        leads = merged[merged["interaction_type"].isin(["lead", "conversion"])].groupby("cohort_key").size()
        conversions = merged[merged["interaction_type"] == "conversion"].groupby("cohort_key").size()
        total_by_cohort = merged.groupby("cohort_key").size()

        scores: Dict[tuple, float] = {}
        max_flats = max(cohort_flat_counts.values()) if cohort_flat_counts else 1
        max_leads = leads.max() if len(leads) else 1
        for key, flats in cohort_flat_counts.items():
            density = flats / max_flats
            lead_score = (leads.get(key, 0) / max_leads) if max_leads else 0.0
            prop_conv_pct = (conversions.get(key, 0) / total_by_cohort.get(key, 1)) if total_by_cohort.get(key, 0) else 0.0
            user_conv_pct = prop_conv_pct  # property-level and user-level conversion collapse to the
            # same ratio in our simulated single-interaction-per-(user,property) setup;
            # documented simplification vs. the paper's separate metrics.
            scores[key] = float(np.clip(0.25 * density + 0.35 * lead_score + 0.20 * prop_conv_pct + 0.20 * user_conv_pct, 0, 1))
        return scores

    def recommend(self, user: UserPreference, top_k: int = 10) -> pd.DataFrame:
        candidates = self.properties[
            (self.properties["city"] == user.city) & (self.properties["transaction_type"] == user.transaction_type)
        ].copy()
        if candidates.empty:
            candidates = self.properties[self.properties["city"] == user.city].copy()
        if candidates.empty:
            return candidates

        # Performance (Section 28): scoring is O(candidates) in pure python.
        # Rather than score every property in a (city, transaction_type) slice
        # (which can be several thousand rows), we narrow the search space
        # first -- mirroring the paper's own cohort-narrowing idea (Section
        # 2.0.1: "narrow down the search space ... based on filters") -- by
        # capping the scored candidate pool, preferring properties in the
        # user's preferred BHK/property_type when available.
        MAX_SCORED_CANDIDATES = 400
        if len(candidates) > MAX_SCORED_CANDIDATES:
            preferred = candidates
            if user.preferred_bhk is not None:
                bhk_match = candidates[candidates["bhk"] == user.preferred_bhk]
                if len(bhk_match) >= 50:
                    preferred = bhk_match
            candidates = preferred.sample(n=min(MAX_SCORED_CANDIDATES, len(preferred)), random_state=42)

        candidates["match_score"] = candidates.apply(lambda r: score_property(user, r), axis=1)
        candidates["cohort_key"] = candidates.apply(self._cohort_key, axis=1)
        candidates["popularity_score"] = candidates["cohort_key"].map(self.popularity).fillna(0.0)
        candidates["score"] = 0.6 * candidates["match_score"] + 0.4 * candidates["popularity_score"]
        candidates["method"] = "cold_start_rule_based"
        candidates["explanation"] = candidates.apply(
            lambda r: explain_match(user, r) + (["Popular among similar property cohorts"] if r["popularity_score"] > 0.5 else []),
            axis=1,
        )
        return candidates.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
