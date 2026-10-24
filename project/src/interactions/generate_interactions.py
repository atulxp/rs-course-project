"""Synthetic interaction generation (Section 9 of the brief).

ACADEMIC INTEGRITY NOTE: every interaction produced here is SIMULATED. There
is no real clickstream in the attached dataset. We do not claim these
interactions are observed user behaviour -- they are a transparent,
config-driven simulation used purely to exercise/evaluate the recommendation
models (content-based, collaborative, hybrid) in place of the proprietary
Housing.com clickstream used by the original paper.

Simulation logic
-----------------
For each synthetic user (src/users/generate_users.py) we:
  1. Determine how many interactions to generate based on the user's target
     category (cold_start / short_term / long_term / short_long_term),
     sampled per config `synthetic.interactions_per_user` and
     `synthetic.category_split`.
  2. Build a candidate pool of properties matching the user's city +
     transaction_type.
  3. Compute a "preference match score" in [0, 1] for every candidate
     property against the user's stated preferences (BHK, budget, area,
     furnishing, locality, property type, amenities) -- reusing the exact
     same scoring function as the rule-based recommender
     (src/models/rule_based.py:score_property) so the simulation and the
     cold-start baseline are consistent with each other.
  4. Sample interacted properties with probability proportional to
     match_score + noise (so users mostly-but-not-exclusively interact with
     well-matching properties -- `synthetic.negative_sample_ratio` controls
     how much "exploration" of non-matching properties is injected, which
     is what makes collaborative filtering learn something beyond the rule
     engine).
  5. For each interacted property, sample an interaction TYPE (impression /
     detail_view / image_view / amenity_view / shortlist / lead /
     conversion) with higher-match properties more likely to receive
     higher-intent interaction types -- mirroring the paper's premise that
     higher engagement/conversion correlates with better preference fit.
  6. Interactions are also tagged `is_recent` (a fixed number of the most
     recently generated interactions per user, used by the short-term /
     long-term split -- see docs/data_dictionary.md and config.yaml
     `user_category_thresholds` for why this replaces the paper's literal
     10-minute recency window).

Run as:
    python -m src.interactions.generate_interactions
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.rule_based import score_property, parse_user_row
from src.utils.config import DATA_PROCESSED_DIR, load_config

INTERACTION_TYPES_BY_INTENT = [
    # (interaction_type, relative likelihood weight when match_score is LOW)
    "impression",
    "image_view",
    "amenity_view",
    "detail_view",
    "shortlist",
    "lead",
    "conversion",
]


def _sample_interaction_type(rng: np.random.Generator, match_score: float) -> str:
    """Higher match_score biases sampling toward higher-intent interaction types."""
    n = len(INTERACTION_TYPES_BY_INTENT)
    # base weights are uniform-ish; skew toward the high-intent end as
    # match_score grows, using a simple linear tilt.
    idx = np.arange(n)
    tilt = match_score * 2.5  # 0 (no tilt) .. 2.5 (strong tilt toward high intent)
    weights = np.exp(tilt * idx / (n - 1))
    weights = weights / weights.sum()
    choice = rng.choice(n, p=weights)
    return INTERACTION_TYPES_BY_INTENT[choice]


def _target_category(rng: np.random.Generator, cfg: dict) -> str:
    split = cfg["synthetic"]["category_split"]
    cats = list(split.keys())
    probs = list(split.values())
    return rng.choice(cats, p=probs)


def _num_interactions_for_category(rng: np.random.Generator, category: str, cfg: dict) -> int:
    spec = cfg["synthetic"]["interactions_per_user"][category]
    if isinstance(spec, list):
        lo, hi = spec
        return int(rng.integers(lo, hi + 1))
    return int(spec)


def generate_interactions(users: pd.DataFrame, properties: pd.DataFrame, cfg: dict | None = None, seed: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = seed if seed is not None else cfg["seed"]
    rng = np.random.default_rng(seed)

    noise_std = cfg["synthetic"]["match_score_noise_std"]
    neg_ratio = cfg["synthetic"]["negative_sample_ratio"]

    props_by_slice: dict[tuple, pd.DataFrame] = {}
    for (city, tx), grp in properties.groupby(["city", "transaction_type"]):
        props_by_slice[(city, tx)] = grp.reset_index(drop=True)

    records = []
    interaction_id = 0

    for _, user_row in users.iterrows():
        user_pref = parse_user_row(user_row)
        category = _target_category(rng, cfg)
        n_interactions = _num_interactions_for_category(rng, category, cfg)
        if n_interactions == 0:
            continue

        key = (user_row["city"], user_row["transaction_type"])
        candidates = props_by_slice.get(key)
        if candidates is None or len(candidates) == 0:
            continue

        pool_size = min(len(candidates), max(50, n_interactions * 10))
        pool = candidates.sample(n=pool_size, random_state=int(rng.integers(0, 1_000_000)))

        match_scores = pool.apply(lambda r: score_property(user_pref, r), axis=1).values
        noisy_scores = np.clip(match_scores + rng.normal(0, noise_std, size=len(match_scores)), 0.0, 1.0)

        n_negative = int(round(n_interactions * neg_ratio))
        n_positive = max(n_interactions - n_negative, 0)

        pos_order = np.argsort(-noisy_scores)
        positive_idx = pos_order[:n_positive]
        remaining_idx = pos_order[n_positive:]
        n_negative = min(n_negative, len(remaining_idx))
        negative_idx = rng.choice(remaining_idx, size=n_negative, replace=False) if n_negative > 0 else np.array([], dtype=int)

        chosen_idx = np.concatenate([positive_idx, negative_idx]).astype(int)
        rng.shuffle(chosen_idx)

        # Recency flag: by construction, a "short_term" simulated user's whole
        # history IS the recent window; a "long_term" user's history is all
        # historical (no recent activity, per the paper's >10-minutes-old
        # definition); a "short_long_term" user has a mix of both, so we mark
        # the most-recently-generated ~30% of their interactions as recent.
        if category == "short_term":
            recent_flags = np.ones(len(chosen_idx), dtype=bool)
        elif category == "long_term":
            recent_flags = np.zeros(len(chosen_idx), dtype=bool)
        elif category == "short_long_term":
            n_recent = max(1, int(round(len(chosen_idx) * 0.3)))
            recent_flags = np.zeros(len(chosen_idx), dtype=bool)
            recent_flags[-n_recent:] = True
        else:
            recent_flags = np.zeros(len(chosen_idx), dtype=bool)

        for order_pos, (idx, is_recent) in enumerate(zip(chosen_idx, recent_flags)):
            prop = pool.iloc[idx]
            score = float(np.clip(match_scores[idx], 0.0, 1.0))
            itype = _sample_interaction_type(rng, score)
            records.append(
                {
                    "interaction_id": interaction_id,
                    "user_id": user_row["user_id"],
                    "property_id": prop["property_id"],
                    "interaction_type": itype,
                    "match_score": round(score, 4),
                    "is_recent": bool(is_recent),
                    "sequence": order_pos,
                    "target_category": category,
                }
            )
            interaction_id += 1

    return pd.DataFrame.from_records(records)


def run(out_dir: Path = DATA_PROCESSED_DIR) -> pd.DataFrame:
    properties = pd.read_csv(out_dir / "properties.csv")
    users = pd.read_csv(out_dir / "users.csv")
    cfg = load_config()
    interactions = generate_interactions(users, properties, cfg)
    out_path = out_dir / "interactions.csv"
    interactions.to_csv(out_path, index=False)
    print(f"[generate_interactions] wrote {len(interactions)} synthetic (simulated) interactions -> {out_path}")
    print(interactions["interaction_type"].value_counts())
    print(f"users with >=1 interaction: {interactions['user_id'].nunique()} / {len(users)}")
    return interactions


if __name__ == "__main__":
    run()
