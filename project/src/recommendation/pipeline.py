"""Assembles a fully-loaded RecommendationOrchestrator from persisted
artifacts on disk (data/processed/*.csv, models/content/*, models/collaborative/*).

Used by scripts/train.py (to fit + save), the API, the Streamlit app, and
the evaluation suite (to load + serve) so there is exactly one code path
for "how do we build the orchestrator", per the project's no-duplicated-logic
principle.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.property_features import CONTENT_MODEL_DIR, load_artifacts as load_content_artifacts
from src.models.collaborative import COLLAB_MODEL_DIR, CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.rule_based import RuleBasedRecommender
from src.recommendation.orchestrator import RecommendationOrchestrator
from src.utils.config import DATA_PROCESSED_DIR, load_config


def load_processed_data(data_dir: Path = DATA_PROCESSED_DIR):
    properties = pd.read_csv(data_dir / "properties.csv")
    users = pd.read_csv(data_dir / "users.csv")
    interactions = pd.read_csv(data_dir / "interactions.csv")
    return properties, users, interactions


def build_orchestrator(
    data_dir: Path = DATA_PROCESSED_DIR,
    content_dir: Path = CONTENT_MODEL_DIR,
    collab_dir: Path = COLLAB_MODEL_DIR,
) -> RecommendationOrchestrator:
    cfg = load_config()
    properties, users, interactions = load_processed_data(data_dir)

    feature_matrix, property_ids, _artifacts = load_content_artifacts(content_dir)
    content_model = ContentBasedRecommender(properties, feature_matrix, property_ids, cfg)

    collab_model = CollaborativeRecommender.load(collab_dir, cfg)

    rule_based = RuleBasedRecommender(properties, interactions, cfg)
    hybrid = HybridRecommender(content_model, collab_model, cfg)

    return RecommendationOrchestrator(
        properties=properties,
        interactions=interactions,
        users=users,
        rule_based=rule_based,
        content_based=content_model,
        collaborative=collab_model,
        hybrid=hybrid,
        cfg=cfg,
    )
