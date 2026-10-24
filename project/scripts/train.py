#!/usr/bin/env python3
"""End-to-end training script.

Run:
    python scripts/train.py

Produces every persisted artifact the application needs to start WITHOUT
retraining:
    data/processed/properties.csv
    data/processed/users.csv
    data/processed/interactions.csv
    models/content/feature_matrix.npz, property_ids.json, encoders.joblib
    models/collaborative/als_model.joblib, user_to_idx.json, item_to_idx.json,
                          user_item_matrix.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.prepare import run as run_prepare
from src.users.generate_users import run as run_generate_users
from src.interactions.generate_interactions import run as run_generate_interactions
from src.features.property_features import build_property_features, save_artifacts as save_content_artifacts
from src.models.collaborative import CollaborativeRecommender
from src.utils.config import DATA_PROCESSED_DIR, load_config


def main():
    t0 = time.time()
    cfg = load_config()
    print("=" * 70)
    print("STEP 1/5 -- Data preprocessing (raw CSVs -> processed/properties.csv)")
    print("=" * 70)
    properties = run_prepare()
    # reload from disk so amenities/features are the canonical ';'-joined
    # strings (the in-memory frame from run_prepare still has python lists)
    properties = pd.read_csv(DATA_PROCESSED_DIR / "properties.csv")

    print("\n" + "=" * 70)
    print("STEP 2/5 -- Synthetic user generation")
    print("=" * 70)
    users = run_generate_users()

    print("\n" + "=" * 70)
    print("STEP 3/5 -- Synthetic interaction generation")
    print("=" * 70)
    interactions = run_generate_interactions()

    print("\n" + "=" * 70)
    print("STEP 4/5 -- Content-based feature engineering")
    print("=" * 70)
    feature_matrix, property_ids, artifacts = build_property_features(properties, cfg)
    save_content_artifacts(feature_matrix, property_ids, artifacts)
    print(f"[train] content feature matrix: {feature_matrix.shape} "
          f"(categorical={artifacts.block_sizes['categorical']}, "
          f"amenities={artifacts.block_sizes['amenities']}, text={artifacts.block_sizes['text']})")

    print("\n" + "=" * 70)
    print("STEP 5/5 -- Collaborative filtering (ALS) training")
    print("=" * 70)
    collab = CollaborativeRecommender(cfg)
    collab.fit(interactions, weighting_scheme="linear")
    collab.save()
    print(f"[train] ALS trained: {len(collab.user_to_idx)} users x {len(collab.item_to_idx)} items, "
          f"factors={cfg['collaborative']['factors']}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(f"  properties : {len(properties)}")
    print(f"  users      : {len(users)}")
    print(f"  interactions: {len(interactions)}")
    print("\nArtifacts written to models/content/ and models/collaborative/.")
    print("Next: python scripts/evaluate.py   (or)   uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
