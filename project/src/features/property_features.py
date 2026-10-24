"""Property feature representation (paper Section 2.0.2 adapted).

The paper converts numerical property features into bins and builds a
binary/categorical vector per property, then a matching user profile
vector, compared via cosine similarity. We follow the same idea but derive
bin widths from OUR data's actual distributions (see docs/data_dictionary.md)
instead of copying the paper's Housing.com-specific bin widths blindly where
they would be inappropriate, while reusing them where they transfer
(area/price bin gaps, priced in the same INR units).

Produces, for the whole processed property table:
  * a categorical one-hot block   (property_type, furnish, age bucket,
                                    price bucket, area bucket, floor bucket,
                                    bhk, city, transaction_type)
  * a multi-hot amenities block
  * a TF-IDF block over the free-text description
all L2-normalized and horizontally stacked into one sparse CSR matrix, in
the same row order as the input DataFrame, so cosine similarity between any
two rows (or between a user profile vector built the same way) is
meaningful.

Persisted artifacts (models/content/):
  feature_matrix.npz        - scipy sparse CSR matrix, one row per property
  property_ids.json         - row order -> property_id
  encoders.joblib           - fitted OneHotEncoder + bin edges + TF-IDF vectorizer
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

from src.utils.config import MODELS_DIR, load_config

CONTENT_MODEL_DIR = MODELS_DIR / "content"

CATEGORICAL_COLUMNS = [
    "property_type",
    "furnish",
    "age_label",
    "city",
    "transaction_type",
    "bhk_bucket",
    "price_bucket",
    "area_bucket",
    "floor_bucket",
]


def _bin_price(row: pd.Series, gap_buy: float, gap_rent: float) -> str:
    price = row["price_capped"]
    tx = row["transaction_type"]
    if pd.isna(price):
        return "unknown"
    gap = gap_rent if tx == "Rent" else gap_buy
    if gap <= 0:
        return "unknown"
    bucket = int(price // gap)
    return f"{tx}_bucket_{bucket}"


def _bin_generic(value: float, gap: float, prefix: str) -> str:
    if pd.isna(value) or gap <= 0:
        return f"{prefix}_unknown"
    bucket = int(value // gap)
    return f"{prefix}_{bucket}"


def add_bucket_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    cf = cfg["content_features"]

    df["price_bucket"] = df.apply(
        lambda r: _bin_price(r, cf["price_bin_gap"]["buy"], cf["price_bin_gap"]["rent"]), axis=1
    )
    df["area_bucket"] = df["area_capped"].apply(lambda v: _bin_generic(v, cf["area_bin_sqft"], "area"))

    age_years_map = {
        "1-5 Year Old Property": 3,
        "5-10 Year Old Property": 7,
        "10+ Year Old Property": 15,
        "Under Construction": 0,
        "Not Specified": np.nan,
    }
    df["_age_years_est"] = df["age_label"].map(age_years_map)
    # age is already categorical/labeled in this dataset (not a continuous
    # year value), so we use the label directly as the "age bucket" -- the
    # paper's 3-year binning is not applicable since our raw AGE field is
    # itself already bucketed at the source.
    df["age_label"] = df["age_label"].fillna("Not Specified")

    df["floor_bucket"] = df["floor_num"].apply(lambda v: _bin_generic(v, cf["floor_bin"], "floor"))

    df["bhk_bucket"] = np.where(
        df["is_bhk_valid"],
        df["bhk"].apply(lambda v: f"{int(v)}BHK" if pd.notna(v) else "unknown"),
        "unknown",
    )

    for col in ["property_type", "furnish", "city", "transaction_type"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


@dataclass
class PropertyFeatureArtifacts:
    ohe: OneHotEncoder
    tfidf: TfidfVectorizer
    amenity_vocab: List[str]
    categorical_columns: List[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))
    block_sizes: Dict[str, int] = field(default_factory=dict)


def _amenities_multihot(amenity_lists: pd.Series, vocab: List[str]) -> sp.csr_matrix:
    index = {a: i for i, a in enumerate(vocab)}
    rows, cols = [], []
    for r, amenities in enumerate(amenity_lists):
        for a in amenities:
            j = index.get(a)
            if j is not None:
                rows.append(r)
                cols.append(j)
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(len(amenity_lists), len(vocab)))


def build_property_features(
    df: pd.DataFrame, cfg: dict | None = None, min_amenity_freq: int = 10
) -> Tuple[sp.csr_matrix, List[str], PropertyFeatureArtifacts]:
    """Fit encoders on `df` and return (feature_matrix, property_ids, artifacts)."""
    cfg = cfg or load_config()
    df = add_bucket_columns(df, cfg)

    # ---- categorical one-hot block -----------------------------------------
    cat_df = df[CATEGORICAL_COLUMNS].fillna("unknown").astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
    cat_block = ohe.fit_transform(cat_df)

    # ---- amenities multi-hot block -----------------------------------------
    amenity_lists = df["amenities"].fillna("").apply(lambda s: [a for a in s.split(";") if a] if isinstance(s, str) else [])
    feature_lists = df["features"].fillna("").apply(lambda s: [a for a in s.split(";") if a] if isinstance(s, str) else [])
    combined_lists = amenity_lists + feature_lists
    freq: Dict[str, int] = {}
    for lst in combined_lists:
        for a in lst:
            freq[a] = freq.get(a, 0) + 1
    vocab = sorted([a for a, c in freq.items() if c >= min_amenity_freq])
    amenity_block = _amenities_multihot(combined_lists, vocab)

    # ---- TF-IDF description block --------------------------------------------
    tfidf = TfidfVectorizer(
        max_features=cfg["content_features"]["tfidf_max_features"],
        min_df=cfg["content_features"]["tfidf_min_df"],
        stop_words="english",
    )
    text_block = tfidf.fit_transform(df["description"].fillna(""))

    # ---- combine, L2-normalize each block, stack -----------------------------
    cat_block = normalize(cat_block)
    amenity_block = normalize(amenity_block)
    text_block = normalize(text_block)

    feature_matrix = sp.hstack([cat_block, amenity_block, text_block], format="csr")
    feature_matrix = normalize(feature_matrix)

    artifacts = PropertyFeatureArtifacts(
        ohe=ohe,
        tfidf=tfidf,
        amenity_vocab=vocab,
        block_sizes={
            "categorical": cat_block.shape[1],
            "amenities": amenity_block.shape[1],
            "text": text_block.shape[1],
        },
    )
    property_ids = df["property_id"].tolist()
    return feature_matrix, property_ids, artifacts


def transform_property_features(df: pd.DataFrame, artifacts: PropertyFeatureArtifacts, cfg: dict | None = None) -> sp.csr_matrix:
    """Apply already-fitted encoders to (possibly new) property rows."""
    cfg = cfg or load_config()
    df = add_bucket_columns(df, cfg)
    cat_df = df[CATEGORICAL_COLUMNS].fillna("unknown").astype(str)
    cat_block = normalize(artifacts.ohe.transform(cat_df))

    amenity_lists = df["amenities"].fillna("").apply(lambda s: [a for a in s.split(";") if a] if isinstance(s, str) else [])
    feature_lists = df["features"].fillna("").apply(lambda s: [a for a in s.split(";") if a] if isinstance(s, str) else [])
    combined_lists = amenity_lists + feature_lists
    amenity_block = normalize(_amenities_multihot(combined_lists, artifacts.amenity_vocab))

    text_block = normalize(artifacts.tfidf.transform(df["description"].fillna("")))

    feature_matrix = sp.hstack([cat_block, amenity_block, text_block], format="csr")
    return normalize(feature_matrix)


def save_artifacts(feature_matrix: sp.csr_matrix, property_ids: List[str], artifacts: PropertyFeatureArtifacts, out_dir: Path = CONTENT_MODEL_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "feature_matrix.npz", feature_matrix)
    with open(out_dir / "property_ids.json", "w") as f:
        json.dump(property_ids, f)
    joblib.dump(artifacts, out_dir / "encoders.joblib")


def load_artifacts(in_dir: Path = CONTENT_MODEL_DIR) -> Tuple[sp.csr_matrix, List[str], PropertyFeatureArtifacts]:
    feature_matrix = sp.load_npz(in_dir / "feature_matrix.npz")
    with open(in_dir / "property_ids.json") as f:
        property_ids = json.load(f)
    artifacts = joblib.load(in_dir / "encoders.joblib")
    return feature_matrix, property_ids, artifacts
