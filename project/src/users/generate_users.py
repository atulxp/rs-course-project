"""Synthetic user generation (Section 8 of the brief).

IMPORTANT (academic integrity, see README): the dataset contains public
PROPERTY listings only -- no real user accounts, sessions, or click logs.
Every user profile produced by this module is a SIMULATED persona. To keep
the personas plausible (not "meaningless random noise", per the brief), each
user's preferences (city, transaction type, BHK, budget, area, furnishing,
locality) are SAMPLED FROM THE ACTUAL EMPIRICAL DISTRIBUTION of the
processed property dataset for the chosen (city, transaction_type) slice --
i.e. a synthetic "2 BHK Mumbai buyer" gets a budget range sampled from real
2BHK-Mumbai-Buy prices, not an invented number.

Run as:
    python -m src.users.generate_users
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR, load_config

RNG_SEED_DEFAULT = 42


def _sample_budget_range(rng: np.random.Generator, prices: pd.Series) -> tuple[float, float]:
    if len(prices) == 0:
        return (0.0, 0.0)
    center = rng.choice(prices.values)
    # Budget window: +/- 20% around a real observed price for this slice.
    lo = center * 0.8
    hi = center * 1.2
    return float(lo), float(hi)


def _sample_area_range(rng: np.random.Generator, areas: pd.Series) -> tuple[float, float]:
    if len(areas) == 0:
        return (0.0, 0.0)
    center = rng.choice(areas.values)
    return float(center * 0.75), float(center * 1.25)


def generate_users(properties: pd.DataFrame, cfg: dict | None = None, seed: int | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = seed if seed is not None else cfg["seed"]
    rng = np.random.default_rng(seed)

    n_users = cfg["synthetic"]["num_users"]
    split = cfg["synthetic"]["archetype_split"]

    props = properties[properties["is_price_valid"] & properties["is_area_valid"]].copy()
    props = props[props["transaction_type"].isin(["Buy", "Rent"])]

    cities = props["city"].unique().tolist()
    furnish_options = props["furnish"].unique().tolist()

    users = []
    archetypes = rng.choice(
        ["buyer", "renter"], size=n_users, p=[split["buyer"], split["renter"]]
    )

    for i in range(n_users):
        archetype = archetypes[i]
        tx = "Buy" if archetype == "buyer" else "Rent"
        city = rng.choice(cities)

        slice_df = props[(props["city"] == city) & (props["transaction_type"] == tx)]
        if len(slice_df) < 5:
            # fall back to city-only slice if the (city, tx) combo is too thin
            slice_df = props[props["city"] == city]
        if len(slice_df) == 0:
            slice_df = props

        bhk_pool = slice_df.loc[slice_df["is_bhk_valid"], "bhk"]
        preferred_bhk = int(rng.choice(bhk_pool.values)) if len(bhk_pool) else rng.integers(1, 4)

        bhk_slice = slice_df[slice_df["bhk"] == preferred_bhk]
        price_pool = bhk_slice["price"] if len(bhk_slice) >= 5 else slice_df["price"]
        area_pool = bhk_slice["area_sqft"] if len(bhk_slice) >= 5 else slice_df["area_sqft"]

        budget_min, budget_max = _sample_budget_range(rng, price_pool.dropna())
        area_min, area_max = _sample_area_range(rng, area_pool.dropna())

        locality_pool = slice_df["locality"].dropna()
        n_localities = min(3, locality_pool.nunique()) or 1
        preferred_localities = (
            list(pd.Series(locality_pool.unique()).sample(n=n_localities, random_state=int(rng.integers(0, 1_000_000))))
            if len(locality_pool) > 0
            else []
        )

        property_type_pool = slice_df["property_type"]
        preferred_property_type = (
            rng.choice(property_type_pool.values) if len(property_type_pool) else "Residential Apartment"
        )

        preferred_furnish = rng.choice(furnish_options)

        amenity_pool = []
        for lst in slice_df["amenities"].dropna().head(200):
            amenity_pool.extend([a for a in str(lst).split(";") if a and not a.startswith("Other:")])
        amenity_pool = list(dict.fromkeys(amenity_pool))  # de-dup, keep order
        k = min(3, len(amenity_pool))
        preferred_amenities = list(rng.choice(amenity_pool, size=k, replace=False)) if k > 0 else []

        users.append(
            {
                "user_id": f"user_{i:05d}",
                "archetype": archetype,
                "city": city,
                "transaction_type": tx,
                "preferred_bhk": preferred_bhk,
                "preferred_property_type": preferred_property_type,
                "budget_min": round(budget_min, 2),
                "budget_max": round(budget_max, 2),
                "preferred_area_min": round(area_min, 2),
                "preferred_area_max": round(area_max, 2),
                "preferred_furnish": preferred_furnish,
                "preferred_localities": json.dumps(preferred_localities),
                "preferred_amenities": json.dumps(preferred_amenities),
            }
        )

    return pd.DataFrame(users)


def run(out_dir: Path = DATA_PROCESSED_DIR) -> pd.DataFrame:
    props = pd.read_csv(out_dir / "properties.csv")
    cfg = load_config()
    users = generate_users(props, cfg)
    out_path = out_dir / "users.csv"
    users.to_csv(out_path, index=False)
    print(f"[generate_users] wrote {len(users)} synthetic users -> {out_path}")
    print(users["archetype"].value_counts())
    print(users["city"].value_counts())
    return users


if __name__ == "__main__":
    run()
