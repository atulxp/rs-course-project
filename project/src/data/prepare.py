"""Data preprocessing pipeline.

Run as:
    python -m src.data.prepare

Reads every raw city CSV under data/raw/, harmonizes the (different!)
per-city schemas into one canonical property table, cleans/parses every
field, and writes:

    data/processed/properties.csv        (canonical, one row per property)
    data/processed/properties_raw_display.csv  (original display-only fields kept)
    docs/data_dictionary.md              (auto-generated data dictionary, see
                                           src/data/data_dictionary.py)

Design notes / adaptations (see also docs/data_dictionary.md and README):
  * The four raw files (mumbai.csv, gurgaon_10k.csv, hyderabad.csv,
    kolkata.csv) have four DIFFERENT schemas (35-67 columns, different
    column names for the same concept e.g. no `LOCALITY` column in
    mumbai/hyderabad/kolkata -- locality is nested inside the `location`
    python-dict column instead). This module harmonizes them into one
    canonical schema rather than assuming a single shared schema.
  * PRICE and AREA are free-text strings ("1.1 Cr", "69.3 - 83.7 L",
    "518-623  sq.ft.") and are parsed with src/data/parsing.py.
  * TRANSACT_TYPE is numeric-coded (1.0=Buy, 2.0=Rent) -- confirmed by
    cross-checking the price magnitude distribution per code during EDA.
  * We do NOT silently drop properties with missing BHK/price/area; instead
    we keep the row and set an explicit `is_*_valid` flag, so that models
    (and the report's missing-value analysis) can make an informed choice
    about what to do with them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.facets import (
    AGE_CODE_MAP,
    FURNISH_CODE_MAP,
    LISTING_STATUS_MAP,
    load_all_facets,
    decode_id_list,
)
from src.data.parsing import (
    parse_area,
    parse_floor_num,
    parse_id_list,
    parse_price,
    parse_pydict,
)
from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, ensure_dirs, load_config

CITY_FILES = {
    "mumbai": "mumbai.csv",
    "gurgaon": "gurgaon_10k.csv",
    "hyderabad": "hyderabad.csv",
    "kolkata": "kolkata.csv",
}

# Canonical columns produced by this pipeline.
CANONICAL_COLUMNS = [
    "property_id",
    "source_city",
    "city",
    "locality",
    "transaction_type",
    "listing_status",
    "property_type",
    "bhk",
    "bathroom_num",
    "balcony_num",
    "furnish",
    "age_label",
    "floor_num",
    "total_floor",
    "price_min",
    "price_max",
    "price",
    "is_price_valid",
    "area_min",
    "area_max",
    "area_sqft",
    "is_area_valid",
    "price_per_sqft",
    "latitude",
    "longitude",
    "amenities",
    "features",
    "num_amenities",
    "description",
    "building_name",
    "society_name",
    "prop_heading",
]


def _get_location_field(loc: dict, *keys, default=None):
    for k in keys:
        if k in loc and loc[k] not in (None, ""):
            return loc[k]
    return default


def load_and_harmonize_city(city_key: str, path: Path, facets: dict) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, on_bad_lines="warn")
    n = len(df)

    out = pd.DataFrame(index=df.index)
    out["source_city"] = city_key

    # ---- property id (globally unique across the 4 files) --------------------
    prop_id_col = "PROP_ID" if "PROP_ID" in df.columns else df.columns[0]
    out["property_id"] = city_key + "_" + df[prop_id_col].astype(str)

    # ---- location dict (city sub-region + locality + lat/lon) ----------------
    loc_dicts = df["location"].apply(parse_pydict) if "location" in df.columns else pd.Series([{}] * n)
    out["city"] = city_key.capitalize()
    out["locality"] = loc_dicts.apply(
        lambda d: _get_location_field(d, "LOCALITY_NAME", default=None)
    )
    if "LOCALITY" in df.columns:
        out["locality"] = out["locality"].fillna(df["LOCALITY"])
    out["locality"] = out["locality"].astype(str).str.strip().replace({"": np.nan, "None": np.nan})

    map_dicts = df["MAP_DETAILS"].apply(parse_pydict) if "MAP_DETAILS" in df.columns else pd.Series([{}] * n)
    out["latitude"] = pd.to_numeric(map_dicts.apply(lambda d: d.get("LATITUDE")), errors="coerce")
    out["longitude"] = pd.to_numeric(map_dicts.apply(lambda d: d.get("LONGITUDE")), errors="coerce")

    # ---- transaction type / listing status --------------------------------------
    # See src/data/facets.py LISTING_STATUS_MAP docstring: the raw TRANSACT_TYPE
    # code distinguishes Resale vs New-Booking *within Buy listings*, it is NOT
    # a Buy/Rent flag. All rows with a present TRANSACT_TYPE code are Buy;
    # rows with a missing code are later split into Buy/Rent by
    # impute_transaction_type() based on price magnitude.
    tx_raw = df["TRANSACT_TYPE"].astype(str) if "TRANSACT_TYPE" in df.columns else pd.Series([""] * n)
    out["listing_status"] = tx_raw.map(LISTING_STATUS_MAP).fillna("Not Specified")
    out["transaction_type"] = np.where(tx_raw.map(LISTING_STATUS_MAP).notna(), "Buy", "Unknown")

    # ---- property type -----------------------------------------------------
    out["property_type"] = df.get("PROPERTY_TYPE", pd.Series([np.nan] * n)).astype(str).replace(
        {"nan": np.nan}
    )

    # ---- BHK / bathroom / balcony -----------------------------------------
    out["bhk"] = pd.to_numeric(df.get("BEDROOM_NUM"), errors="coerce")
    out["bathroom_num"] = pd.to_numeric(df.get("BATHROOM_NUM"), errors="coerce") if "BATHROOM_NUM" in df.columns else np.nan
    out["balcony_num"] = pd.to_numeric(df.get("BALCONY_NUM"), errors="coerce") if "BALCONY_NUM" in df.columns else np.nan

    # ---- furnish / age (raw integer codes, decoded via facets.py map) ------
    furnish_raw = df.get("FURNISH", pd.Series([np.nan] * n)).astype(str).str.strip()
    out["furnish"] = furnish_raw.map(FURNISH_CODE_MAP).fillna("Not Specified")

    age_raw = df.get("AGE", pd.Series([np.nan] * n)).astype(str).str.strip()
    out["age_label"] = age_raw.map(AGE_CODE_MAP).fillna("Not Specified")

    # ---- floor / total floor ------------------------------------------------
    out["floor_num"] = df.get("FLOOR_NUM", pd.Series([np.nan] * n)).apply(parse_floor_num)
    out["total_floor"] = pd.to_numeric(df.get("TOTAL_FLOOR"), errors="coerce")

    # ---- price ---------------------------------------------------------------
    price_min, price_max = zip(*df.get("PRICE", pd.Series([np.nan] * n)).apply(parse_price))
    out["price_min"] = pd.to_numeric(pd.Series(price_min), errors="coerce")
    out["price_max"] = pd.to_numeric(pd.Series(price_max), errors="coerce")
    out["price"] = out[["price_min", "price_max"]].mean(axis=1)
    out["is_price_valid"] = out["price"].notna() & (out["price"] > 0)

    # ---- area ------------------------------------------------------------------
    area_min, area_max = zip(*df.get("AREA", pd.Series([np.nan] * n)).apply(parse_area))
    out["area_min"] = pd.to_numeric(pd.Series(area_min), errors="coerce")
    out["area_max"] = pd.to_numeric(pd.Series(area_max), errors="coerce")
    out["area_sqft"] = out[["area_min", "area_max"]].mean(axis=1)
    out["is_area_valid"] = out["area_sqft"].notna() & (out["area_sqft"] > 50) & (out["area_sqft"] < 50000)

    out["price_per_sqft"] = np.where(
        out["is_price_valid"] & out["is_area_valid"] & (out["area_sqft"] > 0),
        out["price"] / out["area_sqft"],
        np.nan,
    )

    # ---- amenities / features (id lists, decoded via facets where possible) --
    feat_facet = facets.get("FEATURES", {})
    amen_facet = facets.get("AMENITIES", {})
    features_ids = df.get("FEATURES", pd.Series([np.nan] * n)).apply(parse_id_list)
    amenities_ids = df.get("AMENITIES", pd.Series([np.nan] * n)).apply(parse_id_list) if "AMENITIES" in df.columns else pd.Series([[]] * n)

    out["features"] = features_ids.apply(lambda ids: decode_id_list(ids, feat_facet))
    out["amenities"] = amenities_ids.apply(lambda ids: decode_id_list(ids, amen_facet))
    out["num_amenities"] = out["amenities"].apply(len) + out["features"].apply(len)

    # ---- text / display fields -------------------------------------------------
    out["description"] = df.get("DESCRIPTION", pd.Series([""] * n)).astype(str).replace("nan", "")
    out["building_name"] = df.get("BUILDING_NAME", df.get("PROP_NAME", pd.Series([np.nan] * n)))
    out["society_name"] = df.get("SOCIETY_NAME", pd.Series([np.nan] * n))
    out["prop_heading"] = df.get("PROP_HEADING", pd.Series([np.nan] * n))

    return out[CANONICAL_COLUMNS]


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap (not delete) extreme price/area/price-per-sqft outliers per
    (city, transaction_type) group using the 1st/99th percentile, and flag
    BHK values above a sane residential ceiling as invalid rather than
    silently coercing them.
    """
    df = df.copy()
    df["price_capped"] = df["price"]
    df["area_capped"] = df["area_sqft"]

    for (city, tx), grp in df.groupby(["city", "transaction_type"]):
        idx = grp.index
        for col, cap_col, valid_col in [
            ("price", "price_capped", "is_price_valid"),
            ("area_sqft", "area_capped", "is_area_valid"),
        ]:
            vals = grp.loc[grp[valid_col], col]
            if len(vals) < 20:
                continue
            lo, hi = vals.quantile([0.01, 0.99])
            df.loc[idx, cap_col] = grp[col].clip(lower=lo, upper=hi)

    # BHK sanity: residential BHK of 1-10 is plausible; anything above is a
    # data-entry artifact (we saw values up to 60 during EDA) -> flag invalid,
    # keep the raw value for transparency.
    df["is_bhk_valid"] = df["bhk"].between(1, 10)

    df["price_per_sqft_capped"] = np.where(
        df["is_price_valid"] & df["is_area_valid"] & (df["area_capped"] > 0),
        df["price_capped"] / df["area_capped"],
        np.nan,
    )
    return df


TRANSACTION_TYPE_IMPUTE_THRESHOLD = 2_000_000  # INR


def impute_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the ~17% of rows whose raw TRANSACT_TYPE code was missing.

    EDA finding (docs/data_dictionary.md): rows with a missing TRANSACT_TYPE
    code have a price distribution (median ~INR 28k, 90th pct ~INR 100k)
    that matches typical *monthly rent* figures, not purchase prices, while
    confirmed "Buy" rows rarely fall below the ~INR 2,000,000 mark (5th
    percentile ~INR 2.14M). We therefore impute missing transaction types
    from price magnitude using a documented threshold, and keep an explicit
    `transaction_type_imputed` flag so this adaptation is fully transparent
    and reversible.
    """
    df = df.copy()
    df["transaction_type_imputed"] = False
    mask = (df["transaction_type"] == "Unknown") & df["is_price_valid"]
    is_rent = mask & (df["price"] < TRANSACTION_TYPE_IMPUTE_THRESHOLD)
    is_buy = mask & (df["price"] >= TRANSACTION_TYPE_IMPUTE_THRESHOLD)
    df.loc[is_rent, "transaction_type"] = "Rent"
    df.loc[is_buy, "transaction_type"] = "Buy"
    df.loc[is_rent | is_buy, "transaction_type_imputed"] = True
    # any still-Unknown (no valid price either) rows are left as "Unknown"
    # and excluded from transaction-type-specific modeling downstream.
    n_imputed = int((is_rent | is_buy).sum())
    print(f"[prepare] imputed transaction_type for {n_imputed} rows via price-threshold heuristic "
          f"(INR {TRANSACTION_TYPE_IMPUTE_THRESHOLD:,})")
    return df


def run(raw_dir: Path = DATA_RAW_DIR, out_dir: Path = DATA_PROCESSED_DIR) -> pd.DataFrame:
    ensure_dirs()
    facets = load_all_facets()

    frames = []
    for city_key, fname in CITY_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            print(f"[prepare] WARNING: {path} not found, skipping {city_key}", file=sys.stderr)
            continue
        print(f"[prepare] loading {city_key} from {path} ...")
        frame = load_and_harmonize_city(city_key, path, facets)
        print(f"[prepare]   -> {len(frame)} rows")
        frames.append(frame)

    all_props = pd.concat(frames, ignore_index=True)
    before = len(all_props)
    all_props = all_props.drop_duplicates(subset=["property_id"]).reset_index(drop=True)
    dupes_removed = before - len(all_props)
    print(f"[prepare] concatenated {before} rows, removed {dupes_removed} duplicate property_ids -> {len(all_props)}")

    all_props = clean_outliers(all_props)
    all_props = impute_transaction_type(all_props)

    # drop rows that are unusable for every downstream model: no price AND no
    # area AND no bhk -- these carry essentially no signal.
    unusable = (~all_props["is_price_valid"]) & (~all_props["is_area_valid"]) & (all_props["bhk"].isna())
    print(f"[prepare] dropping {unusable.sum()} rows with no price, area, or BHK signal at all")
    all_props = all_props[~unusable].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "properties.csv"
    # amenities/features are python lists -> store as ';'-joined strings for CSV
    export = all_props.copy()
    export["amenities"] = export["amenities"].apply(lambda l: ";".join(l))
    export["features"] = export["features"].apply(lambda l: ";".join(l))
    export.to_csv(out_path, index=False)
    print(f"[prepare] wrote {len(export)} rows x {export.shape[1]} cols -> {out_path}")

    return all_props


if __name__ == "__main__":
    run()
