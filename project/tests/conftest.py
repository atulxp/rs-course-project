"""Shared pytest fixtures. Assumes `python scripts/train.py` has already been
run at least once (so data/processed/*.csv and models/*/* exist) -- exactly
like the README instructs users to do before running tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.utils.config import DATA_PROCESSED_DIR, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def properties():
    path = DATA_PROCESSED_DIR / "properties.csv"
    if not path.exists():
        pytest.skip("data/processed/properties.csv missing -- run `python scripts/train.py` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def users():
    path = DATA_PROCESSED_DIR / "users.csv"
    if not path.exists():
        pytest.skip("data/processed/users.csv missing -- run `python scripts/train.py` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def interactions():
    path = DATA_PROCESSED_DIR / "interactions.csv"
    if not path.exists():
        pytest.skip("data/processed/interactions.csv missing -- run `python scripts/train.py` first")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def orchestrator():
    from src.recommendation.pipeline import build_orchestrator

    try:
        return build_orchestrator()
    except FileNotFoundError:
        pytest.skip("Model artifacts missing -- run `python scripts/train.py` first")
