"""Configuration and path utilities shared across the whole project.

Every script/module loads its tunable parameters through `load_config()`
instead of hard-coding numbers, per the project's reproducibility
requirements.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = two levels up from this file (src/utils/config.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "recommendation.yaml"

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"


@functools.lru_cache(maxsize=1)
def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load (and cache) the central YAML configuration file."""
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs() -> None:
    """Create every output directory the project writes to, if missing."""
    for d in [
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        RESULTS_METRICS_DIR,
        RESULTS_PLOTS_DIR,
        RESULTS_TABLES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
