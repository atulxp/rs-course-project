#!/usr/bin/env python3
"""Generates every plot required by Section 30 of the brief, from the
ACTUAL processed data / evaluation output on disk (no invented numbers).

Run:
    python scripts/generate_plots.py
(after scripts/train.py and scripts/evaluate.py have been run)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR, RESULTS_METRICS_DIR, RESULTS_PLOTS_DIR, RESULTS_TABLES_DIR, ensure_dirs

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def savefig(fig, name):
    path = RESULTS_PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ensure_dirs()
    properties = pd.read_csv(DATA_PROCESSED_DIR / "properties.csv")
    users = pd.read_csv(DATA_PROCESSED_DIR / "users.csv")
    interactions = pd.read_csv(DATA_PROCESSED_DIR / "interactions.csv")

    print("Dataset plots:")
    # 1. city distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    properties["city"].value_counts().plot(kind="bar", ax=ax, color="#3B6EA5")
    ax.set_title("Property Count by City")
    ax.set_ylabel("# properties")
    savefig(fig, "01_city_distribution.png")

    # 2. property type distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    properties["property_type"].value_counts().plot(kind="barh", ax=ax, color="#5A9367")
    ax.set_title("Property Type Distribution")
    savefig(fig, "02_property_type_distribution.png")

    # 3. price distribution (log scale, buy vs rent separately since scales differ)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    buy_prices = properties.loc[(properties.transaction_type == "Buy") & properties.is_price_valid, "price"]
    rent_prices = properties.loc[(properties.transaction_type == "Rent") & properties.is_price_valid, "price"]
    axes[0].hist(buy_prices, bins=60, color="#D98C3C")
    axes[0].set_xscale("log")
    axes[0].set_title("Buy Price Distribution (INR, log scale)")
    axes[1].hist(rent_prices, bins=60, color="#C0504D")
    axes[1].set_xscale("log")
    axes[1].set_title("Rent Price Distribution (INR/month, log scale)")
    savefig(fig, "03_price_distribution.png")

    # 4. BHK distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    properties.loc[properties.is_bhk_valid, "bhk"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#8064A2")
    ax.set_title("BHK Distribution")
    savefig(fig, "04_bhk_distribution.png")

    # 5. missing value analysis
    key_cols = ["price", "area_sqft", "bhk", "locality", "furnish", "floor_num", "latitude"]
    miss = (properties[key_cols].isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    miss.plot(kind="barh", ax=ax, color="#C0504D")
    ax.set_xlabel("% missing")
    ax.set_title("Missing-Value Analysis (key columns)")
    savefig(fig, "05_missing_value_analysis.png")

    # 6. (simulated) user interaction distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    interactions["interaction_type"].value_counts().plot(kind="bar", ax=ax, color="#4BACC6")
    ax.set_title("Simulated Interaction-Type Distribution\n(SYNTHETIC data)")
    savefig(fig, "06_interaction_distribution.png")

    print("\nEvaluation plots:")
    tables_dir = RESULTS_TABLES_DIR
    if (tables_dir / "model_comparison.csv").exists():
        mc = pd.read_csv(tables_dir / "model_comparison.csv", index_col=0)

        # 7. model comparison bar chart (precision/recall/map/ndcg @10)
        cols = [c for c in ["precision@10", "recall@10", "map@10", "ndcg@10"] if c in mc.columns]
        if cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            mc[cols].plot(kind="bar", ax=ax)
            ax.set_title("Model Comparison @K=10 (simulated evaluation environment)")
            ax.set_ylabel("score")
            savefig(fig, "07_model_comparison.png")

        # 8-11: precision/recall/map/ndcg vs K curves
        k_values = [5, 10, 20]
        for metric, fname in [("precision", "08_precision_at_k.png"), ("recall", "09_recall_at_k.png"),
                                ("map", "10_map_at_k.png"), ("ndcg", "11_ndcg_at_k.png")]:
            fig, ax = plt.subplots(figsize=(7, 4))
            for model in mc.index:
                ys = [mc.loc[model, f"{metric}@{k}"] for k in k_values if f"{metric}@{k}" in mc.columns]
                if ys:
                    ax.plot(k_values[: len(ys)], ys, marker="o", label=model)
            ax.set_xlabel("K")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()}@K by Model")
            ax.legend()
            savefig(fig, fname)

    if (tables_dir / "hybrid_alpha_sweep.csv").exists():
        alpha_df = pd.read_csv(tables_dir / "hybrid_alpha_sweep.csv", index_col=0)
        metric_col = [c for c in alpha_df.columns if c.startswith("ndcg")]
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in [c for c in alpha_df.columns]:
            ax.plot(alpha_df.index, alpha_df[c], marker="o", label=c)
        ax.set_xlabel("alpha (content weight)")
        ax.set_title("Hybrid Alpha Sweep")
        ax.legend(fontsize=7)
        savefig(fig, "12_hybrid_alpha_sweep.png")

    if (tables_dir / "city_performance.csv").exists():
        city_df = pd.read_csv(tables_dir / "city_performance.csv", index_col=0)
        metric_cols = [c for c in city_df.columns if "@" in c]
        fig, ax = plt.subplots(figsize=(7, 4))
        city_df[metric_cols].plot(kind="bar", ax=ax)
        ax.set_title("City-wise Orchestrator Performance")
        savefig(fig, "13_city_wise_performance.png")

    if (tables_dir / "category_performance.csv").exists():
        cat_df = pd.read_csv(tables_dir / "category_performance.csv", index_col=0)
        metric_cols = [c for c in cat_df.columns if "@" in c]
        fig, ax = plt.subplots(figsize=(7, 4))
        cat_df[metric_cols].plot(kind="bar", ax=ax)
        ax.set_title("User-Category-wise Orchestrator Performance")
        savefig(fig, "14_user_type_performance.png")

    if (tables_dir / "latency_benchmark.csv").exists():
        lat_df = pd.read_csv(tables_dir / "latency_benchmark.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        lat_df[["mean_ms", "median_ms", "p95_ms"]].plot(kind="bar", ax=ax)
        ax.set_ylabel("ms")
        ax.set_title("Recommendation Latency by Model (local benchmark)")
        savefig(fig, "15_latency_benchmark.png")

    print("\nAll plots written to results/plots/")


if __name__ == "__main__":
    main()
