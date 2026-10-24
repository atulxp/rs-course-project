#!/usr/bin/env python3
"""Full evaluation suite (Sections 18-20, 28 of the brief).

Run:
    python scripts/evaluate.py

Trains EVALUATION-ONLY copies of the content and collaborative models on a
leave-last-N-out TRAIN split (src/evaluation/split.py) so that test
interactions never leak into training, then computes:

  1. results/tables/model_comparison.csv       - Precision/Recall/MAP/NDCG/HitRate
                                                   @5,10,20 for each of the 4 models
  2. results/tables/hybrid_alpha_sweep.csv      - hybrid performance for alpha in
                                                   the configured grid
  3. results/tables/category_performance.csv    - orchestrator-routed performance
                                                   broken down by user category
  4. results/tables/city_performance.csv        - orchestrator-routed performance
                                                   broken down by city
  5. results/tables/weighting_scheme_ablation.csv - paper Table-3-style comparison
                                                     of linear / exponential-decay /
                                                     TF-IDF interaction weighting
                                                     for the collaborative model
  6. results/tables/latency_benchmark.csv       - local inference latency per model
  7. results/metrics/summary.json               - headline numbers used by the README

IMPORTANT: every number in these files comes from actually running this
script against the SIMULATED interactions -- there are no hand-entered
results anywhere in this project (Section 39 of the brief).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_recommendations
from src.evaluation.split import split_interactions
from src.features.property_features import load_artifacts as load_content_artifacts
from src.interactions.weighting import WEIGHTING_FUNCS
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.rule_based import RuleBasedRecommender, parse_user_row
from src.users.categorize import categorize_user
from src.utils.config import DATA_PROCESSED_DIR, RESULTS_METRICS_DIR, RESULTS_TABLES_DIR, ensure_dirs, load_config


def load_data():
    properties = pd.read_csv(DATA_PROCESSED_DIR / "properties.csv")
    users = pd.read_csv(DATA_PROCESSED_DIR / "users.csv")
    interactions = pd.read_csv(DATA_PROCESSED_DIR / "interactions.csv")
    return properties, users, interactions


def build_eval_models(properties, train_interactions, cfg):
    feature_matrix, property_ids, _ = load_content_artifacts()
    content_model = ContentBasedRecommender(properties, feature_matrix, property_ids, cfg)

    collab_model = CollaborativeRecommender(cfg)
    collab_model.fit(train_interactions, weighting_scheme="linear")

    rule_model = RuleBasedRecommender(properties, train_interactions, cfg)
    hybrid_model = HybridRecommender(content_model, collab_model, cfg)
    return rule_model, content_model, collab_model, hybrid_model


def get_relevant(test_interactions: pd.DataFrame, user_id: str) -> set:
    return set(test_interactions.loc[test_interactions["user_id"] == user_id, "property_id"])


# ---------------------------------------------------------------------------
# 1. Model comparison
# ---------------------------------------------------------------------------
def run_model_comparison(properties, users, train_interactions, test_interactions, rule_model, content_model, collab_model, hybrid_model, cfg, max_users: int = 150):
    k_values = cfg["evaluation"]["k_values"]
    eligible_users = sorted(set(test_interactions["user_id"]) & set(train_interactions["user_id"]))
    if len(eligible_users) > max_users:
        rng = np.random.default_rng(cfg["seed"])
        eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))
    print(f"[evaluate] model comparison over {len(eligible_users)} users with train+test interactions")

    rows = []
    per_model_metrics = {m: [] for m in ["rule_based", "content_based", "collaborative", "hybrid"]}

    for uid in eligible_users:
        user_row = users[users["user_id"] == uid]
        if user_row.empty:
            continue
        user_row = user_row.iloc[0]
        user_pref = parse_user_row(user_row)
        relevant = get_relevant(test_interactions, uid)
        if not relevant:
            continue
        user_train_inter = train_interactions[train_interactions["user_id"] == uid]
        exclude = set(user_train_inter["property_id"])  # don't re-recommend already-seen-in-train items
        candidate_pids = properties[
            (properties["city"] == user_row["city"]) & (properties["transaction_type"] == user_row["transaction_type"])
        ]["property_id"]

        max_k = max(k_values)

        # rule-based
        rb = rule_model.recommend(user_pref, top_k=max_k + len(exclude))
        rb_ranked = [p for p in rb["property_id"].tolist() if p not in exclude][:max_k]
        per_model_metrics["rule_based"].append(evaluate_recommendations(rb_ranked, relevant, k_values))

        # content-based (uses this user's TRAIN interactions only)
        cb = content_model.recommend_for_interactions(
            user_train_inter, user_row["city"], user_row["transaction_type"], top_k=max_k, exclude_property_ids=exclude
        )
        cb_ranked = cb["property_id"].tolist() if not cb.empty else []
        per_model_metrics["content_based"].append(evaluate_recommendations(cb_ranked, relevant, k_values))

        # collaborative
        collab = collab_model.recommend(uid, top_k=max_k, filter_already_interacted=True)
        collab_ranked = collab["property_id"].tolist() if not collab.empty else []
        per_model_metrics["collaborative"].append(evaluate_recommendations(collab_ranked, relevant, k_values))

        # hybrid
        hy = hybrid_model.recommend(
            uid, user_train_inter, user_row["city"], user_row["transaction_type"], top_k=max_k, exclude_property_ids=exclude
        )
        hy_ranked = hy["property_id"].tolist() if not hy.empty else []
        per_model_metrics["hybrid"].append(evaluate_recommendations(hy_ranked, relevant, k_values))

    for model_name, metric_list in per_model_metrics.items():
        if not metric_list:
            continue
        df = pd.DataFrame(metric_list)
        agg = df.mean().to_dict()
        agg["model"] = model_name
        agg["n_users_evaluated"] = len(metric_list)
        rows.append(agg)

    result = pd.DataFrame(rows).set_index("model")
    cols = ["n_users_evaluated"] + [f"{m}@{k}" for k in k_values for m in ["precision", "recall", "map", "ndcg", "hit_rate"]]
    result = result[[c for c in cols if c in result.columns]]
    return result


# ---------------------------------------------------------------------------
# 2. Hybrid alpha sweep
# ---------------------------------------------------------------------------
def run_alpha_sweep(properties, users, train_interactions, test_interactions, hybrid_model, cfg, max_users: int = 80):
    k = cfg["evaluation"]["paper_k"]
    alphas = cfg["hybrid"]["alpha_grid"]
    eligible_users = sorted(set(test_interactions["user_id"]) & set(train_interactions["user_id"]))
    if len(eligible_users) > max_users:
        rng = np.random.default_rng(cfg["seed"])
        eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))

    rows = []
    for alpha in alphas:
        metrics = []
        for uid in eligible_users:
            user_row = users[users["user_id"] == uid]
            if user_row.empty:
                continue
            user_row = user_row.iloc[0]
            relevant = get_relevant(test_interactions, uid)
            if not relevant:
                continue
            user_train_inter = train_interactions[train_interactions["user_id"] == uid]
            exclude = set(user_train_inter["property_id"])
            hy = hybrid_model.recommend(
                uid, user_train_inter, user_row["city"], user_row["transaction_type"],
                alpha=alpha, top_k=k, exclude_property_ids=exclude,
            )
            ranked = hy["property_id"].tolist() if not hy.empty else []
            metrics.append(evaluate_recommendations(ranked, relevant, [k]))
        if metrics:
            agg = pd.DataFrame(metrics).mean().to_dict()
            agg["alpha"] = alpha
            rows.append(agg)
    return pd.DataFrame(rows).set_index("alpha")


# ---------------------------------------------------------------------------
# 3. Category-wise & 4. city-wise performance (orchestrator routing)
# ---------------------------------------------------------------------------
def run_orchestrator_breakdown(properties, users, train_interactions, test_interactions, rule_model, content_model, collab_model, hybrid_model, cfg, max_users: int = 200):
    k = cfg["evaluation"]["paper_k"]
    eligible_users = sorted(set(test_interactions["user_id"]) & set(train_interactions["user_id"]))
    if len(eligible_users) > max_users:
        rng = np.random.default_rng(cfg["seed"])
        eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))

    records = []
    for uid in eligible_users:
        user_row = users[users["user_id"] == uid]
        if user_row.empty:
            continue
        user_row = user_row.iloc[0]
        relevant = get_relevant(test_interactions, uid)
        if not relevant:
            continue
        category = categorize_user(uid, train_interactions, cfg)
        user_train_inter = train_interactions[train_interactions["user_id"] == uid]
        exclude = set(user_train_inter["property_id"])
        user_pref = parse_user_row(user_row)

        if category == "cold_start":
            df = rule_model.recommend(user_pref, top_k=k)
        elif category == "short_term":
            df = content_model.recommend_for_interactions(
                user_train_inter, user_row["city"], user_row["transaction_type"], top_k=k, only_recent=True, exclude_property_ids=exclude
            )
        else:
            df = hybrid_model.recommend(
                uid, user_train_inter, user_row["city"], user_row["transaction_type"], top_k=k, exclude_property_ids=exclude
            )
        ranked = df["property_id"].tolist() if (df is not None and not df.empty) else []
        m = evaluate_recommendations(ranked, relevant, [k])
        m["user_id"] = uid
        m["category"] = category
        m["city"] = user_row["city"]
        records.append(m)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Weighting scheme ablation (paper Table 3 structure)
# ---------------------------------------------------------------------------
def run_weighting_ablation(properties, users, train_interactions, test_interactions, cfg, max_users: int = 150):
    k = cfg["evaluation"]["paper_k"]
    rows = []
    rng = np.random.default_rng(cfg["seed"])
    for scheme in WEIGHTING_FUNCS.keys():
        collab = CollaborativeRecommender(cfg)
        collab.fit(train_interactions, weighting_scheme=scheme)
        eligible_users = sorted(set(test_interactions["user_id"]) & set(collab.user_to_idx.keys()))
        if len(eligible_users) > max_users:
            eligible_users = list(rng.choice(eligible_users, size=max_users, replace=False))
        metrics = []
        for uid in eligible_users:
            relevant = get_relevant(test_interactions, uid)
            if not relevant:
                continue
            rec = collab.recommend(uid, top_k=k, filter_already_interacted=True)
            ranked = rec["property_id"].tolist() if not rec.empty else []
            metrics.append(evaluate_recommendations(ranked, relevant, [k]))
        if metrics:
            agg = pd.DataFrame(metrics).mean().to_dict()
            agg["weighting_scheme"] = scheme
            agg["n_users"] = len(metrics)
            rows.append(agg)
    return pd.DataFrame(rows).set_index("weighting_scheme")


# ---------------------------------------------------------------------------
# 6. Latency benchmark (Section 28 -- local benchmark, not the paper's <40ms claim)
# ---------------------------------------------------------------------------
def run_latency_benchmark(properties, users, interactions, rule_model, content_model, collab_model, hybrid_model, cfg, n_samples=100):
    rows = []
    rng = np.random.default_rng(cfg["seed"])
    sample_users = users.sample(n=min(n_samples, len(users)), random_state=cfg["seed"])

    def _time_call(fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            fn(*args, **kwargs)
        except Exception:
            return None
        return (time.perf_counter() - t0) * 1000  # ms

    for _, user_row in sample_users.iterrows():
        uid = user_row["user_id"]
        user_pref = parse_user_row(user_row)
        user_inter = interactions[interactions["user_id"] == uid]

        t = _time_call(rule_model.recommend, user_pref, top_k=10)
        if t is not None:
            rows.append({"model": "rule_based", "latency_ms": t})

        if not user_inter.empty:
            t = _time_call(
                content_model.recommend_for_interactions, user_inter, user_row["city"], user_row["transaction_type"], top_k=10
            )
            if t is not None:
                rows.append({"model": "content_based", "latency_ms": t})

            t = _time_call(collab_model.recommend, uid, top_k=10)
            if t is not None:
                rows.append({"model": "collaborative", "latency_ms": t})

            t = _time_call(
                hybrid_model.recommend, uid, user_inter, user_row["city"], user_row["transaction_type"], top_k=10
            )
            if t is not None:
                rows.append({"model": "hybrid", "latency_ms": t})

    df = pd.DataFrame(rows)
    summary = df.groupby("model")["latency_ms"].agg(["mean", "median", lambda s: s.quantile(0.95), "count"])
    summary.columns = ["mean_ms", "median_ms", "p95_ms", "n_calls"]
    return summary


# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    cfg = load_config()
    properties, users, interactions = load_data()
    train_interactions, test_interactions = split_interactions(interactions, cfg)
    print(f"[evaluate] split: {len(train_interactions)} train / {len(test_interactions)} test interactions "
          f"({test_interactions['user_id'].nunique()} users have a held-out test set)")

    rule_model, content_model, collab_model, hybrid_model = build_eval_models(properties, train_interactions, cfg)

    print("\n[evaluate] 1/6 model comparison ...")
    model_comparison = run_model_comparison(
        properties, users, train_interactions, test_interactions, rule_model, content_model, collab_model, hybrid_model, cfg
    )
    model_comparison.to_csv(RESULTS_TABLES_DIR / "model_comparison.csv")
    print(model_comparison.round(4))

    print("\n[evaluate] 2/6 hybrid alpha sweep ...")
    alpha_sweep = run_alpha_sweep(properties, users, train_interactions, test_interactions, hybrid_model, cfg)
    alpha_sweep.to_csv(RESULTS_TABLES_DIR / "hybrid_alpha_sweep.csv")
    print(alpha_sweep.round(4))

    print("\n[evaluate] 3/6 orchestrator breakdown (category + city) ...")
    breakdown = run_orchestrator_breakdown(
        properties, users, train_interactions, test_interactions, rule_model, content_model, collab_model, hybrid_model, cfg
    )
    k = cfg["evaluation"]["paper_k"]
    metric_cols = [c for c in breakdown.columns if "@" in c]
    category_perf = breakdown.groupby("category")[metric_cols].mean()
    category_perf["n_users"] = breakdown.groupby("category").size()
    category_perf.to_csv(RESULTS_TABLES_DIR / "category_performance.csv")
    print(category_perf.round(4))

    city_perf = breakdown.groupby("city")[metric_cols].mean()
    city_perf["n_users"] = breakdown.groupby("city").size()
    city_perf.to_csv(RESULTS_TABLES_DIR / "city_performance.csv")
    print(city_perf.round(4))

    print("\n[evaluate] 4/6 weighting scheme ablation (paper Table 3 structure) ...")
    weighting_ablation = run_weighting_ablation(properties, users, train_interactions, test_interactions, cfg)
    weighting_ablation.to_csv(RESULTS_TABLES_DIR / "weighting_scheme_ablation.csv")
    print(weighting_ablation.round(4))

    print("\n[evaluate] 5/6 latency benchmark (local, student-laptop scale) ...")
    latency = run_latency_benchmark(properties, users, interactions, rule_model, content_model, collab_model, hybrid_model, cfg)
    latency.to_csv(RESULTS_TABLES_DIR / "latency_benchmark.csv")
    print(latency.round(3))

    print("\n[evaluate] 6/6 writing summary.json ...")
    best_alpha = alpha_sweep[f"ndcg@{k}"].idxmax() if f"ndcg@{k}" in alpha_sweep.columns and not alpha_sweep.empty else None
    summary = {
        "n_properties": len(properties),
        "n_users": len(users),
        "n_interactions": len(interactions),
        "n_train_interactions": len(train_interactions),
        "n_test_interactions": len(test_interactions),
        "model_comparison_at_k10": json.loads(model_comparison.round(4).to_json(orient="index")),
        "best_hybrid_alpha_by_ndcg": None if best_alpha is None else float(best_alpha),
        "category_performance": json.loads(category_perf.round(4).to_json(orient="index")),
        "city_performance": json.loads(city_perf.round(4).to_json(orient="index")),
        "weighting_scheme_ablation": json.loads(weighting_ablation.round(4).to_json(orient="index")),
        "latency_ms": json.loads(latency.round(3).to_json(orient="index")),
    }
    with open(RESULTS_METRICS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[evaluate] wrote {RESULTS_METRICS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
