import pandas as pd

from src.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.split import split_interactions


def test_precision_at_k_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_none_relevant():
    assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], {"a", "b", "c", "d"}, 2) == 0.5


def test_ndcg_perfect_ranking_is_one():
    assert abs(ndcg_at_k(["a", "b"], {"a", "b"}, 2) - 1.0) < 1e-9


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], {"z"}, 2) == 0.0


def test_map_at_k():
    # relevant at rank 1 and 3
    ap = average_precision_at_k(["a", "x", "b"], {"a", "b"}, 3)
    assert 0 < ap <= 1.0


def test_hit_rate():
    assert hit_rate_at_k(["a", "b"], {"b", "z"}, 2) == 1.0
    assert hit_rate_at_k(["a", "b"], {"z"}, 2) == 0.0


def test_metrics_empty_recommendation_list():
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert ndcg_at_k([], {"a"}, 5) == 0.0


def test_split_no_leakage(interactions, cfg):
    train, test = split_interactions(interactions, cfg)
    train_keys = set(zip(train["user_id"], train["property_id"], train["interaction_id"]))
    test_keys = set(zip(test["user_id"], test["property_id"], test["interaction_id"]))
    assert train_keys.isdisjoint(test_keys), "train and test must not share the same interaction rows"


def test_split_preserves_all_rows(interactions, cfg):
    train, test = split_interactions(interactions, cfg)
    assert len(train) + len(test) == len(interactions)


def test_split_every_test_user_has_train_history(interactions, cfg):
    train, test = split_interactions(interactions, cfg)
    test_users = set(test["user_id"])
    train_users = set(train["user_id"])
    assert test_users.issubset(train_users), "every user with a test interaction must also have train history"
