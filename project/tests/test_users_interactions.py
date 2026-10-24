import pandas as pd

from src.interactions.generate_interactions import generate_interactions
from src.users.generate_users import generate_users


def test_generate_users_basic(properties, cfg):
    small_cfg = dict(cfg)
    small_cfg["synthetic"] = dict(cfg["synthetic"])
    small_cfg["synthetic"]["num_users"] = 50
    users = generate_users(properties, small_cfg, seed=1)
    assert len(users) == 50
    assert users["user_id"].is_unique
    assert set(users["transaction_type"].unique()) <= {"Buy", "Rent"}


def test_generate_users_budget_within_plausible_range(properties, cfg):
    small_cfg = dict(cfg)
    small_cfg["synthetic"] = dict(cfg["synthetic"])
    small_cfg["synthetic"]["num_users"] = 50
    users = generate_users(properties, small_cfg, seed=2)
    assert (users["budget_max"] >= users["budget_min"]).all()


def test_generate_interactions_basic(properties, cfg):
    small_cfg = dict(cfg)
    small_cfg["synthetic"] = dict(cfg["synthetic"])
    small_cfg["synthetic"]["num_users"] = 30
    users = __import__("src.users.generate_users", fromlist=["generate_users"]).generate_users(properties, small_cfg, seed=3)
    interactions = generate_interactions(users, properties, small_cfg, seed=3)
    if len(interactions):
        assert set(interactions["user_id"]) <= set(users["user_id"])
        assert set(interactions["property_id"]) <= set(properties["property_id"])
        assert interactions["match_score"].between(0, 1).all()


def test_no_interactions_for_true_cold_start_users(interactions):
    """Users generated with target_category == 'cold_start' get 0 interactions
    by construction (see config.yaml synthetic.interactions_per_user.cold_start),
    so 'cold_start' should never appear as a target_category value in the
    generated interactions table at all."""
    assert "cold_start" not in set(interactions["target_category"].unique())


def test_recent_flag_is_boolean(interactions):
    assert interactions["is_recent"].dtype == bool or set(interactions["is_recent"].unique()) <= {True, False, 0, 1}
