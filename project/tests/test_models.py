import pandas as pd
import pytest

from src.models.rule_based import RuleBasedRecommender, UserPreference, parse_user_row, score_property


def test_score_property_range(properties):
    user = UserPreference(city="Mumbai", transaction_type="Buy", preferred_bhk=2, budget_min=5_000_000, budget_max=9_000_000)
    row = properties.iloc[0]
    score = score_property(user, row)
    assert 0.0 <= score <= 1.0


def test_rule_based_recommend_returns_topk(properties, interactions, cfg):
    rb = RuleBasedRecommender(properties, interactions, cfg)
    user = UserPreference(city="Mumbai", transaction_type="Buy", preferred_bhk=2, budget_min=5_000_000, budget_max=9_000_000)
    recs = rb.recommend(user, top_k=5)
    assert len(recs) <= 5
    if len(recs):
        assert recs["score"].is_monotonic_decreasing


def test_rule_based_handles_unknown_city_gracefully(properties, interactions, cfg):
    rb = RuleBasedRecommender(properties, interactions, cfg)
    user = UserPreference(city="Atlantis", transaction_type="Buy")
    recs = rb.recommend(user, top_k=5)
    # should not raise; may be empty
    assert isinstance(recs, pd.DataFrame)


def test_content_based_recommend(properties, interactions, orchestrator):
    uid = interactions["user_id"].iloc[0]
    user_row = orchestrator.users[orchestrator.users.user_id == uid]
    if user_row.empty:
        pytest.skip("sample user not in users table")
    row = user_row.iloc[0]
    user_inter = interactions[interactions.user_id == uid]
    recs = orchestrator.content_based.recommend_for_interactions(user_inter, row["city"], row["transaction_type"], top_k=5)
    assert isinstance(recs, pd.DataFrame)
    if not recs.empty:
        assert (recs["score"] >= -1e-6).all()


def test_collaborative_recommend_for_known_user(orchestrator):
    known_user = next(iter(orchestrator.collaborative.user_to_idx.keys()))
    recs = orchestrator.collaborative.recommend(known_user, top_k=5)
    assert isinstance(recs, pd.DataFrame)


def test_collaborative_recommend_for_unknown_user_returns_empty(orchestrator):
    recs = orchestrator.collaborative.recommend("nonexistent_user_xyz", top_k=5)
    assert recs.empty


def test_hybrid_recommend(orchestrator):
    known_user = next(iter(orchestrator.collaborative.user_to_idx.keys()))
    user_inter = orchestrator.interactions[orchestrator.interactions.user_id == known_user]
    row = orchestrator.users[orchestrator.users.user_id == known_user].iloc[0]
    recs = orchestrator.hybrid.recommend(known_user, user_inter, row["city"], row["transaction_type"], top_k=5)
    assert isinstance(recs, pd.DataFrame)


def test_orchestrator_all_four_categories_reachable(orchestrator):
    categories_seen = set()
    for uid in orchestrator.users["user_id"].head(400):
        categories_seen.add(orchestrator.get_user_category(uid))
        if len(categories_seen) == 4:
            break
    assert categories_seen == {"cold_start", "short_term", "long_term", "short_long_term"}


def test_orchestrator_recommend_cold_start_user(orchestrator):
    for uid in orchestrator.users["user_id"]:
        if orchestrator.get_user_category(uid) == "cold_start":
            recs = orchestrator.recommend(uid, top_k=5)
            assert isinstance(recs, list)
            return
    pytest.skip("no cold_start user found")


def test_orchestrator_unknown_user_raises(orchestrator):
    with pytest.raises(ValueError):
        orchestrator.recommend("totally_unknown_user_id_123", top_k=5)


def test_orchestrator_explicit_preferences_for_new_user(orchestrator):
    prefs = {
        "city": "Mumbai",
        "transaction_type": "Buy",
        "preferred_bhk": 2,
        "budget_min": 5_000_000,
        "budget_max": 9_000_000,
    }
    recs = orchestrator.recommend("brand_new_user_not_in_table", explicit_preferences=prefs, top_k=5)
    assert isinstance(recs, list)


def test_recommendation_explanations_nonempty(orchestrator):
    for uid in orchestrator.users["user_id"].head(50):
        recs = orchestrator.recommend(uid, top_k=3)
        if recs:
            for r in recs:
                assert len(r.explanation) > 0
            return


def test_explanations_do_not_claim_unsupported_signals(orchestrator):
    user = UserPreference(city="Mumbai", transaction_type="Buy")
    row = pd.Series({"score": 0.0, "content_score": 0.0, "collaborative_score": 0.0})
    empty_history = pd.DataFrame(columns=["interaction_type"])

    content_reasons = orchestrator._content_explanation(user, row, empty_history)
    hybrid_reasons = orchestrator._hybrid_explanation(user, row, empty_history)

    assert not any("interaction history" in reason.lower() for reason in content_reasons)
    assert not any("similar users" in reason.lower() for reason in hybrid_reasons)


def test_explanations_include_supported_model_signals(orchestrator):
    user = UserPreference(city="Mumbai", transaction_type="Buy")
    row = pd.Series({"score": 0.8, "content_score": 0.8, "collaborative_score": 0.4})
    history = pd.DataFrame([{"interaction_type": "shortlist"}])

    content_reasons = orchestrator._content_explanation(user, row, history)
    hybrid_reasons = orchestrator._hybrid_explanation(user, row, history)

    assert any("shortlisted" in reason for reason in content_reasons)
    assert any("similar users" in reason.lower() for reason in hybrid_reasons)
