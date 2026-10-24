import pandas as pd

from src.recommendation.history import build_interaction_history


def test_history_preserves_repeated_events_and_latest_event():
    interactions = pd.DataFrame(
        [
            {"interaction_id": 1, "user_id": "u1", "property_id": "p1", "interaction_type": "detail_view", "sequence": 1},
            {"interaction_id": 2, "user_id": "u1", "property_id": "p1", "interaction_type": "shortlist", "sequence": 2},
            {"interaction_id": 3, "user_id": "u2", "property_id": "p2", "interaction_type": "lead", "sequence": 1},
        ]
    )
    properties = pd.DataFrame(
        [
            {"property_id": "p1", "prop_heading": "First property", "city": "Mumbai"},
            {"property_id": "p2", "prop_heading": "Second property", "city": "Pune"},
        ]
    )

    history = build_interaction_history(interactions, properties)

    assert len(history) == 3
    assert list(history.loc[history["property_id"] == "p1", "interaction_type"]) == ["detail_view", "shortlist"]
    assert set(history["most_recent_interaction"]) == {"shortlist", "lead"}
    assert history.loc[history["property_id"] == "p1", "city"].iloc[0] == "Mumbai"