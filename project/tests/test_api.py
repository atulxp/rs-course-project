import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_status(client):
    r = client.get("/models/status")
    assert r.status_code == 200


def test_list_properties(client):
    r = client.get("/properties?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) <= 5


def test_get_property_not_found(client):
    r = client.get("/properties/does_not_exist_xyz")
    assert r.status_code == 404


def test_get_user_not_found(client):
    r = client.get("/users/does_not_exist_xyz")
    assert r.status_code == 404


def test_create_user_and_get_recommendations(client):
    payload = {
        "city": "Mumbai",
        "transaction_type": "Buy",
        "preferred_bhk": 2,
        "budget_min": 5000000,
        "budget_max": 9000000,
        "preferred_area_min": 500,
        "preferred_area_max": 1200,
        "preferred_furnish": "Furnished",
        "preferred_localities": [],
        "preferred_amenities": [],
    }
    r = client.post("/users", json=payload)
    assert r.status_code == 200
    user_id = r.json()["user_id"]

    r2 = client.get(f"/recommendations/{user_id}?top_k=5")
    assert r2.status_code == 200
    body = r2.json()
    assert body["category"] == "cold_start"
    assert isinstance(body["recommendations"], list)


def test_recommendations_for_unknown_user(client):
    r = client.get("/recommendations/totally_unknown_user_999")
    assert r.status_code == 404


def test_add_interaction_and_profile_updates(client):
    payload = {
        "city": "Gurgaon",
        "transaction_type": "Rent",
        "preferred_bhk": 2,
        "budget_min": 20000,
        "budget_max": 40000,
    }
    r = client.post("/users", json=payload)
    user_id = r.json()["user_id"]

    props = client.get("/properties?city=Gurgaon&transaction_type=Rent&limit=1").json()["results"]
    if not props:
        pytest.skip("no Gurgaon rent properties available")
    prop_id = props[0]["property_id"]

    r2 = client.post(f"/users/{user_id}/interactions", json={"property_id": prop_id, "interaction_type": "lead", "is_recent": True})
    assert r2.status_code == 200

    r3 = client.get(f"/users/{user_id}/profile")
    assert r3.status_code == 200
    assert r3.json()["total_interactions"] >= 1


def test_evaluation_results_endpoint(client):
    r = client.get("/evaluation/results")
    # either 200 (if scripts/evaluate.py has been run) or 404
    assert r.status_code in (200, 404)


def test_post_recommendations_endpoint(client):
    payload = {
        "user_id": "brand_new_via_post",
        "city": "Kolkata",
        "transaction_type": "Buy",
        "top_k": 5,
        "explicit_preferences": {
            "city": "Kolkata",
            "transaction_type": "Buy",
            "preferred_bhk": 3,
            "budget_min": 3000000,
            "budget_max": 8000000,
        },
    }
    r = client.post("/recommendations", json=payload)
    assert r.status_code == 200
