"""FastAPI backend (Section 24 of the brief).

Run:
    uvicorn src.api.main:app --reload

The orchestrator (and every model it wraps) is loaded ONCE at startup from
the persisted artifacts written by scripts/train.py -- the API never
retrains anything at request time (Section 27/28).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.recommendation.pipeline import build_orchestrator
from src.utils.config import DATA_PROCESSED_DIR, RESULTS_METRICS_DIR, load_config

STATE = {"orchestrator": None, "loaded_at": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["orchestrator"] = build_orchestrator()
        STATE["loaded_at"] = time.time()
        print(f"[api] orchestrator loaded: {len(STATE['orchestrator'].properties)} properties, "
              f"{len(STATE['orchestrator'].users)} users, {len(STATE['orchestrator'].interactions)} interactions")
    except FileNotFoundError as e:
        print(f"[api] WARNING: model artifacts not found ({e}). Run `python scripts/train.py` first. "
              f"The API will start but endpoints will return 503 until trained.")
    yield
    STATE["orchestrator"] = None


app = FastAPI(
    title="RE-RecSys-Inspired Real-Estate Recommendation API",
    description=(
        "An academic, RE-RecSys-inspired (Venkatesh et al., CODS-COMAD 2024) recommendation API "
        "built on public Indian real-estate listings with SIMULATED user interactions. "
        "Not affiliated with Housing.com."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def get_orchestrator():
    if STATE["orchestrator"] is None:
        raise HTTPException(status_code=503, detail="Models not trained/loaded yet. Run `python scripts/train.py` first.")
    return STATE["orchestrator"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    city: str
    transaction_type: str = Field(..., description="'Buy' or 'Rent'")
    preferred_bhk: Optional[int] = None
    preferred_property_type: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    preferred_area_min: Optional[float] = None
    preferred_area_max: Optional[float] = None
    preferred_furnish: Optional[str] = None
    preferred_localities: List[str] = []
    preferred_amenities: List[str] = []


class InteractionCreate(BaseModel):
    property_id: str
    interaction_type: str = Field(..., description="impression|image_view|detail_view|amenity_view|shortlist|lead|conversion")
    is_recent: bool = True


class RecommendationRequest(BaseModel):
    user_id: str
    city: Optional[str] = None
    transaction_type: Optional[str] = None
    top_k: int = 10
    alpha: Optional[float] = None
    explicit_preferences: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": STATE["orchestrator"] is not None}


@app.get("/properties")
def list_properties(city: Optional[str] = None, transaction_type: Optional[str] = None, limit: int = 50, offset: int = 0):
    orch = get_orchestrator()
    df = orch.properties
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if transaction_type:
        df = df[df["transaction_type"].str.lower() == transaction_type.lower()]
    total = len(df)
    page = df.iloc[offset : offset + min(limit, 200)]
    return {"total": total, "results": page.fillna("").to_dict(orient="records")}


@app.get("/properties/{property_id}")
def get_property(property_id: str):
    orch = get_orchestrator()
    row = orch.properties[orch.properties["property_id"] == property_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"property_id '{property_id}' not found")
    return row.iloc[0].fillna("").to_dict()


@app.post("/users")
def create_user(user: UserCreate):
    orch = get_orchestrator()
    new_id = f"user_api_{int(time.time() * 1000)}"
    row = user.model_dump()
    row["user_id"] = new_id
    row["archetype"] = "buyer" if user.transaction_type.lower() == "buy" else "renter"
    row["preferred_localities"] = row["preferred_localities"]
    row["preferred_amenities"] = row["preferred_amenities"]
    orch.users = pd.concat([orch.users, pd.DataFrame([row])], ignore_index=True)
    return {"user_id": new_id}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    orch = get_orchestrator()
    row = orch.users[orch.users["user_id"] == user_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not found")
    return row.iloc[0].fillna("").to_dict()


@app.get("/users/{user_id}/profile")
def get_user_profile(user_id: str):
    orch = get_orchestrator()
    row = orch.users[orch.users["user_id"] == user_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not found")
    category = orch.get_user_category(user_id)
    interactions = orch.interactions[orch.interactions["user_id"] == user_id]
    return {
        "user_id": user_id,
        "preferences": row.iloc[0].fillna("").to_dict(),
        "category": category,
        "total_interactions": int(len(interactions)),
        "recent_interactions": int(interactions["is_recent"].sum()) if len(interactions) else 0,
        "interaction_breakdown": interactions["interaction_type"].value_counts().to_dict(),
    }


@app.post("/users/{user_id}/interactions")
def add_interaction(user_id: str, interaction: InteractionCreate):
    orch = get_orchestrator()
    if orch.users[orch.users["user_id"] == user_id].empty:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not found")
    if orch.properties[orch.properties["property_id"] == interaction.property_id].empty:
        raise HTTPException(status_code=404, detail=f"property_id '{interaction.property_id}' not found")

    existing = orch.interactions[orch.interactions["user_id"] == user_id]
    next_seq = int(existing["sequence"].max()) + 1 if len(existing) else 0
    next_id = int(orch.interactions["interaction_id"].max()) + 1 if len(orch.interactions) else 0

    from src.models.rule_based import parse_user_row, score_property

    user_row = orch.users[orch.users["user_id"] == user_id].iloc[0]
    user_pref = parse_user_row(user_row)
    prop_row = orch.properties[orch.properties["property_id"] == interaction.property_id].iloc[0]
    match_score = score_property(user_pref, prop_row)

    new_row = {
        "interaction_id": next_id,
        "user_id": user_id,
        "property_id": interaction.property_id,
        "interaction_type": interaction.interaction_type,
        "match_score": round(match_score, 4),
        "is_recent": interaction.is_recent,
        "sequence": next_seq,
        "target_category": "api_live",
    }
    orch.interactions = pd.concat([orch.interactions, pd.DataFrame([new_row])], ignore_index=True)
    # NOTE (Section 23): the content model reflects this immediately (it reads
    # orch.interactions live on every request); the collaborative ALS model is
    # intentionally NOT retrained here -- see README "Interaction Simulation"
    # section for why (paper: ALS retrains periodically, not per-interaction).
    new_category = orch.get_user_category(user_id)
    return {"status": "recorded", "new_category": new_category}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_k: int = 10, city: Optional[str] = None, transaction_type: Optional[str] = None, alpha: Optional[float] = None):
    orch = get_orchestrator()
    try:
        results = orch.recommend(user_id, city=city, transaction_type=transaction_type, top_k=top_k, alpha=alpha)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"user_id": user_id, "category": orch.get_user_category(user_id), "recommendations": [r.__dict__ for r in results]}


@app.post("/recommendations")
def post_recommendations(req: RecommendationRequest):
    orch = get_orchestrator()
    try:
        results = orch.recommend(
            req.user_id,
            city=req.city,
            transaction_type=req.transaction_type,
            explicit_preferences=req.explicit_preferences,
            top_k=req.top_k,
            alpha=req.alpha,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"user_id": req.user_id, "category": orch.get_user_category(req.user_id), "recommendations": [r.__dict__ for r in results]}


@app.post("/models/train")
def trigger_train():
    raise HTTPException(
        status_code=501,
        detail="Training is a long-running batch job by design (Section 27/28: models are trained "
        "offline, not per-request). Run `python scripts/train.py` from a terminal instead.",
    )


@app.get("/models/status")
def models_status():
    orch = STATE["orchestrator"]
    if orch is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "loaded_at": STATE["loaded_at"],
        "n_properties": len(orch.properties),
        "n_users": len(orch.users),
        "n_interactions": len(orch.interactions),
        "collaborative_users": len(orch.collaborative.user_to_idx),
        "collaborative_items": len(orch.collaborative.item_to_idx),
    }


@app.get("/evaluation/results")
def evaluation_results():
    path = RESULTS_METRICS_DIR / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results yet. Run `python scripts/evaluate.py` first.")
    import json

    with open(path) as f:
        return json.load(f)
