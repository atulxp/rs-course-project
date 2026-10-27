"""Streamlit frontend (Sections 21, 22, 23, 31 of the brief).

Run:
    streamlit run app/frontend/app.py

Talks directly to the in-process RecommendationOrchestrator (loaded once via
st.cache_resource) rather than over HTTP, since Streamlit and the model
objects live in the same python process for this demo -- the FastAPI service
(src/api/main.py) is the separate, network-facing way to serve the same
models, per Section 24.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.recommendation.pipeline import build_orchestrator
from src.recommendation.history import PROPERTY_COLUMNS, build_interaction_history
from src.models.rule_based import UserPreference
from src.utils.config import RESULTS_METRICS_DIR, RESULTS_PLOTS_DIR, RESULTS_TABLES_DIR

st.set_page_config(page_title="RE-RecSys-Inspired Real Estate Recommender", layout="wide")

METHOD_LABELS = {
    "cold_start_rule_based": "Cold Start (Rule-Based)",
    "cold_start_rule_based_fallback": "🧊 Cold Start (Rule-Based fallback)",
    "content_based": "Content-Based",
    "collaborative": "Collaborative",
    "hybrid": "Hybrid (Content + Collaborative)",
}


@st.cache_resource(show_spinner="Loading trained models (run scripts/train.py first if this fails)...")
def get_orchestrator():
    return build_orchestrator()


def init_session_state():
    if "live_interactions" not in st.session_state:
        st.session_state.live_interactions = []  # list of dict, appended during demo simulation
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = None


def current_interactions_df(orch, user_id: str) -> pd.DataFrame:
    base = orch.interactions[orch.interactions["user_id"] == user_id]
    live = pd.DataFrame([r for r in st.session_state.live_interactions if r["user_id"] == user_id])
    if live.empty:
        return base
    return pd.concat([base, live], ignore_index=True)


def render_property_card(rec, rank: int):
    d = rec.details
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**#{rank} · {d.get('prop_heading') or d.get('building_name') or d.get('property_id')}**")
            st.caption(f"{d.get('locality', '?')}, {d.get('city', '?')}")
            price = d.get("price")
            price_str = f"₹{price:,.0f}" if isinstance(price, (int, float)) else "N/A"
            st.write(
                f"**{price_str}**"
                + (" /month" if d.get("transaction_type") == "Rent" else "")
                + f"  ·  {d.get('bhk', '?')} BHK  ·  {d.get('area_sqft', '?')} sqft  ·  {d.get('furnish', '?')}"
            )
            with st.expander("Why this recommendation?"):
                for reason in rec.explanation:
                    st.write(f"- {reason}")
        with cols[1]:
            st.metric("Score", f"{rec.score:.3f}")
            st.caption(METHOD_LABELS.get(rec.method, rec.method))
            if st.button("View", key=f"view_{rank}_{d.get('property_id')}"):
                _simulate(d.get("property_id"), "detail_view")
            if st.button("Shortlist", key=f"short_{rank}_{d.get('property_id')}"):
                _simulate(d.get("property_id"), "shortlist")
            if st.button("Lead", key=f"lead_{rank}_{d.get('property_id')}"):
                _simulate(d.get("property_id"), "lead")


def _simulate(property_id: str, interaction_type: str):
    uid = st.session_state.selected_user_id
    existing = [r for r in st.session_state.live_interactions if r["user_id"] == uid]
    seq = len(existing) + 100000  # offset so it never collides with base interaction sequence numbers
    st.session_state.live_interactions.append(
        {
            "interaction_id": -1,
            "user_id": uid,
            "property_id": property_id,
            "interaction_type": interaction_type,
            "match_score": 1.0,
            "is_recent": True,
            "sequence": seq,
            "target_category": "demo_live",
        }
    )
    st.toast(f"Recorded '{interaction_type}' on {property_id}")
    st.rerun()


def page_activity(orch):
    st.title("My Activity")
    uid = st.session_state.selected_user_id
    if uid is None:
        st.info("Select a user on the Recommendations page to view their activity.")
        return

    interactions = current_interactions_df(orch, uid).copy()
    if interactions.empty:
        st.info("No interactions recorded for this user yet.")
        return

    history = build_interaction_history(interactions, orch.properties, PROPERTY_COLUMNS)

    tabs = st.tabs(["All Activity", "Viewed Properties", "Shortlisted Properties", "Leads"])
    filters = [None, {"detail_view", "image_view", "impression"}, {"shortlist"}, {"lead", "conversion"}]
    display_columns = [
        "property_id", "prop_heading", "property_type", "city", "locality", "price",
        "transaction_type", "bhk", "area_sqft", "interaction_type",
        "most_recent_interaction", "sequence",
    ]
    for tab, allowed_types in zip(tabs, filters):
        with tab:
            view = history if allowed_types is None else history[history["interaction_type"].isin(allowed_types)]
            if view.empty:
                st.info("No matching activity for this user.")
                continue
            st.dataframe(view[display_columns].sort_values("sequence", ascending=False), hide_index=True)
            st.caption("Interaction timestamps are unavailable because the current event schema stores sequence numbers, not wall-clock time.")
            for property_id, property_events in view.groupby("property_id", sort=False):
                property_row = property_events.iloc[0]
                title = property_row.get("prop_heading") or property_row.get("building_name") or property_id
                with st.expander(f"{title} · {property_id}"):
                    st.write(
                        f"{property_row.get('locality', '?')}, {property_row.get('city', '?')} · "
                        f"{property_row.get('bhk', '?')} BHK · {property_row.get('area_sqft', '?')} sqft"
                    )
                    st.write(f"Price: {property_row.get('price', 'N/A')} ({property_row.get('transaction_type', '?')})")
                    st.write("Events: " + ", ".join(property_events["interaction_type"].astype(str)))


def page_recommendations(orch):
    st.title("Real Estate Recommender")

    with st.sidebar:
        st.header("User / Demo Profile")
        mode = st.radio("Mode", ["Load Demo User", "Create New User"])

        if mode == "Load Demo User":
            demo_users = orch.users.sample(n=min(30, len(orch.users)), random_state=1)
            options = demo_users["user_id"].tolist()
            labels = {
                uid: f"{uid} · {row.city} · {row.transaction_type} · {row.preferred_bhk}BHK · {orch.get_user_category(uid)}"
                for uid, row in zip(options, demo_users.itertuples())
            }
            selected = st.selectbox("Choose a demo user", options, format_func=lambda u: labels[u])
            st.session_state.selected_user_id = selected
            explicit_prefs = None
            city = None
            transaction_type = None
        else:
            st.subheader("New User Preferences")
            city = st.selectbox("City", sorted(orch.properties["city"].unique()))
            transaction_type = st.selectbox("Transaction Type", ["Buy", "Rent"])
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
            ptype_options = sorted(orch.properties["property_type"].unique())
            property_type = st.selectbox("Property Type", ptype_options)
            furnish_options = sorted(orch.properties["furnish"].unique())
            furnish = st.selectbox("Furnishing", furnish_options)
            budget_max = st.number_input(
                "Maximum budget (INR)", min_value=0, value=10_000_000, step=100_000, format="%d"
            )
            budget_min = st.number_input(
                "Minimum budget (INR)", min_value=0, max_value=budget_max, value=0, step=100_000, format="%d"
            )
            area_min, area_max = st.slider("Area range (sqft)", 0, 5000, (500, 1500), step=50)
            locality_pool = sorted(orch.properties[orch.properties["city"] == city]["locality"].dropna().unique())
            preferred_localities = st.multiselect("Preferred Localities", locality_pool[:200])
            amenity_pool = sorted({a for lst in orch.properties["amenities"].dropna().head(2000) for a in str(lst).split(";") if a})
            preferred_amenities = st.multiselect("Preferred Amenities", amenity_pool[:100])

            new_id = "demo_new_user"
            st.session_state.selected_user_id = new_id
            explicit_prefs = {
                "city": city,
                "transaction_type": transaction_type,
                "preferred_bhk": bhk,
                "preferred_property_type": property_type,
                "preferred_furnish": furnish,
                "budget_min": budget_min,
                "budget_max": budget_max,
                "preferred_area_min": area_min,
                "preferred_area_max": area_max,
                "preferred_localities": json.dumps(preferred_localities),
                "preferred_amenities": json.dumps(preferred_amenities),
            }
            # ensure this synthetic user exists in orch.users lookup for category calc etc.
            if new_id not in orch.users["user_id"].values:
                row = dict(explicit_prefs)
                row["user_id"] = new_id
                row["archetype"] = "buyer" if transaction_type == "Buy" else "renter"
                orch.users = pd.concat([orch.users, pd.DataFrame([row])], ignore_index=True)

        top_k = st.slider("Number of recommendations", 3, 20, 10)
        alpha = st.slider("Hybrid alpha (content weight)", 0.0, 1.0, 0.5, 0.1)

        st.divider()
        st.subheader("Interaction Simulation")
        if st.button("Reset simulated interactions for this user"):
            uid = st.session_state.selected_user_id
            st.session_state.live_interactions = [r for r in st.session_state.live_interactions if r["user_id"] != uid]
            st.rerun()

    uid = st.session_state.selected_user_id
    if uid is None:
        st.info("Select or create a user in the sidebar to get started.")
        return

    # patch orchestrator's live interactions in for this session (non-persistent, demo only)
    live_df = current_interactions_df(orch, uid)
    orig_interactions = orch.interactions
    orch.interactions = pd.concat(
        [orig_interactions[orig_interactions["user_id"] != uid], live_df], ignore_index=True
    )

    category = orch.get_user_category(uid)
    n_total = len(live_df)
    n_recent = int(live_df["is_recent"].sum()) if n_total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("User Category", category.replace("_", " ").title())
    c2.metric("Total Interactions", n_total)
    c3.metric("Recent Interactions", n_recent)

    st.progress(min(n_total / 10, 1.0), text="Interaction history (demonstrates cold-start → short-term → long-term evolution)")

    try:
        if uid in orch.users["user_id"].values and orch.users[orch.users.user_id == uid].iloc[0].get("user_id") == "demo_new_user":
            recs = orch.recommend(uid, city=city, transaction_type=transaction_type, explicit_preferences=explicit_prefs, top_k=top_k, alpha=alpha)
        else:
            recs = orch.recommend(uid, top_k=top_k, alpha=alpha)
    except ValueError as e:
        st.error(str(e))
        orch.interactions = orig_interactions
        return

    if not recs:
        st.warning("No recommendations could be generated for this user's filters.")
    else:
        st.subheader(f"Recommendations ({METHOD_LABELS.get(recs[0].method, recs[0].method)})")
        for i, r in enumerate(recs, start=1):
            render_property_card(r, i)

    orch.interactions = orig_interactions  # restore global state (session interactions stay in st.session_state)


def page_evaluation():
    st.title("📊 Model Evaluation Dashboard")
    st.caption("All numbers below are produced by `scripts/evaluate.py` run against the SIMULATED evaluation environment.")

    summary_path = RESULTS_METRICS_DIR / "summary.json"
    if not summary_path.exists():
        st.warning("No evaluation results found. Run `python scripts/evaluate.py` first.")
        return
    with open(summary_path) as f:
        summary = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties", summary["n_properties"])
    c2.metric("Synthetic Users", summary["n_users"])
    c3.metric("Simulated Interactions", summary["n_interactions"])
    c4.metric("Best Hybrid Alpha (by NDCG@6)", summary.get("best_hybrid_alpha_by_ndcg", "N/A"))

    st.subheader("Model Comparison")
    mc_path = RESULTS_TABLES_DIR / "model_comparison.csv"
    if mc_path.exists():
        st.dataframe(pd.read_csv(mc_path, index_col=0).round(4))

    tabs = st.tabs(["Hybrid Alpha Sweep", "Category-wise", "City-wise", "Weighting Ablation", "Latency"])
    files = [
        "hybrid_alpha_sweep.csv",
        "category_performance.csv",
        "city_performance.csv",
        "weighting_scheme_ablation.csv",
        "latency_benchmark.csv",
    ]
    for tab, fname in zip(tabs, files):
        with tab:
            path = RESULTS_TABLES_DIR / fname
            if path.exists():
                st.dataframe(pd.read_csv(path, index_col=0).round(4))
            else:
                st.info(f"{fname} not found -- run scripts/evaluate.py")

    st.subheader("Plots")
    plot_files = sorted(RESULTS_PLOTS_DIR.glob("*.png")) if RESULTS_PLOTS_DIR.exists() else []
    if not plot_files:
        st.info("No plots found -- run scripts/generate_plots.py")
    else:
        cols = st.columns(3)
        for i, p in enumerate(plot_files):
            with cols[i % 3]:
                st.image(str(p), caption=p.stem, use_container_width=True)


def main():
    init_session_state()
    try:
        orch = get_orchestrator()
    except FileNotFoundError:
        st.error("Model artifacts not found. Run `python scripts/train.py` first, then restart this app.")
        return

    page = st.sidebar.radio("Page", ["Recommendations", "My Activity", "Model Evaluation"])
    st.sidebar.divider()
    if page == "Recommendations":
        page_recommendations(orch)
    elif page == "My Activity":
        page_activity(orch)
    else:
        page_evaluation()


if __name__ == "__main__":
    main()
