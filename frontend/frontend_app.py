"""
CartIQ Streamlit Frontend — Incremental Cart-Building Flow
===========================================================
REWRITE vs. the original single-submit form:
  - User picks city/cuisine/meal_time once, then builds the cart
    incrementally — adding items one at a time and watching
    recommendations update live after each addition.
  - Recommendations can themselves be added to the cart with one click,
    creating the interactive loop an interviewer expects to see.
  - /full_pipeline is called on every cart change, not once at the end.
  - Cart state lives in st.session_state so it survives Streamlit reruns.
  - API_URL is read from the API_URL environment variable (defaults to
    localhost for local dev), so the frontend can be pointed at a
    deployed backend without code changes.

CHANGES vs. the original:
  - Cuisine dropdown matches the actual catalog (CUISINES from
    generate_items.py) — same fix as before, preserved.
  - City list fetched from /cities endpoint (or falls back to the full
    10-city list), not hardcoded to the old 4-city list.
  - Item display shows real names/prices/categories from the API, not
    placeholder labels.
"""

import streamlit as st
import requests
import os
import time
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────
# API_URL resolution order, so the same code runs everywhere unchanged:
#   1. st.secrets["API_URL"]  -> Streamlit Community Cloud (set in app Secrets)
#   2. os.environ["API_URL"]  -> Docker / docker-compose / Render
#   3. localhost default      -> bare local dev
def _resolve_api_url():
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        # No secrets.toml present (e.g. Docker/local) — st.secrets access can
        # raise rather than return empty; fall through to env/default.
        pass
    return os.environ.get("API_URL", "http://127.0.0.1:8000")


API_URL = _resolve_api_url()

st.set_page_config(page_title="CartIQ - AI Cart Intelligence", page_icon="\U0001F9E0", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"]  { background-color: #0e1117; color: white; }
.hero-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-subtitle { font-size: 1.2rem; color: #a0a0a0; }
.card {
    background: #161b22; padding: 1rem; border-radius: 14px;
    margin-bottom: 1.2rem; border: 1px solid #2a2f36; transition: 0.3s ease;
}
.card:hover { border: 1px solid #00f5a0; transform: scale(1.01); }
.metric-box {
    background: #161b22; padding: 1rem; border-radius: 14px;
    margin-bottom: 1rem; border: 1px solid #2a2f36;
}
.cart-item {
    background: #1c2333; padding: 0.6rem 1rem; border-radius: 10px;
    margin-bottom: 0.5rem; border-left: 3px solid #00f5a0;
}
.rec-card {
    background: #161b22; padding: 0.8rem 1rem; border-radius: 12px;
    margin-bottom: 0.8rem; border: 1px solid #2a2f36;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">CartIQ - AI Cart Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Real-time LightGBM Ranking Engine for Smart Cart Optimization</div>', unsafe_allow_html=True)
st.divider()

# ── Session State Initialization ─────────────────────────────────────
# Cart state persists across Streamlit reruns. This is what makes the
# incremental flow work — adding an item triggers a rerun, but the cart
# survives because it's in session_state, not a local variable.
if "cart_items" not in st.session_state:
    st.session_state.cart_items = []       # list of item_id (int)
    st.session_state.cart_details = []     # list of {item_id, item_name, category, price}
    st.session_state.recommendations = [] # latest recommendations from API
    st.session_state.rec_method = ""
    st.session_state.rec_latency = 0.0


# ── Helper: fetch recommendations from API ───────────────────────────
def fetch_recommendations(city, cuisine, meal_time, cart_items):
    """Call /full_pipeline and return (recommendations, method, latency_ms)."""
    try:
        start = time.time()
        resp = requests.post(
            f"{API_URL}/full_pipeline",
            json={
                "cart_items": cart_items,
                "city": city,
                "meal_time": meal_time,
                "cuisine": cuisine,
            },
            timeout=10,
        )
        latency = round((time.time() - start) * 1000, 2)
        if resp.status_code == 200:
            data = resp.json()
            recs = data.get("recommendations", [])
            # Filter out items already in cart (belt-and-suspenders)
            recs = [r for r in recs if r["item_id"] not in cart_items]
            method = "model" if any(r.get("score") is not None for r in recs) else "fallback"
            return recs, method, latency
        else:
            return [], "error", latency
    except Exception:
        return [], "offline", 0.0


# ── Helper: fetch options from API (with fallbacks) ──────────────────
@st.cache_data(ttl=300)
def fetch_cities():
    try:
        resp = requests.get(f"{API_URL}/cities", timeout=5)
        return resp.json()["cities"]
    except Exception:
        return ["Bangalore", "Delhi", "Mumbai", "Hyderabad", "Chennai",
                "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]


@st.cache_data(ttl=300)
def fetch_cuisines():
    try:
        resp = requests.get(f"{API_URL}/cuisines", timeout=5)
        return resp.json()["cuisines"]
    except Exception:
        return ["North Indian", "Mughlai", "Chinese", "Italian", "South Indian"]


@st.cache_data(ttl=300)
def fetch_items_for_cuisine(cuisine):
    try:
        resp = requests.get(f"{API_URL}/items", params={"cuisine": cuisine}, timeout=5)
        return resp.json()
    except Exception:
        return []


# ── Sidebar: Context Selection ───────────────────────────────────────
st.sidebar.title("\U0001F6D2 Order Context")

cities = fetch_cities()
cuisines = fetch_cuisines()

city = st.sidebar.selectbox("City", cities, key="city_select")
cuisine = st.sidebar.selectbox("Cuisine", cuisines, key="cuisine_select")
meal_time = st.sidebar.selectbox("Meal Time", ["Breakfast", "Lunch", "Snacks", "Dinner"], key="meal_select")

st.sidebar.divider()

# Clear cart button
if st.sidebar.button("\U0001F5D1\uFE0F Clear Cart & Start Over", use_container_width=True):
    st.session_state.cart_items = []
    st.session_state.cart_details = []
    st.session_state.recommendations = []
    st.session_state.rec_method = ""
    st.session_state.rec_latency = 0.0
    st.rerun()

# ── Sidebar: Cart Summary ───────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader(f"\U0001F6D2 Your Cart ({len(st.session_state.cart_items)} items)")
if st.session_state.cart_details:
    cart_total = 0
    for item in st.session_state.cart_details:
        st.sidebar.markdown(
            f'<div class="cart-item">'
            f'<strong>{item["item_name"]}</strong><br>'
            f'<span style="color:#a0a0a0">{item["category"]} · ₹{item["price"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        cart_total += item["price"]
    st.sidebar.metric("Cart Total", f"₹{cart_total}")
else:
    st.sidebar.info("Cart is empty — add items below to see recommendations.")

# Sidebar model info
st.sidebar.divider()
st.sidebar.markdown('<div class="metric-box">', unsafe_allow_html=True)
st.sidebar.markdown("### \u2699\uFE0F Model Engine")
st.sidebar.markdown("""
- LightGBM LambdaRank model
- Real-time inference per cart change
- Cart-sensitive feature encoding
- Context-aware (city/cuisine/meal)
""")
st.sidebar.markdown('</div>', unsafe_allow_html=True)


# ── Main Area Layout ─────────────────────────────────────────────────
col_menu, col_recs = st.columns([3, 2])

# ── Left Column: Menu Items ──────────────────────────────────────────
with col_menu:
    st.subheader(f"\U0001F37D\uFE0F {cuisine} Menu")

    menu_items = fetch_items_for_cuisine(cuisine)

    if not menu_items:
        st.warning("Could not load menu items. Is the backend running?")
    else:
        # Group items by category for cleaner display
        categories_order = ["Main", "Side", "Drink", "Dessert"]
        items_by_cat = {}
        for item in menu_items:
            cat = item.get("category", "Other")
            items_by_cat.setdefault(cat, []).append(item)

        for cat in categories_order:
            cat_items = items_by_cat.get(cat, [])
            if not cat_items:
                continue

            st.markdown(f"**{cat}s**")
            for item in cat_items:
                item_id = item["item_id"]
                already_in_cart = item_id in st.session_state.cart_items

                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    label = f"{item['item_name']}"
                    if already_in_cart:
                        label += "  ✅"
                    st.write(label)
                with c2:
                    st.write(f"₹{item['price']}")
                with c3:
                    if already_in_cart:
                        st.write("In cart")
                    else:
                        if st.button("Add", key=f"add_{item_id}", use_container_width=True):
                            st.session_state.cart_items.append(item_id)
                            st.session_state.cart_details.append({
                                "item_id": item_id,
                                "item_name": item["item_name"],
                                "category": item["category"],
                                "price": item["price"],
                            })
                            # Fetch fresh recommendations after adding
                            recs, method, latency = fetch_recommendations(
                                city, cuisine, meal_time,
                                st.session_state.cart_items,
                            )
                            st.session_state.recommendations = recs
                            st.session_state.rec_method = method
                            st.session_state.rec_latency = latency
                            st.rerun()
            st.write("")  # spacing between categories

# ── Right Column: Recommendations ────────────────────────────────────
with col_recs:
    st.subheader("\U0001F9E0 AI Recommendations")

    if not st.session_state.cart_items:
        st.info("\U0001F449 Add your first item from the menu to see personalized recommendations.")
    elif not st.session_state.recommendations:
        # Cart has items but no recommendations fetched yet (e.g. page reload)
        # Fetch now
        recs, method, latency = fetch_recommendations(
            city, cuisine, meal_time,
            st.session_state.cart_items,
        )
        st.session_state.recommendations = recs
        st.session_state.rec_method = method
        st.session_state.rec_latency = latency

    if st.session_state.recommendations:
        st.caption(
            f"\u26A1 {st.session_state.rec_latency}ms · "
            f"via {'model ranking' if st.session_state.rec_method == 'model' else 'popularity fallback'}"
        )

        for idx, rec in enumerate(st.session_state.recommendations[:5], start=1):
            score = rec.get("score")
            item_id = rec["item_id"]
            already_in_cart = item_id in st.session_state.cart_items

            medal = "\U0001F947" if idx == 1 else ("\U0001F948" if idx == 2 else ("\U0001F949" if idx == 3 else "\U0001F3C6"))

            st.markdown(f'<div class="rec-card">', unsafe_allow_html=True)
            rc1, rc2 = st.columns([4, 1])
            with rc1:
                st.markdown(f"**{medal} #{idx} — {rec['item_name']}**")
                if score is not None:
                    # Confidence tier is computed SERVER-SIDE from the
                    # calibrated score distribution (terciles of held-out
                    # top-1 scores) and returned by the API. We just render
                    # it — no thresholds hardcoded here, so nothing drifts
                    # if the model is retrained. LGBMRanker scores aren't
                    # probabilities, so a fixed 0-1 cutoff in the UI would
                    # be meaningless; this is the fix for that.
                    tier = rec.get("confidence", "")
                    dot = {"High": "\U0001F7E2", "Medium": "\U0001F7E1", "Low": "\U0001F534"}.get(tier, "⚪")
                    st.caption(f"Score: {round(score, 4)} · {dot} {tier} confidence")
                else:
                    reason = rec.get("fallback_reason", "n/a")
                    st.caption(f"Fallback ({reason})")
            with rc2:
                if already_in_cart:
                    st.write("✅")
                else:
                    if st.button("Add \u2795", key=f"rec_{item_id}", use_container_width=True):
                        st.session_state.cart_items.append(item_id)
                        st.session_state.cart_details.append({
                            "item_id": item_id,
                            "item_name": rec["item_name"],
                            "category": rec.get("category", ""),
                            "price": rec.get("price", 0),
                        })
                        # Re-fetch recommendations with updated cart
                        recs, method, latency = fetch_recommendations(
                            city, cuisine, meal_time,
                            st.session_state.cart_items,
                        )
                        st.session_state.recommendations = recs
                        st.session_state.rec_method = method
                        st.session_state.rec_latency = latency
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.cart_items and not st.session_state.recommendations:
        st.success("\U0001F389 No more items to recommend — your cart looks complete!")


# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.caption(f"CartIQ \u2022 Live Ranking Engine \u2022 {datetime.now().year}")
