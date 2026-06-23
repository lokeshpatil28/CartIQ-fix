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
  - API_URL is read from st.secrets / the API_URL env var (defaults to
    localhost for local dev), so the frontend can be pointed at a
    deployed backend without code changes.

VISUAL DESIGN — "itemized order ticket"
  The styling is deliberately built on restaurant/receipt vernacular, not a
  generic AI-dashboard look:
    - Typefaces: Fraunces (characterful serif, printed-menu headers),
      IBM Plex Sans (restrained UI body), IBM Plex Mono (ALL numerics —
      scores, prices, latency — because receipts and order tickets are
      set in monospace).
    - Palette: warm paper/receipt tones; ONE chromatic accent (--stamp, an
      order-stamp brick red) doing real work — it marks only the top
      recommendation and its primary action. Everything else is neutral.
    - Signature element: recommendations are laid out as itemized receipt
      lines with the score/confidence as a quiet, precisely aligned
      monospace data column (a 3-pip meter, not a colored badge).
  This is a visual/typographic layer only — the functionality, the API
  contract, and the calibrated confidence tiers are unchanged.
"""

import streamlit as st
import requests
import os
import time
from datetime import datetime

# set_page_config MUST be the first Streamlit command executed — before ANY
# other st.* call. st.secrets access counts as a Streamlit command, so the
# API_URL resolution (which reads st.secrets) must come AFTER this line, not
# before it. Keep set_page_config immediately after the imports.
st.set_page_config(page_title="CartIQ — Cart Add-on Ranking", layout="wide")


# ── Configuration ────────────────────────────────────────────────────
# API_URL resolution order, so the same code runs everywhere unchanged:
#   1. os.environ["API_URL"]  -> Docker / docker-compose / Render
#   2. st.secrets["API_URL"]  -> Streamlit Community Cloud (set in app Secrets)
#   3. localhost default      -> bare local dev
# Env is checked FIRST so that in Docker/Render (where env is set) we never
# touch st.secrets — accessing st.secrets with no secrets.toml present makes
# Streamlit render a "No secrets found" warning box. On Streamlit Cloud env
# is unset and the secret exists, so the secrets branch runs cleanly there.
def _resolve_api_url():
    env = os.environ.get("API_URL")
    if env:
        return env
    try:
        return st.secrets["API_URL"]
    except Exception:
        return "http://127.0.0.1:8000"


API_URL = _resolve_api_url()

# ── Design system (the one injected style block) ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root{
  --paper:#F2EEE4;   /* page — warm receipt stock        */
  --ticket:#FBF9F4;  /* surface — fresh receipt paper     */
  --ink:#2B2724;     /* primary text — warm espresso      */
  --muted:#6B6258;   /* secondary text & data labels      */
  --rule:#D8D0C0;    /* hairlines, receipt rules, leaders */
  --stamp:#B4341F;   /* THE accent — order-stamp red      */
}

/* surfaces */
.stApp, [data-testid="stAppViewContainer"]{ background:var(--paper); }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stSidebar"]{ background:var(--ticket); border-right:1px solid var(--rule); }
#MainMenu, [data-testid="stToolbar"], footer{ visibility:hidden; }

/* base typography */
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  color:var(--ink);
}
.stMarkdown p, label, span, li, [data-testid="stCaptionContainer"]{ color:var(--ink); }
hr{ border-color:var(--rule) !important; margin:.6rem 0; }

.mono{ font-family:'IBM Plex Mono', ui-monospace, monospace; font-variant-numeric:tabular-nums; }

/* masthead */
.kicker{ font-family:'IBM Plex Mono', monospace; text-transform:uppercase;
         letter-spacing:.2em; font-size:.68rem; color:var(--muted); }
.wordmark{ font-family:'Fraunces', Georgia, serif; font-weight:600;
           font-size:2.7rem; line-height:1.04; letter-spacing:-.01em;
           color:var(--ink); margin:.15rem 0 0; }
.tagline{ color:var(--muted); font-size:.95rem; max-width:60ch; margin:.35rem 0 0; }
.section-head{ font-family:'Fraunces', Georgia, serif; font-weight:600;
               font-size:1.45rem; color:var(--ink); margin:.1rem 0 .2rem; }
.course-head{ font-family:'Fraunces', Georgia, serif; font-weight:600;
              font-size:1.05rem; color:var(--ink); margin:1.1rem 0 .35rem;
              padding-bottom:.25rem; border-bottom:1px solid var(--rule); }

/* left: printed menu rows */
.menu-row{ display:flex; align-items:baseline; gap:.55rem; padding:.18rem 0; }
.mi-name{ color:var(--ink); }
.mi-name.added{ color:var(--muted); }
.leader{ flex:1 1 auto; min-width:1rem; border-bottom:1px dotted var(--rule);
         transform:translateY(-.2rem); }
.mi-price{ font-family:'IBM Plex Mono', monospace; color:var(--muted); font-size:.9rem; }
.tag{ font-family:'IBM Plex Mono', monospace; font-size:.6rem; text-transform:uppercase;
      letter-spacing:.12em; color:var(--muted); }

/* right: recommendation receipt lines (the signature element) */
.ticket{ padding:.5rem 0 .55rem; border-bottom:1px dotted var(--rule); }
.ticket.top{ border-left:3px solid var(--stamp); padding-left:.6rem; }
.tl-row1{ display:flex; align-items:baseline; gap:.5rem; }
.rank{ font-family:'IBM Plex Mono', monospace; color:var(--muted); font-size:.8rem; }
.tl-name{ color:var(--ink); font-weight:500; }
.stamp-tag{ font-family:'IBM Plex Mono', monospace; font-size:.58rem; text-transform:uppercase;
            letter-spacing:.14em; color:#fff; background:var(--stamp);
            padding:.06rem .32rem; border-radius:1px; }
.tl-spacer{ flex:1 1 auto; min-width:.5rem; border-bottom:1px dotted var(--rule);
            transform:translateY(-.2rem); }
.tl-price{ font-family:'IBM Plex Mono', monospace; color:var(--muted); font-size:.9rem; }
.tl-row2{ display:flex; align-items:baseline; gap:1.1rem; margin-top:.3rem;
          font-family:'IBM Plex Mono', monospace; font-size:.74rem; color:var(--muted); }
.tl-row2 .val{ color:var(--ink); }
.pip{ letter-spacing:.1em; }
.pip.on{ color:var(--ink); }
.pip.off{ color:var(--rule); }
.tier{ text-transform:uppercase; letter-spacing:.12em; }

/* sidebar: guest check */
.check-line{ display:flex; align-items:baseline; gap:.5rem; padding:.22rem 0;
             border-bottom:1px dotted var(--rule); }
.check-name{ color:var(--ink); font-size:.9rem; }
.check-cat{ font-family:'IBM Plex Mono', monospace; font-size:.58rem; text-transform:uppercase;
            letter-spacing:.1em; color:var(--muted); margin-left:.15rem; }
.check-price{ flex:0 0 auto; margin-left:auto; font-family:'IBM Plex Mono', monospace;
              color:var(--muted); font-size:.85rem; }
.check-total{ display:flex; justify-content:space-between; align-items:baseline;
              margin-top:.6rem; font-family:'IBM Plex Mono', monospace; }
.check-total .t-lbl{ text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; color:var(--muted); }
.check-total .t-val{ font-size:1.15rem; color:var(--ink); font-weight:600; }
.note{ color:var(--muted); font-size:.85rem; }
.spec li{ color:var(--muted); font-size:.82rem; }

/* buttons — quiet order affordances */
.stButton > button{ font-family:'IBM Plex Sans', sans-serif; font-weight:500; font-size:.78rem;
  color:var(--ink); background:var(--ticket); border:1px solid var(--rule);
  border-radius:2px; padding:.28rem .55rem; box-shadow:none !important;
  transition:border-color .15s ease, background .15s ease; }
.stButton > button:hover{ border-color:var(--ink); background:#fff; color:var(--ink); }
.stButton > button:active{ background:var(--paper); }
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"]{
  background:var(--stamp); color:#fff; border-color:var(--stamp); }
.stButton > button[kind="primary"]:hover{ background:#94291A; border-color:#94291A; color:#fff; }

/* selects */
[data-baseweb="select"] > div{ background:var(--ticket); border:1px solid var(--rule);
  border-radius:2px; color:var(--ink); }
.stSelectbox label, [data-testid="stWidgetLabel"] p{ font-family:'IBM Plex Mono', monospace;
  text-transform:uppercase; letter-spacing:.12em; font-size:.66rem; color:var(--muted); }

/* visible keyboard focus everywhere */
*:focus-visible{ outline:2px solid var(--stamp); outline-offset:2px; }
.stButton > button:focus-visible{ outline:2px solid var(--stamp); outline-offset:2px; }

@media (max-width:640px){
  .wordmark{ font-size:2.05rem; }
  .section-head{ font-size:1.25rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="kicker">Cart add-on ranking &middot; live</div>'
    '<div class="wordmark">CartIQ</div>'
    '<div class="tagline">Real-time add-on recommendations for a food-delivery cart, '
    'ranked by a LightGBM learning-to-rank model conditioned on city, cuisine, meal time, '
    'and the items already in the cart.</div>',
    unsafe_allow_html=True,
)
st.markdown("<hr/>", unsafe_allow_html=True)

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


# ── Presentation helper: confidence as a quiet 3-pip mono meter ──────
def pips_html(tier: str) -> str:
    """Render confidence as a monochrome 3-pip meter (no colored badge).

    High = 3 filled, Medium = 2, Low = 1 — filled pips in --ink, empty in
    --rule, precisely aligned in the monospace data column.
    """
    n = {"High": 3, "Medium": 2, "Low": 1}.get(tier, 0)
    return "".join(
        f'<span class="pip {"on" if i < n else "off"}">●</span>' for i in range(3)
    )


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


# ── Sidebar: the "guest check" ───────────────────────────────────────
st.sidebar.markdown('<div class="kicker">Order Context</div>', unsafe_allow_html=True)

cities = fetch_cities()
cuisines = fetch_cuisines()

city = st.sidebar.selectbox("City", cities, key="city_select")
cuisine = st.sidebar.selectbox("Cuisine", cuisines, key="cuisine_select")
meal_time = st.sidebar.selectbox("Meal time", ["Breakfast", "Lunch", "Snacks", "Dinner"], key="meal_select")

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

if st.sidebar.button("Clear cart & start over", use_container_width=True):
    st.session_state.cart_items = []
    st.session_state.cart_details = []
    st.session_state.recommendations = []
    st.session_state.rec_method = ""
    st.session_state.rec_latency = 0.0
    st.rerun()

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="kicker">Guest Check &middot; {len(st.session_state.cart_items)} '
    f'{"item" if len(st.session_state.cart_items)==1 else "items"}</div>',
    unsafe_allow_html=True,
)
if st.session_state.cart_details:
    cart_total = 0
    for item in st.session_state.cart_details:
        st.sidebar.markdown(
            f'<div class="check-line">'
            f'<span class="check-name">{item["item_name"]}</span>'
            f'<span class="check-cat">{item["category"]}</span>'
            f'<span class="check-price">&#8377;{item["price"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        cart_total += item["price"]
    st.sidebar.markdown(
        f'<div class="check-total"><span class="t-lbl">Subtotal</span>'
        f'<span class="t-val">&#8377;{cart_total}</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="note">Cart is empty — add items from the menu to see recommendations.</div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown('<div class="kicker">Model</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<ul class="spec">'
    '<li>LightGBM LambdaRank (learning-to-rank)</li>'
    '<li>Inference on every cart change</li>'
    '<li>Conditioned on cart + city + cuisine + meal time</li>'
    '<li>Confidence tiers calibrated from held-out scores</li>'
    '</ul>',
    unsafe_allow_html=True,
)


# ── Main Area Layout ─────────────────────────────────────────────────
col_menu, col_recs = st.columns([3, 2], gap="large")

# ── Left Column: Menu ────────────────────────────────────────────────
with col_menu:
    st.markdown(
        f'<div class="kicker">Menu</div><div class="section-head">{cuisine}</div>',
        unsafe_allow_html=True,
    )

    menu_items = fetch_items_for_cuisine(cuisine)

    if not menu_items:
        st.markdown(
            '<div class="note">Could not load the menu. The backend may be waking from '
            'idle (free-tier cold start) — give it a moment and rerun.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Group items by category for menu-style display
        categories_order = ["Main", "Side", "Drink", "Dessert"]
        items_by_cat = {}
        for item in menu_items:
            cat = item.get("category", "Other")
            items_by_cat.setdefault(cat, []).append(item)

        for cat in categories_order:
            cat_items = items_by_cat.get(cat, [])
            if not cat_items:
                continue

            st.markdown(f'<div class="course-head">{cat}s</div>', unsafe_allow_html=True)
            for item in cat_items:
                item_id = item["item_id"]
                already_in_cart = item_id in st.session_state.cart_items

                c_row, c_btn = st.columns([6, 1], gap="small")
                with c_row:
                    name_cls = "mi-name added" if already_in_cart else "mi-name"
                    st.markdown(
                        f'<div class="menu-row">'
                        f'<span class="{name_cls}">{item["item_name"]}</span>'
                        f'<span class="leader"></span>'
                        f'<span class="mi-price">&#8377;{item["price"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if already_in_cart:
                        st.markdown('<div class="tag">in cart</div>', unsafe_allow_html=True)
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

# ── Right Column: Recommendations (the receipt) ──────────────────────
with col_recs:
    st.markdown(
        '<div class="kicker">Recommended Add-ons</div>'
        '<div class="section-head">What to add next</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.cart_items:
        st.markdown(
            '<div class="note">Add your first item from the menu to see recommendations '
            'update live as the cart grows.</div>',
            unsafe_allow_html=True,
        )
    elif not st.session_state.recommendations:
        # Cart has items but no recommendations fetched yet (e.g. page reload)
        recs, method, latency = fetch_recommendations(
            city, cuisine, meal_time,
            st.session_state.cart_items,
        )
        st.session_state.recommendations = recs
        st.session_state.rec_method = method
        st.session_state.rec_latency = latency

    if st.session_state.recommendations:
        source = "model ranking" if st.session_state.rec_method == "model" else "popularity fallback"
        st.markdown(
            f'<div class="mono" style="color:var(--muted);font-size:.72rem;'
            f'letter-spacing:.04em;margin:.1rem 0 .4rem;">'
            f'{source} &middot; {st.session_state.rec_latency} ms round-trip</div>',
            unsafe_allow_html=True,
        )

        for idx, rec in enumerate(st.session_state.recommendations[:5], start=1):
            score = rec.get("score")
            item_id = rec["item_id"]
            already_in_cart = item_id in st.session_state.cart_items
            is_top = idx == 1 and score is not None

            stamp = '<span class="stamp-tag">Top pick</span>' if is_top else ""
            price = rec.get("price", 0)
            row1 = (
                f'<div class="tl-row1">'
                f'<span class="rank">{idx:02d}</span>'
                f'<span class="tl-name">{rec["item_name"]}</span>'
                f'{stamp}'
                f'<span class="tl-spacer"></span>'
                f'<span class="tl-price">&#8377;{price}</span>'
                f'</div>'
            )
            if score is not None:
                # Confidence tier is computed SERVER-SIDE from the calibrated
                # score distribution (terciles of held-out top-1 scores) and
                # returned by the API. We just render it as a quiet mono data
                # column — no thresholds hardcoded here, so nothing drifts if
                # the model is retrained. LGBMRanker scores aren't
                # probabilities, so a fixed 0-1 cutoff in the UI would be
                # meaningless; this is the fix for that.
                tier = rec.get("confidence", "")
                row2 = (
                    f'<div class="tl-row2">'
                    f'<span>score <span class="val">{score:.4f}</span></span>'
                    f'<span>{pips_html(tier)} <span class="val tier">{tier}</span></span>'
                    f'</div>'
                )
            else:
                reason = rec.get("fallback_reason", "n/a")
                row2 = f'<div class="tl-row2"><span>fallback &middot; {reason}</span></div>'

            c_line, c_btn = st.columns([6, 1], gap="small")
            with c_line:
                cls = "ticket top" if is_top else "ticket"
                st.markdown(f'<div class="{cls}">{row1}{row2}</div>', unsafe_allow_html=True)
            with c_btn:
                if already_in_cart:
                    st.markdown('<div class="tag">in cart</div>', unsafe_allow_html=True)
                else:
                    # The top pick's action carries the one accent (primary).
                    if st.button("Add", key=f"rec_{item_id}", use_container_width=True,
                                 type="primary" if is_top else "secondary"):
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

    elif st.session_state.cart_items and not st.session_state.recommendations:
        st.markdown(
            '<div class="note">Nothing left to add in this cuisine — the cart looks complete.</div>',
            unsafe_allow_html=True,
        )


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    f'<div class="mono" style="color:var(--muted);font-size:.7rem;letter-spacing:.08em;">'
    f'CartIQ &middot; learning-to-rank cart add-ons &middot; {datetime.now().year}</div>',
    unsafe_allow_html=True,
)
