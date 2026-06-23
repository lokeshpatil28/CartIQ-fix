"""
CSAO real-time inference API.

CHANGES vs. the original:
  - Feature computation is imported from data_generation/phase2_feature_
    engineering.py (build_features, FEATURE_COLUMNS) instead of being
    re-implemented here. There is now exactly one feature-computation code
    path shared by training and serving - eliminates the train/serve skew
    risk that exists whenever the same logic is maintained in two places.
  - model.predict(X) is used directly. The original used model.predict(X)
    on an LGBMClassifier, which returns hard 0/1 class labels, not scores -
    sorting candidates by a 0/1 label is close to meaningless as a ranking
    signal (most candidates tie at the same label). LGBMRanker.predict()
    returns a continuous relevance score, which is what ranking requires.
  - Config (paths, city/cuisine lists) is centralized and imported from a
    single config module instead of being re-typed in every file.
  - Basic input validation: unknown cart items, unknown city/cuisine, and
    empty-candidate-pool cases are handled explicitly rather than allowed
    to raise unhandled exceptions or return silently wrong results.
"""

from pathlib import Path
import sys
import json
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import pandas as pd
import joblib

# Structured (JSON-per-line) request logging. One line per request with
# method/path/status/latency is enough to drive latency dashboards and error
# rates once this is behind a real log aggregator, and JSON keeps it
# machine-parseable rather than a free-text string nobody can query.
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("csao.api")

sys.path.append(str(Path(__file__).resolve().parent.parent / "data_generation"))
from phase2_feature_engineering import build_features, FEATURE_COLUMNS, CITY_LIST, MEAL_MAP
from generate_items import CUISINES

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model_training" / "model.joblib"
METRICS_PATH = BASE_DIR / "model_training" / "metrics.json"
ITEMS_PATH = BASE_DIR / "data_generation" / "items.csv"

MAX_CANDIDATE_POOL = 25
TOP_N_RESULTS = 10

app = FastAPI(title="CSAO ML Inference API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Load artifacts once at startup.
# ----------------------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run model_training/train_ranker.py first."
    ) from e

items_df = pd.read_csv(ITEMS_PATH)
items_dict = items_df.set_index("item_id").to_dict(orient="index")
VALID_ITEM_IDS = set(items_dict.keys())

# Confidence tiers are CALIBRATED from the held-out score distribution at
# training time (terciles of per-cart top-1 ranker scores) and persisted to
# metrics.json - see train_ranker.py. We attach the tier server-side so the
# threshold lives in one place; LGBMRanker scores aren't probabilities, so a
# hardcoded 0-1 cutoff in the frontend would be meaningless. Fallback values
# are used only if metrics.json is somehow absent.
try:
    with open(METRICS_PATH) as f:
        _tiers = json.load(f).get("score_tiers", {})
    SCORE_LOW_MAX = float(_tiers.get("low_max", 0.22))
    SCORE_HIGH_MIN = float(_tiers.get("high_min", 0.43))
except (FileNotFoundError, ValueError):
    SCORE_LOW_MAX, SCORE_HIGH_MIN = 0.22, 0.43


def confidence_tier(score: float) -> str:
    """Map a raw ranker score to a calibrated Low/Medium/High tier."""
    if score >= SCORE_HIGH_MIN:
        return "High"
    if score >= SCORE_LOW_MAX:
        return "Medium"
    return "Low"


class CartRequest(BaseModel):
    cart_items: list[int]
    city: str
    meal_time: str
    cuisine: str

    @field_validator("city")
    @classmethod
    def city_must_be_known(cls, v):
        if v not in CITY_LIST:
            raise ValueError(f"Unknown city '{v}'. Expected one of {CITY_LIST}.")
        return v

    @field_validator("meal_time")
    @classmethod
    def meal_time_must_be_known(cls, v):
        if v not in MEAL_MAP:
            raise ValueError(f"Unknown meal_time '{v}'. Expected one of {list(MEAL_MAP)}.")
        return v

    @field_validator("cuisine")
    @classmethod
    def cuisine_must_be_known(cls, v):
        if v not in CUISINES:
            raise ValueError(f"Unknown cuisine '{v}'. Expected one of {CUISINES}.")
        return v


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Emit one structured log line per request: method, path, status, latency."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(json.dumps({
        "event": "http_request",
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": latency_ms,
    }))
    return response


@app.get("/")
def root():
    return {"message": "CSAO ML Inference API running", "model": str(MODEL_PATH.name)}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "n_items": len(items_df)}


@app.get("/cities")
def get_cities():
    """Return list of valid cities — sourced from the shared CITY_LIST."""
    return {"cities": CITY_LIST}


@app.get("/cuisines")
def get_cuisines():
    """Return list of valid cuisines — sourced from generate_items.CUISINES."""
    return {"cuisines": CUISINES}


@app.get("/items")
def get_items(cuisine: str = None):
    """Return available items, optionally filtered by cuisine."""
    if cuisine:
        filtered = items_df[items_df["cuisine"] == cuisine]
        return filtered.to_dict(orient="records")
    return items_df.to_dict(orient="records")


def generate_candidates(cart_items: list[int], cuisine: str) -> list[int]:
    filtered = items_df[items_df["cuisine"] == cuisine]
    filtered = filtered[~filtered["item_id"].isin(cart_items)]
    filtered = filtered.sort_values(by="popularity_score", ascending=False).head(MAX_CANDIDATE_POOL)
    return filtered["item_id"].tolist()


@app.post("/full_pipeline")
def full_pipeline(data: CartRequest):
    # Edge case: unknown item ids in the cart (e.g. stale frontend cache,
    # an item that's been delisted). Drop them rather than crashing, since
    # a missing item shouldn't take down the whole recommendation request.
    cart_items = [i for i in data.cart_items if i in VALID_ITEM_IDS]

    candidates = generate_candidates(cart_items, data.cuisine)

    # Edge cases per the original "robustness handling" slide, now
    # actually implemented instead of just claimed:
    if not candidates:
        # Empty cart or no candidates left in this cuisine -> trending
        # fallback (top popularity items overall, not cuisine-scoped).
        fallback = (
            items_df[~items_df["item_id"].isin(cart_items)]
            .sort_values("popularity_score", ascending=False)
            .head(TOP_N_RESULTS)
        )
        return {
            "recommendations": [
                {"item_id": int(r.item_id), "item_name": r.item_name, "category": r.category, "price": float(r.price), "score": None, "fallback_reason": "no_candidates_in_cuisine"}
                for r in fallback.itertuples()
            ]
        }

    feature_rows = [
        build_features(
            cart_items=cart_items,
            candidate_item=c,
            city=data.city,
            meal_time=data.meal_time,
            cuisine=data.cuisine,
            item_dict=items_dict,
        )
        for c in candidates
    ]
    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    scores = model.predict(X)  # LGBMRanker.predict -> continuous relevance score

    results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:TOP_N_RESULTS]

    response = [
        {
            "item_id": int(item_id),
            "item_name": items_dict[item_id]["item_name"],
            "category": items_dict[item_id]["category"],
            "price": float(items_dict[item_id]["price"]),
            "score": float(round(score, 4)),
            "confidence": confidence_tier(score),
        }
        for item_id, score in results
    ]
    return {"recommendations": response}
