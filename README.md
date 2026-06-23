# CartIQ — Real-Time Cart Add-On Ranking System

A learning-to-rank system that recommends cart add-ons (sides, drinks, desserts) for a food-delivery order in real time, ranking candidates by predicted relevance rather than static rules.

> Originally built at the Zomathon hackathon as "CSAO" (Cart Smart Add-On). This version is a substantially rebuilt iteration — see [`docs/CHANGES.md`](docs/CHANGES.md) for exactly what changed and why, and [`docs/RESULTS.md`](docs/RESULTS.md) for evaluation methodology and numbers.

## Problem

Given a partial cart (items already added) plus context (city, meal time, cuisine), rank a pool of candidate add-on items by how likely they are to be added next — under a 100ms latency budget, without disrupting checkout.

## Architecture

```
data_generation/
  generate_items.py          # item catalog (price, margin, popularity, cuisine, category)
  generate_orders.py         # synthetic full orders (Main -> Side -> Drink -> Dessert)
  build_snapshots.py         # slices each order into (partial_cart -> next_item) pairs
  generate_training_data.py  # labels each (cart, candidate) pair — see docs/CHANGES.md
  phase2_feature_engineering.py  # the ONE feature-computation module, shared by training and serving

model_training/
  train_ranker.py            # LGBMRanker (lambdarank), group-aware split, metrics persisted to metrics.json

backend/
  api.py                     # FastAPI inference service, imports feature logic from data_generation

frontend/
  frontend_app.py            # Streamlit demo client

tests/
  test_pipeline.py           # automated tests covering data generation, labels, features, train/serve parity
```

## Results (held-out test set, real LGBMRanker run)

Three baselines are reported, deliberately spanning a context gradient — this three-point comparison is the whole point, so the headline lift number can never be read in isolation:

| Metric | **Ranker** (cart + city + meal_time) | Context-aware (meal_time + item, no cart) | Popularity (item only) | Random |
|---|---|---|---|---|
| Precision@3 | **0.2280** | 0.2205 | 0.1353 | 0.1301 |
| Recall@3 | **0.1753** | 0.1708 | 0.1019 | 0.0983 |
| NDCG@3 | **0.2390** | 0.2315 | 0.1411 | 0.1365 |

**How to read this:** the plain popularity baseline is only **~4% above random** (it ignores meal_time entirely). So **~69% of the lift over random is meal_time/category context** — which any context-aware baseline recovers — and **cart-awareness specifically adds a measurable but modest ~3% on top of a context-aware baseline**. The cart-aware "+3%" is always stated relative to the context-aware baseline, never as an unqualified effect. Full methodology, the per-city sensitivity finding, and feature importances are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Why this exists / what's honest about it

This is **synthetic data** — there are no real users or real orders behind it. That's stated plainly rather than glossed over. What the project demonstrates is the *methodology*: a non-circular label-generation process (see `docs/CHANGES.md`), a proper learning-to-rank setup with group-aware evaluation, and a real, deployed inference path — not a claim that this system has been validated on real consumer behavior.

## Running it

```bash
pip install -r requirements.txt

python run_pipeline.py
# runs: item/order/snapshot generation -> label generation -> feature
# engineering -> LGBMRanker training. Writes model_training/model.joblib
# and model_training/metrics.json.

# in one terminal:
uvicorn backend.api:app --reload --app-dir .

# in another terminal:
streamlit run frontend/frontend_app.py
```

Run tests with `pytest tests/`.

## API

`POST /full_pipeline`
```json
{
  "cart_items": [12, 45],
  "city": "Hyderabad",
  "meal_time": "Dinner",
  "cuisine": "Mughlai"
}
```
Returns up to 10 ranked candidates with relevance scores. See `backend/api.py` for the request/response schema and edge-case handling (empty cart, no candidates in cuisine, unknown item ids).

## What's NOT in this version (explicitly out of scope, not overlooked)

- Containerization + CI are in place: `backend/Dockerfile` and `frontend/Dockerfile`, `docker-compose.yml` for local dev (`docker compose up --build`), structured JSON request logging on the API, and a GitHub Actions workflow (`.github/workflows/ci.yml`) running the test suite on every push/PR. Live deployment URLs are the next step.
- No real-world dataset validation — see `docs/RESULTS.md` for why and what the tradeoff was.
- No SHAP-based per-recommendation explainability yet (the frontend currently shows a relevance score and confidence band, not a feature-attribution explanation).

## License

MIT (or whichever you and your teammate prefer — add a LICENSE file before making the repo public).
