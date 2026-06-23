# What Changed, and Why — Interview Prep Notes

This document exists so you and your teammate can explain every significant change cold, in an interview, without having to re-derive the reasoning live. Read it once, then try explaining each section out loud without looking — that's the actual test of whether you're ready.

---

## 1. The core fix: de-circularizing the label generator

### The bug, precisely
The original `generate_training_data.py` computed a label probability as an explicit hand-written formula:
```
prob = 0.02 + 0.06*(cuisine_match) + 0.32*(dinner & dessert) + 0.30*(city_cuisine_match) + ...
label = Bernoulli(prob)
```
The original `phase2_feature_engineering.py` then computed features `is_dinner_dessert`, `city_cuisine_match`, `cuisine_match` using **the same predicates**, and even reconstructed the label formula's weighted sum verbatim as a feature called `contextual_score_boost`.

This means: every input the model needed to perfectly reconstruct the label was handed to it directly, by a feature engineer who had already read the label-generating code. The model achieving near-perfect AUC under this setup proves gradient boosting can memorize an injective lookup table — it proves nothing about behavioral pattern discovery.

### The fix
`generate_training_data.py` now generates labels from:
1. **A hidden affinity matrix** between (meal_time, candidate_category) and between (cuisine, candidate_cuisine) — drawn randomly once per dataset generation, not chosen to match feature names.
2. **A per-cart latent "impulsivity" trait** sampled from a Beta distribution — never written to any output column, so no downstream feature can read it directly.
3. **Nonlinear combination** — the latent trait enters squared, and price ratio enters through a sine term, so no linear combination of visible features can perfectly invert the label.
4. **Irreducible Gaussian noise** added to the logit before the Bernoulli draw — this caps achievable AUC below 1.0, which is the expected and correct behavior for a benchmark modeling real-world-style uncertainty.

`phase2_feature_engineering.py` was rewritten to only use information a real serving system would actually have (cart contents, candidate metadata, coarse context) — it has no access to, and makes no attempt to reverse-engineer, the hidden affinity matrices.

### How to talk about it in an interview
**If asked "is this real data":** No — be upfront. "This is synthetic data. What I focused on was making sure the methodology — feature engineering, model selection, evaluation — was exercised honestly, by designing the generator so the model has to discover structure it wasn't told about, with an irreducible noise floor, the same way it would on real behavioral data."

**If asked "how do you know your model learned something real and not just memorized your formula":** Point to the regression test `test_labels_are_not_deterministic_given_features_alone` in `tests/test_pipeline.py` — it proves the label isn't a deterministic function of anything a feature could see. Also point to the AUC: ~0.65 (RandomForest sanity check), nowhere near 0.99, which is exactly what you'd expect from a noisy, partially-learnable signal rather than a memorized lookup.

**If asked "what would you do differently with real data":** "I'd replace the synthetic generator with real order logs, but the rest of the pipeline — group-aware splitting, ranking objective, evaluation against baselines — doesn't change. The label source is the only synthetic part; everything downstream of it is methodology that transfers directly."

---

## 2. Classifier → Ranker

### The bug
`train_lightgbm.py` trained `LGBMClassifier(objective="binary")` and then sorted candidates by the classifier's raw score. That's a **pointwise** objective (each row scored independently) repurposed as a ranking signal, after the fact. It's not what the loss function was optimizing for. Also: in `backend/api.py`, `model.predict(X)` on a classifier returns hard 0/1 labels, not probabilities — `predict_proba` was needed but not used, meaning the live API was likely sorting candidates by a near-degenerate 0/1 signal.

### The fix
`model_training/train_ranker.py` trains `LGBMRanker(objective="lambdarank", metric="ndcg", eval_at=[3])`, with training data grouped by `cart_id` (using LightGBM's `group` parameter — number of candidate rows per cart, in matching row order). This optimizes directly for getting the *order* of candidates within a cart right, which is the actual task.

`backend/api.py` calls `model.predict(X)` on the ranker, which returns a continuous relevance score — correct for sorting.

### How to talk about it
"Pointwise classification asks 'is this candidate good in isolation'. Listwise ranking with lambdarank asks 'did I get the relative order right within this cart', which is what Precision@3/NDCG@3 actually measure. The trained ranker gets Precision@3 0.228 / NDCG@3 0.239 on the held-out test set — see `metrics.json`. How much of that is the ranker versus context, and what that lift number really means, is the next section."

---

## 2b. The three baselines, and why the "+68% over popularity" number must never stand alone

### The trap
The ranker beats the popularity baseline by **+68% Precision@3**. Quoted alone, that sounds like cart modelling doing heavy lifting. It isn't, and a sharp interviewer will catch it in one step: the popularity baseline is only **~4% above random** (Precision@3 0.135 vs 0.130). A "+68% lift over a baseline that itself barely beats chance" is mostly a statement about how weak that baseline is.

### The fix: a third, context-aware baseline
`train_ranker.py` now evaluates **three** baselines spanning a context gradient:
1. **random** (pure chance),
2. **popularity** — `candidate_popularity_score` only, **zero context**,
3. **context-aware** — each item's empirical "added-next" rate within its `(meal_time, candidate_category)` bucket, fit on train, scored on test. Adds **meal_time context but no cart awareness** — it sits exactly between popularity and the ranker.

The context-aware baseline scores **Precision@3 0.221 (+69% over random on its own)** — i.e. almost all of the achievable lift comes from meal_time context alone. The full ranker (0.228) adds a further **~3%** over it.

### How to talk about it
"The +68% over popularity is real, but I report it next to the fact that popularity is only ~4% above random. The honest decomposition: **~69% of the lift over random is meal_time/category context — which any context-aware baseline recovers — and cart-awareness specifically adds a measurable but modest ~3% on top of a context-aware baseline.** I always state that ~3% as 'over a context-aware baseline', never as an unqualified model effect, because the model also earns the meal_time-context credit the bare popularity number misses. I built the third baseline specifically so I'm not quoting a lift number that's really just measuring a weak baseline — it's a smaller, honest claim, and it's the one I can defend." This is consistent with the generator: the dominant within-cart signal is `(meal_time, category)` affinity; the only genuinely cart-dependent term is a smaller nonlinear `sin(price_ratio)` on `cart_total_price` — which is exactly what the top-2 features (`cart_total_price`, `price_ratio`) key off.

---

## 3. Leakage-safe train/test split

### The bug
`train_lightgbm.py` used `sklearn.train_test_split` directly on individual (cart, candidate) rows, stratified by label. Since each cart contributes ~20–30 candidate rows, this allowed some candidates from the same cart to land in train and others in test — the model could partially "see" a cart's context during training and then get evaluated on a held-out slice of the *same* cart. That inflates reported metrics relative to genuine generalization to unseen carts.

### The fix
`train_ranker.py` uses `GroupShuffleSplit` on `cart_id`, so an entire cart (all its candidate rows) is either fully in train or fully in test. The script asserts zero `cart_id` overlap between splits before proceeding — this assertion is the proof, not just a claim.

### How to talk about it
"I split at the cart level, not the row level, and added an explicit assertion that there's zero cart_id overlap between train and test before training even starts — that's the difference between measuring generalization to new carts versus partially measuring memorization of carts the model already partly saw."

---

## 4. Dropped four structurally-constant features (real EDA finding, not an oversight)

During rebuild, an automated constant-column check on the new feature matrix caught:
- `cuisine_match` — always 1 (candidates are only ever drawn from the cart's own cuisine, by construction of the candidate pool)
- `number_of_mains` / `has_main` — always 1 / always true (every cart always contains exactly one Main, added first)
- `number_of_desserts` / `has_dessert` — always 0 (dessert is always the last item added when present, and snapshot-slicing never includes an order's final item inside `cart_items`)

These were removed from the feature set. This is a legitimate, explainable EDA finding rooted in how the synthetic order-building sequence works — not a bug to hide. **If asked "why isn't X in your feature set" in a review, this is the honest answer**, and it's a stronger answer than not having noticed at all.

---

## 5. Train/serve consistency

The original had feature logic duplicated between `phase2_feature_engineering.py` (training) and `backend/api.py` (serving) — two independently maintained implementations with real risk of drifting apart (train/serve skew). The rewrite has exactly one feature-computation function, `build_features()`, defined once in `phase2_feature_engineering.py` and imported directly by `backend/api.py`. There is no second copy to drift.

---

## 6. Repo hygiene fixes

- `backend/phases/phase1.py` through `phase8.py` — an entirely separate, disconnected lineage of exploratory prototypes (different mock menus per file, `phase5.py` was literally an unrelated SQLite "quote of the day" app) — removed from the active pipeline. If you want to preserve them for the development-iteration story, keep them in a clearly labeled `research/` folder with a note that they're exploratory dead-ends, not part of the running system.
- `data_generation/main.py` (imported a nonexistent `generate_all` function) and `debug_context_check.py` (referenced undefined `FEATURES`/`ranker` at the bottom) — both broken, not carried into the rebuild as-is. Use `run_pipeline.py` at the project root instead.
- `backend/main.py` chained the eight disconnected phase files together — also not carried forward, since those files don't talk to the real pipeline.
- Model serialization standardized on `joblib` everywhere (was `pickle`-saved, `joblib`-loaded — inconsistent).
- Cuisine list unified to one canonical source (`generate_items.py:CUISINES`), imported everywhere else. The original had `"Pizza"` in the item generator but `"Italian"` in the orders generator, city-bias maps, and frontend dropdown — meaning selecting Italian cuisine in the UI always returned zero candidates.
- `requirements.txt` was a full `pip freeze` dump including unused packages (`torch`, `transformers`, `gradio`, `jupyterlab`, `nltk`, etc.) in UTF-16 encoding. Replaced with a scoped, pinned, UTF-8 list of only what's actually imported.
- Frontend cart-item picker showed placeholder labels (`Item_1`...`Item_150`) unrelated to the real catalog's naming scheme; now shows real item names, category, and price, scoped to the selected cuisine.
- Frontend's per-recommendation "explanations" were a hardcoded dict of five fixed strings keyed by rank position (so rank #1 always said the same sentence regardless of which item was actually recommended). Removed rather than left in place misleadingly — true feature-attribution explanations (SHAP) are listed as a clear next step in `README.md`, not faked.
- **Confidence tiers are calibrated to the real score distribution, not an assumed 0-1 range.** The frontend originally labelled recommendations High/Medium/Low using hardcoded cutoffs (`score >= 0.6` / `>= 0.3`). LGBMRanker `lambdarank` scores are **not probabilities** and have no fixed zero point — on the held-out test set they're ~75% negative with a median of −0.28, so under those cutoffs ~99.7% of candidates would read "Low" and "High" was essentially unreachable. The tiers are now derived from the **terciles of the held-out per-cart top-1 scores** (`Low < 0.219 ≤ Medium < 0.426 ≤ High`), computed in `train_ranker.py`, persisted to `metrics.json`, and applied **server-side** in `/full_pipeline` so the threshold lives in exactly one place (no frontend drift on retrain). **Interview point:** this is a clean example of "the model's raw output is not a calibrated probability, so I calibrated the user-facing tier against the actual score distribution rather than assuming a range" — and it's *why* a low-signal context like Breakfast (top-1 scores averaging 0.17 vs Lunch's 0.50) correctly surfaces as "Low confidence": honest uncertainty, not a bug.

---

## 7. What is still honestly a limitation (say this if asked, don't get caught flat-footed)

- **Synthetic data.** No real consumer orders. The whole point of the rebuild was to make the *methodology* defensible despite this, not to pretend the data is real.
- **No production hardening yet** — no Docker, no CI, no observability/metrics endpoint beyond a basic `/health` check, no caching layer, no rate limiting. These are real next steps, not silently dropped scope — see the broader project roadmap for the phased plan.
- **No explainability layer** (SHAP) on recommendations yet — the API returns a relevance score, not a feature-attribution breakdown.
- **Revenue/business-impact numbers** (e.g. the original pitch's "₹16 Crore annual uplift") have been deliberately **not** carried forward as a headline claim. See `docs/RESULTS.md` for what's actually measured (Precision@3 / NDCG@3 lift over a popularity baseline, on held-out synthetic data) versus what would require real production A/B data to claim.
