# Polymarket Bot System: Working, Challenges, and Best Solutions

## 1) System Goal

This system continuously monitors Polymarket markets, enriches them with social/news signals, computes surge predictions, and sends high-conviction Telegram alerts.

It also exposes runtime health and evaluation data in a Streamlit dashboard.

---

## 2) High-Level Architecture

### Runtime Processes

1. `src/streamlit_app.py`
- Runs the dashboard UI.
- Starts the backend worker process automatically (`backend.py`).
- Shows live operations, source health, and evaluation output.
- Sends Telegram test/startup alerts through GitHub relay.

2. `backend.py`
- Long-running scheduler + immediate startup catch-up runs.
- Executes ingestion and prediction jobs at intervals.

### Core Data Store

- SQLite database: `polymarket.db`
- ORM models in `database.py`:
  - `Market`
  - `MarketSnapshot`
  - `SocialSignal`
  - `ProcessedSubmission`

### External Integrations

- Polymarket Gamma API (market fetch and snapshots)
- Google News RSS (social/news mention proxy)
- Google Trends (boost signal, when available)
- Telegram delivery via GitHub Actions relay (`repository_dispatch`)
- Optional HF dataset sync for DB persistence across restarts

---

## 3) Scheduled Jobs and What They Do

Defined in `backend.py`:

1. `run_tracker` every 30 min
- Fetches latest per-market probability/volume snapshots for tracked markets.
- Writes `MarketSnapshot`.

2. `run_sentry` every 6 hours
- Paginates active markets from Polymarket.
- Filters and inserts new markets.
- Creates baseline snapshot per new market.

3. `run_social` every 2 hours
- For each market, queries Google RSS based on cleaned question query.
- Computes aggregate features and writes `SocialSignal`.
- De-duplicates mentions via `ProcessedSubmission`.

4. `run_predictor` every 1 hour
- Retrains if stale (time-gated).
- Produces top predictions (`predictions_latest.json`).
- Updates shadow evaluation files.
- Triggers high-conviction alerts (with per-market cooldown).

Startup also runs all four once as a catch-up.

---

## 4) Prediction and Alerting Logic

### Predictor

- Uses market deltas + social features (counts, weighted score, sentiment, acceleration, z-scores, volatility context).
- Falls back to heuristic mode if model data is insufficient.
- Writes current output to `predictions_latest.json`.

### Alerting

- High-conviction alerts are sent when:
  - heuristic high-conviction flag is true, or
  - predicted probability >= threshold (default `0.80`)
- Cooldown is per market (default `6` hours), not global.

Message includes market ID, question, predicted probability, model, and UTC time.

---

## 5) Dashboard Behavior

Current UI sections:

1. Backend worker status
2. Telegram relay startup check
3. Live operations metrics (tracked markets, snapshot throughput, latest snapshot/signal times)
4. Latest evaluation (top predictions + shadow results)
5. Source health (Google RSS/social signal ingestion metrics)

This provides operational visibility and quick diagnosis of pipeline failures.

---

## 6) Key Challenges Observed

## A) Huge Polymarket pagination cost

Symptoms:
- Sentry scanned very deep offsets and large market volume.

Risk:
- High runtime cost, slower startup convergence, pressure on downstream jobs.

Current mitigation:
- Page cap (`SENTRY_MAX_PAGES`) and stop conditions.

Best solution:
1. Keep page cap configurable per environment.
2. Filter upstream by recency/liquidity where API supports.
3. Incremental ingestion strategy (checkpoint by last seen IDs/timestamps).

## B) Duplicate insert collisions (`UNIQUE market_id`)

Symptoms:
- `sqlite3.IntegrityError` during bulk insert.

Root cause:
- Duplicate IDs within large fetched set / race with existing state.

Current mitigation:
- In-memory dedupe in sentry run before insert.

Best solution:
1. Keep dedupe guard.
2. Prefer DB-level upsert patterns where possible.
3. Chunked writes + retry on conflict for resilience.

## C) Social coverage scaling problem

Symptoms:
- Large market count can make per-market RSS queries expensive and slow.

Risk:
- Slow cycles, rate-limit sensitivity, delayed signals.

Best solution:
1. Prioritized crawling (top N by liquidity/interest first).
2. Batch/round-robin coverage for long-tail markets.
3. Multi-source enrichment (not RSS-only) with source reliability weighting.

## D) Signal noise and false positives

Symptoms:
- Mentions can be unrelated, duplicated, low-quality, or market-irrelevant.
- RSS/title-level matching can overcount weak correlations.
- High mention volume does not always imply tradable price movement.

Risk:
- False alerts, reduced trust, threshold overfitting, and noisy model features.

### Hypothetical worst case: 100% noise

If all social/news input is pure noise, the model should behave as if social features are useless.
In that case, the best system behavior is:
1. detect signal collapse quickly,
2. down-weight or disable social features automatically,
3. fall back to market-only baseline logic,
4. suppress low-confidence alerts.

### Best method to overcome noise (robust strategy)

1. Reliability-gated feature fusion:
- Maintain per-source reliability scores from historical predictive value.
- Weight source contributions by rolling reliability, not static constants.
- Auto-decay a source toward zero weight when recent precision drops.

2. Regime detector + automatic fallback:
- Monitor rolling uplift from social features versus market-only baseline.
- If uplift is statistically insignificant for N windows, switch to baseline-only mode.
- Require stronger consensus before re-enabling social weight.

3. Causal/temporal validation guardrails:
- Only count mentions that precede measurable market response windows.
- Penalize post-move mentions (reactionary chatter).
- Keep strict dedup and novelty checks.

4. Alert calibration and abstention:
- Add uncertainty band and "no-trade/no-alert" zone.
- If confidence is low or feature disagreement is high, abstain.
- Prefer fewer high-quality alerts over high alert volume.

5. Shadow evaluation as control loop:
- Continuously compare:
  - market-only model,
  - social+market model,
  - heuristic baseline.
- Promote the best performer by recent out-of-sample metrics.

## E) HF dataset sync instability (404/auth)

Symptoms:
- Dataset preupload 404 for private dataset path/auth mismatch.

Root cause:
- Token/repo permission mismatch or wrong dataset repo wiring.

Current mitigation:
- Configurable `HF_DATASET_REPO_ID`, safer error handling.

Best solution:
1. Validate token scopes and repo visibility once at startup.
2. Add a startup health assertion that clearly reports auth/state.
3. Keep local DB fallback (already present) to avoid total outage.

## F) Logging formatter collisions

Symptoms:
- `KeyError: 'gear'` from third-party log records.

Root cause:
- Using root logger formatting expecting custom field.

Current mitigation:
- Dedicated `system` logger instead of custom formatting on root.

Best solution:
1. Keep app logs namespaced.
2. Avoid root formatter assumptions.
3. Add structured logging for ingestion KPIs.

## G) Streamlit + worker coupling in one container

Risk:
- Process lifecycle coupling can cause subtle restart/race behavior.

Best solution:
1. Move scheduler worker to separate process/service for production.
2. Keep Streamlit as read-only control/observability UI.
3. Use queue/event boundaries for clean separation.

---

## 7) Trustworthiness: Current State

Operational trust: improving and now functional.

Decision trust (prediction quality): not yet production-grade without sustained validation.

Why:
- System is now running end-to-end, but model quality requires longitudinal evaluation.
- Need rolling precision/recall and alert outcome tracking across real cycles.

---

## 8) Recommended Roadmap (Priority Order)

## P0 (Immediate)

1. Confirm stable HF dataset sync auth (`HF_TOKEN`, `HF_DATASET_REPO_ID`).
2. Keep `SENTRY_MAX_PAGES` conservative to avoid startup overload.
3. Ensure social job writes non-zero rows consistently.

## P1 (Near-term)

1. Add social crawl prioritization (top-liquidity first, then round-robin).
2. Add per-job latency and success/error counters in DB.
3. Add alert outcome panel (rolling precision by horizon).

## P2 (Scale / reliability)

1. Split scheduler worker from Streamlit app into separate deployment unit.
2. Move from single SQLite file toward durable managed storage (if scale grows).
3. Add replayable event logs for deterministic reprocessing.

---

## 9) Environment Variables to Keep Correct

Core:
- `GITHUB_RELAY_TOKEN`
- `GITHUB_REPO`

Scheduler tuning:
- `SENTRY_MAX_PAGES`

Dataset sync:
- `HF_DATASET_REPO_ID`
- `HF_TOKEN`

Alert behavior:
- `HIGH_CONVICTION_THRESHOLD`
- `ALERT_COOLDOWN_HOURS`

---

## 10) Bottom Line

The system is now operational for market ingestion, prediction execution, and Telegram delivery.

The biggest remaining work is not basic functionality; it is production hardening:
- efficient social coverage,
- robust dataset sync auth,
- and long-run quality measurement of prediction outputs.
