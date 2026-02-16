## Predictor System (Gear 4)

This gear predicts the probability of a volatility-adjusted surge over the next `6 hours`.

### Inputs
- `markets` table
- `market_snapshots` table
- `social_signals` table

### Social data source
- Primary: Google News RSS mention ingestion
- Boost: Google Trends short-term acceleration
- No Reddit credentials required for this mode

### Features used
- Social: `raw_count`, `weighted_score`, `avg_sentiment`, `unique_authors`
- Social quality: `is_organic` (source-diversity proxy)
- Social acceleration: delta from previous signal
- Market context: `prob_delta_1h`, `prob_delta_6h`, `vol_delta_1h`, `vol_delta_6h`, volume ratios
- Positioning context: distance from 50/50 and hours to resolution
- Anti-leakage normalization: rolling 7-day z-scores from past-only signal windows

### Labeling
- Dynamic surge label (volatility-adjusted): `future_move > 2.0 * rolling_std(last 48 snapshot deltas)`
- This replaces a static fixed move threshold.

### Output artifacts
- `model_weights.json`: trained logistic regression parameters and validation metrics
- `predictions_latest.json`: ranked markets with `predicted_surge_probability`
- `shadow_pending.jsonl`: pending predictions waiting for horizon maturity
- `shadow_results.log`: predicted vs actual outcomes in shadow mode

### Run manually
```powershell
python signal_predictor.py --mode train
python signal_predictor.py --mode predict
python signal_predictor.py --mode cycle
```

### Scheduler integration
- Added in `backend.py` as `run_predictor()`
- Runs every 1 hour
- Retrains only when model is stale (`>= 12h old`)
