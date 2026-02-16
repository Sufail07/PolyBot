import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from sqlalchemy.orm import Session

from database import Market, MarketSnapshot, SessionLocal, SocialSignal

MODEL_PATH = "model_weights.json"
SHADOW_PENDING_PATH = "shadow_pending.jsonl"
SHADOW_RESULTS_PATH = "shadow_results.log"
ALERT_STATE_PATH = "alert_state.json"


@dataclass
class Row:
    market_id: str
    timestamp: datetime
    features: Dict[str, float]
    target: int
    future_return: float
    surge_threshold: float


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _normalize_timestamp(ts: datetime) -> datetime:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _to_epoch_hours(ts: datetime) -> float:
    return _normalize_timestamp(ts).timestamp() / 3600.0


def _nearest_snapshot(
    snapshots: List[MarketSnapshot], target_ts: datetime, max_delta_hours: float
) -> Optional[MarketSnapshot]:
    if not snapshots:
        return None
    target_hours = _to_epoch_hours(target_ts)
    best = None
    best_delta = None
    for snap in snapshots:
        snap_ts = _normalize_timestamp(snap.timestamp)
        delta = abs(_to_epoch_hours(snap_ts) - target_hours)
        if delta <= max_delta_hours and (best_delta is None or delta < best_delta):
            best = snap
            best_delta = delta
    return best


def _latest_snapshot_before(
    snapshots: List[MarketSnapshot], ts: datetime
) -> Optional[MarketSnapshot]:
    ts = _normalize_timestamp(ts)
    prior = [s for s in snapshots if _normalize_timestamp(s.timestamp) <= ts]
    if not prior:
        return None
    return max(prior, key=lambda s: _normalize_timestamp(s.timestamp))


def _latest_signal_before(signals: List[SocialSignal], ts: datetime) -> Optional[SocialSignal]:
    ts = _normalize_timestamp(ts)
    prior = [s for s in signals if _normalize_timestamp(s.timestamp) < ts]
    if not prior:
        return None
    return max(prior, key=lambda s: _normalize_timestamp(s.timestamp))


def _rolling_signal_zscore(
    signals: List[SocialSignal],
    signal_ts: datetime,
    current_value: float,
    value_getter,
    window_days: int = 7,
) -> float:
    window_start = signal_ts - timedelta(days=window_days)
    past_values = []
    for s in signals:
        ts = _normalize_timestamp(s.timestamp)
        if ts < signal_ts and ts >= window_start:
            past_values.append(_safe_float(value_getter(s)))
    if len(past_values) < 3:
        return 0.0
    mean = float(np.mean(past_values))
    std = float(np.std(past_values))
    if std < 1e-8:
        return 0.0
    return (_safe_float(current_value) - mean) / std


def _rolling_volatility_std(
    snapshots: List[MarketSnapshot], signal_ts: datetime, lookback_snapshots: int = 48
) -> float:
    history = [
        s for s in snapshots if _normalize_timestamp(s.timestamp) <= signal_ts
    ]
    if len(history) < 3:
        return 0.0
    history = history[-lookback_snapshots:]
    probs = [_safe_float(s.probability) for s in history]
    deltas = np.diff(probs)
    if len(deltas) < 2:
        return 0.0
    return float(np.std(deltas))


def _snapshot_at_or_before(snapshots: List[MarketSnapshot], ts: datetime) -> Optional[MarketSnapshot]:
    return _latest_snapshot_before(snapshots, ts)


def _build_feature_row(
    market: Market,
    all_signals_for_market: List[SocialSignal],
    signal: SocialSignal,
    prev_signal: Optional[SocialSignal],
    current_snap: MarketSnapshot,
    prior_snap_1h: Optional[MarketSnapshot],
    prior_snap_6h: Optional[MarketSnapshot],
    now: datetime,
) -> Dict[str, float]:
    signal_ts = _normalize_timestamp(signal.timestamp)
    raw_count = _safe_float(signal.raw_count)
    weighted_score = _safe_float(signal.weighted_score)
    avg_sentiment = _safe_float(signal.avg_sentiment)
    unique_authors = _safe_float(signal.unique_authors)
    is_organic = 1.0 if getattr(signal, "is_organic", False) else 0.0

    prev_weighted = _safe_float(prev_signal.weighted_score) if prev_signal else 0.0
    prev_raw_count = _safe_float(prev_signal.raw_count) if prev_signal else 0.0

    current_prob = _safe_float(current_snap.probability)
    current_volume = _safe_float(current_snap.volume)

    prob_1h_ago = _safe_float(prior_snap_1h.probability, current_prob) if prior_snap_1h else current_prob
    vol_1h_ago = _safe_float(prior_snap_1h.volume, current_volume) if prior_snap_1h else current_volume

    prob_6h_ago = _safe_float(prior_snap_6h.probability, current_prob) if prior_snap_6h else current_prob
    vol_6h_ago = _safe_float(prior_snap_6h.volume, current_volume) if prior_snap_6h else current_volume

    prob_delta_1h = current_prob - prob_1h_ago
    prob_delta_6h = current_prob - prob_6h_ago
    vol_delta_1h = current_volume - vol_1h_ago
    vol_delta_6h = current_volume - vol_6h_ago

    vol_ratio_1h = 0.0 if vol_1h_ago <= 0 else current_volume / vol_1h_ago - 1.0
    vol_ratio_6h = 0.0 if vol_6h_ago <= 0 else current_volume / vol_6h_ago - 1.0

    signal_accel = weighted_score - prev_weighted
    mention_accel = raw_count - prev_raw_count
    crossover_density = 0.0 if raw_count <= 0 else unique_authors / raw_count

    prob_distance_from_50 = abs(current_prob - 0.5)
    hours_to_resolution = 9999.0
    if market.resolution_date:
        hours_to_resolution = max(
            0.0,
            (_normalize_timestamp(market.resolution_date) - _normalize_timestamp(now)).total_seconds() / 3600.0,
        )

    weighted_z_7d = _rolling_signal_zscore(
        all_signals_for_market, signal_ts, weighted_score, lambda s: s.weighted_score, window_days=7
    )
    raw_count_z_7d = _rolling_signal_zscore(
        all_signals_for_market, signal_ts, raw_count, lambda s: s.raw_count, window_days=7
    )

    return {
        "raw_count": raw_count,
        "weighted_score_log": math.log1p(max(weighted_score, 0.0)),
        "avg_sentiment": avg_sentiment,
        "unique_authors": unique_authors,
        "signal_accel_log": math.copysign(math.log1p(abs(signal_accel)), signal_accel),
        "mention_accel": mention_accel,
        "crossover_density": crossover_density,
        "prob_delta_1h": prob_delta_1h,
        "prob_delta_6h": prob_delta_6h,
        "vol_delta_1h_log": math.copysign(math.log1p(abs(vol_delta_1h)), vol_delta_1h),
        "vol_delta_6h_log": math.copysign(math.log1p(abs(vol_delta_6h)), vol_delta_6h),
        "vol_ratio_1h": vol_ratio_1h,
        "vol_ratio_6h": vol_ratio_6h,
        "current_prob": current_prob,
        "prob_distance_from_50": prob_distance_from_50,
        "hours_to_resolution_log": math.log1p(hours_to_resolution),
        "weighted_score_z_7d": weighted_z_7d,
        "raw_count_z_7d": raw_count_z_7d,
        "is_organic": is_organic,
    }


def build_training_rows(
    db: Session,
    horizon_hours: int = 6,
    volatility_multiplier: float = 2.0,
    future_tolerance_hours: float = 2.0,
) -> List[Row]:
    markets = db.query(Market).all()
    rows: List[Row] = []

    for market in markets:
        signals = (
            db.query(SocialSignal)
            .filter(SocialSignal.market_id == market.market_id)
            .order_by(SocialSignal.timestamp.asc())
            .all()
        )
        if len(signals) < 2:
            continue

        snapshots = (
            db.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market.market_id)
            .order_by(MarketSnapshot.timestamp.asc())
            .all()
        )
        if len(snapshots) < 3:
            continue

        for signal in signals:
            signal_ts = _normalize_timestamp(signal.timestamp)
            current_snap = _latest_snapshot_before(snapshots, signal_ts)
            if current_snap is None:
                continue

            prior_1h = _nearest_snapshot(
                snapshots, signal_ts - timedelta(hours=1), max_delta_hours=1.0
            )
            prior_6h = _nearest_snapshot(
                snapshots, signal_ts - timedelta(hours=6), max_delta_hours=2.0
            )

            future_snap = _nearest_snapshot(
                snapshots,
                signal_ts + timedelta(hours=horizon_hours),
                max_delta_hours=future_tolerance_hours,
            )
            if future_snap is None:
                continue

            current_prob = _safe_float(current_snap.probability)
            future_prob = _safe_float(future_snap.probability)
            future_return = future_prob - current_prob
            rolling_std = _rolling_volatility_std(
                snapshots=snapshots, signal_ts=signal_ts, lookback_snapshots=48
            )
            if rolling_std <= 0:
                continue
            surge_threshold = rolling_std * volatility_multiplier
            target = 1 if future_return >= surge_threshold else 0

            prev_signal = _latest_signal_before(signals, signal_ts)
            features = _build_feature_row(
                market=market,
                all_signals_for_market=signals,
                signal=signal,
                prev_signal=prev_signal,
                current_snap=current_snap,
                prior_snap_1h=prior_1h,
                prior_snap_6h=prior_6h,
                now=signal_ts,
            )

            rows.append(
                Row(
                    market_id=market.market_id,
                    timestamp=signal_ts,
                    features=features,
                    target=target,
                    future_return=future_return,
                    surge_threshold=surge_threshold,
                )
            )
    return rows


def _rows_to_matrix(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not rows:
        return np.zeros((0, 0)), np.zeros((0,)), []
    feature_names = sorted(rows[0].features.keys())
    x = np.array([[r.features[name] for name in feature_names] for r in rows], dtype=float)
    y = np.array([r.target for r in rows], dtype=float)
    return x, y, feature_names


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.03,
    epochs: int = 1200,
    l2: float = 0.001,
) -> Tuple[np.ndarray, float]:
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        logits = np.dot(x, weights) + bias
        preds = _sigmoid(logits)

        error = preds - y
        dw = (np.dot(x.T, error) / n_samples) + (l2 * weights)
        db = np.sum(error) / n_samples

        weights -= lr * dw
        bias -= lr * db

    return weights, bias


def _predict_probs(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return _sigmoid(np.dot(x, weights) + bias)


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    if len(y_true) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "base_rate": 0.0}

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    base_rate = float(np.mean(y_true))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "base_rate": base_rate,
    }


def _topk_precision(y_true: np.ndarray, y_prob: np.ndarray, k: int = 20) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_idx = np.argsort(-y_prob)[:k]
    return float(np.mean(y_true[top_idx]))


def _time_split(rows: List[Row], train_ratio: float = 0.8) -> Tuple[List[Row], List[Row]]:
    if not rows:
        return [], []
    rows_sorted = sorted(rows, key=lambda r: r.timestamp)
    split_idx = max(1, int(len(rows_sorted) * train_ratio))
    return rows_sorted[:split_idx], rows_sorted[split_idx:]


def train_and_evaluate(
    db: Session,
    logger: logging.Logger,
    horizon_hours: int = 6,
    volatility_multiplier: float = 2.0,
):
    rows = build_training_rows(
        db=db,
        horizon_hours=horizon_hours,
        volatility_multiplier=volatility_multiplier,
        future_tolerance_hours=2.0,
    )
    if len(rows) < 40:
        logger.warning(f"Insufficient training rows ({len(rows)}). Need at least 40.")
        return None

    train_rows, test_rows = _time_split(rows, train_ratio=0.8)
    if len(test_rows) < 10:
        logger.warning(f"Insufficient test rows ({len(test_rows)}). Need at least 10.")
        return None

    x_train, y_train, feature_names = _rows_to_matrix(train_rows)
    x_test, y_test, _ = _rows_to_matrix(test_rows)

    mean, std = _standardize_fit(x_train)
    x_train_s = _standardize_apply(x_train, mean, std)
    x_test_s = _standardize_apply(x_test, mean, std)

    weights, bias = train_logreg(x_train_s, y_train)
    test_probs = _predict_probs(x_test_s, weights, bias)
    train_probs = _predict_probs(x_train_s, weights, bias)

    train_metrics = _classification_metrics(y_train, train_probs, threshold=0.5)
    test_metrics = _classification_metrics(y_test, test_probs, threshold=0.5)
    top20 = _topk_precision(y_test, test_probs, k=20)

    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": horizon_hours,
        "volatility_multiplier": volatility_multiplier,
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
            "top20_precision": top20,
            "rows_train": len(train_rows),
            "rows_test": len(test_rows),
        },
    }

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "Model trained. Test accuracy=%.3f precision=%.3f recall=%.3f top20_precision=%.3f",
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        top20,
    )
    return payload


def _load_model(path: str = MODEL_PATH) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: str, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _send_discord(discord_webhook_url: str, message: str, logger: logging.Logger):
    try:
        requests.post(discord_webhook_url, json={"content": message}, timeout=10).raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Discord alert failed: %s", exc)


def _send_telegram(bot_token: str, chat_id: str, message: str, logger: logging.Logger):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10).raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Telegram alert failed: %s", exc)


def _send_high_conviction_alerts(predictions: List[dict], logger: logging.Logger):
    if not predictions:
        return

    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not discord_webhook_url and not (telegram_bot_token and telegram_chat_id):
        return

    threshold = _safe_float(os.getenv("HIGH_CONVICTION_THRESHOLD", "0.80"), 0.8)
    cooldown_hours = _safe_float(os.getenv("ALERT_COOLDOWN_HOURS", "6"), 6.0)
    cooldown = timedelta(hours=cooldown_hours)

    state = _load_json(ALERT_STATE_PATH, {"last_sent": {}})
    last_sent = state.get("last_sent", {})
    now = datetime.now(timezone.utc)
    alerts_sent = 0

    for p in predictions:
        market_id = p.get("market_id")
        if not market_id:
            continue

        probability = _safe_float(p.get("predicted_surge_probability"))
        heuristic_flag = bool(p.get("high_conviction_alert", False))
        should_alert = heuristic_flag or (probability >= threshold)
        if not should_alert:
            continue

        last_sent_raw = last_sent.get(market_id)
        if last_sent_raw:
            try:
                last_dt = datetime.fromisoformat(last_sent_raw.replace("Z", "+00:00"))
                if now - last_dt < cooldown:
                    continue
            except ValueError:
                pass

        question = (p.get("question") or "").strip()
        model_name = p.get("model", "ml")
        msg = (
            f"High Conviction Alert\n"
            f"Market: {market_id}\n"
            f"Question: {question}\n"
            f"Predicted surge probability: {probability:.3f}\n"
            f"Model: {model_name}\n"
            f"Time (UTC): {now.isoformat()}"
        )

        if discord_webhook_url:
            _send_discord(discord_webhook_url, msg, logger)
        if telegram_bot_token and telegram_chat_id:
            _send_telegram(telegram_bot_token, telegram_chat_id, msg, logger)

        last_sent[market_id] = now.isoformat()
        alerts_sent += 1

    state["last_sent"] = last_sent
    _write_json(ALERT_STATE_PATH, state)
    if alerts_sent:
        logger.info("Sent %d high-conviction alert(s).", alerts_sent)


def _latest_signal_for_market(db: Session, market_id: str) -> Optional[SocialSignal]:
    return (
        db.query(SocialSignal)
        .filter(SocialSignal.market_id == market_id)
        .order_by(SocialSignal.timestamp.desc())
        .first()
    )


def _latest_snapshots_for_market(db: Session, market_id: str, limit: int = 20) -> List[MarketSnapshot]:
    return (
        db.query(MarketSnapshot)
        .filter(MarketSnapshot.market_id == market_id)
        .order_by(MarketSnapshot.timestamp.desc())
        .limit(limit)
        .all()[::-1]
    )


def predict_current_markets(db: Session, logger: logging.Logger, top_n: int = 20):
    model = _load_model()
    if not model:
        logger.warning("No model file found. Falling back to heuristic ranking.")
        return predict_current_markets_heuristic(db=db, logger=logger, top_n=top_n)

    feature_names = model["feature_names"]
    mean = np.array(model["mean"], dtype=float)
    std = np.array(model["std"], dtype=float)
    weights = np.array(model["weights"], dtype=float)
    bias = float(model["bias"])

    scored = []
    now = datetime.now(timezone.utc)
    markets = db.query(Market).all()

    for market in markets:
        signal = _latest_signal_for_market(db, market.market_id)
        if signal is None:
            continue

        signals = (
            db.query(SocialSignal)
            .filter(SocialSignal.market_id == market.market_id)
            .order_by(SocialSignal.timestamp.asc())
            .all()
        )
        prev_signal = _latest_signal_before(signals, _normalize_timestamp(signal.timestamp))

        snaps = _latest_snapshots_for_market(db, market.market_id, limit=60)
        current_snap = _latest_snapshot_before(snaps, _normalize_timestamp(signal.timestamp))
        if current_snap is None:
            continue

        prior_1h = _nearest_snapshot(
            snaps, _normalize_timestamp(signal.timestamp) - timedelta(hours=1), max_delta_hours=1.0
        )
        prior_6h = _nearest_snapshot(
            snaps, _normalize_timestamp(signal.timestamp) - timedelta(hours=6), max_delta_hours=2.0
        )

        features = _build_feature_row(
            market=market,
            all_signals_for_market=signals,
            signal=signal,
            prev_signal=prev_signal,
            current_snap=current_snap,
            prior_snap_1h=prior_1h,
            prior_snap_6h=prior_6h,
            now=now,
        )
        x = np.array([features.get(name, 0.0) for name in feature_names], dtype=float)
        x = (x - mean) / std
        prob = float(_sigmoid(np.dot(x, weights) + bias))

        prediction_time = datetime.now(timezone.utc).isoformat()
        scored.append(
            {
                "market_id": market.market_id,
                "question": market.question,
                "predicted_surge_probability": prob,
                "latest_signal_time": _normalize_timestamp(signal.timestamp).isoformat(),
                "latest_raw_count": int(signal.raw_count or 0),
                "latest_weighted_score": float(signal.weighted_score or 0.0),
                "prediction_time": prediction_time,
                "base_probability": float(current_snap.probability or 0.0),
            }
        )

    scored = sorted(scored, key=lambda x: x["predicted_surge_probability"], reverse=True)
    top = scored[:top_n]

    with open("predictions_latest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "top_predictions": top,
                "model_horizon_hours": model.get("horizon_hours"),
                "model_volatility_multiplier": model.get("volatility_multiplier"),
            },
            f,
            indent=2,
        )

    logger.info("Generated %d market predictions. Top candidate: %s", len(top), top[0]["market_id"] if top else "N/A")
    return top


def _heuristic_alert_for_market(
    market: Market, signals: List[SocialSignal], snapshots: List[MarketSnapshot]
) -> Tuple[float, bool]:
    if not signals or not snapshots:
        return 0.0, False
    latest_signal = signals[-1]
    signal_ts = _normalize_timestamp(latest_signal.timestamp)
    latest_score = _safe_float(latest_signal.weighted_score)

    recent_signals = [
        s for s in signals
        if _normalize_timestamp(s.timestamp) < signal_ts
        and _normalize_timestamp(s.timestamp) >= signal_ts - timedelta(hours=6)
    ]
    avg_6h = float(np.mean([_safe_float(s.weighted_score) for s in recent_signals])) if recent_signals else 0.0
    accel_ratio = ((latest_score - avg_6h) / avg_6h) if avg_6h > 0 else 0.0

    snap_now = _snapshot_at_or_before(snapshots, signal_ts)
    snap_1h = _nearest_snapshot(snapshots, signal_ts - timedelta(hours=1), max_delta_hours=1.0)
    snap_2h = _nearest_snapshot(snapshots, signal_ts - timedelta(hours=2), max_delta_hours=1.0)
    if not snap_now:
        return 0.0, False

    vol_now = _safe_float(snap_now.volume)
    vol_1h = _safe_float(snap_1h.volume, vol_now) if snap_1h else vol_now
    volume_1h = max(0.0, vol_now - vol_1h)

    prob_now = _safe_float(snap_now.probability)
    prob_2h = _safe_float(snap_2h.probability, prob_now) if snap_2h else prob_now
    price_move_2h = prob_now - prob_2h
    sigma = _rolling_volatility_std(snapshots, signal_ts, lookback_snapshots=48)

    cond_a = accel_ratio > 0.5
    cond_b = volume_1h > 1000.0 and (_safe_float(market.bid_ask_spread, 1.0) < 0.02)
    cond_c = abs(price_move_2h) <= sigma if sigma > 0 else True
    high_conviction = cond_a and cond_b and cond_c and bool(getattr(latest_signal, "is_organic", False))

    score = (
        math.log1p(max(latest_score, 0.0))
        + max(0.0, accel_ratio)
        + (0.5 if getattr(latest_signal, "is_organic", False) else 0.0)
    )
    return score, high_conviction


def predict_current_markets_heuristic(db: Session, logger: logging.Logger, top_n: int = 20):
    scored = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for market in db.query(Market).all():
        signals = (
            db.query(SocialSignal)
            .filter(SocialSignal.market_id == market.market_id)
            .order_by(SocialSignal.timestamp.asc())
            .all()
        )
        snapshots = (
            db.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market.market_id)
            .order_by(MarketSnapshot.timestamp.asc())
            .all()
        )
        if not signals:
            continue

        heuristic_score, high_conviction = _heuristic_alert_for_market(
            market=market, signals=signals, snapshots=snapshots
        )
        latest_signal = signals[-1]
        latest_snap = _snapshot_at_or_before(snapshots, _normalize_timestamp(latest_signal.timestamp))

        scored.append(
            {
                "market_id": market.market_id,
                "question": market.question,
                "predicted_surge_probability": float(min(max(heuristic_score / 5.0, 0.0), 1.0)),
                "latest_signal_time": _normalize_timestamp(latest_signal.timestamp).isoformat(),
                "latest_raw_count": int(latest_signal.raw_count or 0),
                "latest_weighted_score": float(latest_signal.weighted_score or 0.0),
                "prediction_time": now_iso,
                "base_probability": float(latest_snap.probability or 0.0) if latest_snap else 0.0,
                "high_conviction_alert": high_conviction,
                "model": "heuristic",
            }
        )

    scored = sorted(scored, key=lambda x: x["predicted_surge_probability"], reverse=True)
    top = scored[:top_n]
    with open("predictions_latest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "top_predictions": top,
                "model": "heuristic",
            },
            f,
            indent=2,
        )
    logger.info("Generated %d heuristic predictions.", len(top))
    return top


def _append_shadow_pending(predictions: List[dict], horizon_hours: int):
    existing = _load_jsonl(SHADOW_PENDING_PATH)
    keys = {row.get("key") for row in existing}
    now = datetime.now(timezone.utc)

    for p in predictions:
        prediction_time = p.get("prediction_time") or now.isoformat()
        market_id = p.get("market_id")
        key = f"{market_id}|{prediction_time}"
        if key in keys:
            continue
        due = datetime.fromisoformat(prediction_time.replace("Z", "+00:00")) + timedelta(hours=horizon_hours)
        existing.append(
            {
                "key": key,
                "market_id": market_id,
                "question": p.get("question"),
                "prediction_time": prediction_time,
                "due_time": due.isoformat(),
                "predicted_prob": p.get("predicted_surge_probability", 0.0),
                "base_probability": p.get("base_probability", 0.0),
                "evaluated": False,
            }
        )
        keys.add(key)
    _write_jsonl(SHADOW_PENDING_PATH, existing)


def _evaluate_shadow_pending(db: Session, logger: logging.Logger, horizon_hours: int):
    pending = _load_jsonl(SHADOW_PENDING_PATH)
    if not pending:
        return

    now = datetime.now(timezone.utc)
    updated = []
    result_lines = []

    for row in pending:
        if row.get("evaluated"):
            updated.append(row)
            continue
        due_time = datetime.fromisoformat(row["due_time"].replace("Z", "+00:00"))
        if due_time > now:
            updated.append(row)
            continue

        market_id = row["market_id"]
        pred_time = datetime.fromisoformat(row["prediction_time"].replace("Z", "+00:00"))
        snapshots = (
            db.query(MarketSnapshot)
            .filter(MarketSnapshot.market_id == market_id)
            .order_by(MarketSnapshot.timestamp.asc())
            .all()
        )
        base_snap = _snapshot_at_or_before(snapshots, pred_time)
        future_snap = _nearest_snapshot(snapshots, due_time, max_delta_hours=2.0)
        if not base_snap or not future_snap:
            updated.append(row)
            continue

        base_prob = _safe_float(base_snap.probability)
        future_prob = _safe_float(future_snap.probability)
        actual_move = future_prob - base_prob
        predicted_prob = _safe_float(row.get("predicted_prob"))
        predicted_label = 1 if predicted_prob >= 0.6 else 0
        actual_label = 1 if actual_move > 0 else 0

        row["evaluated"] = True
        row["actual_move"] = actual_move
        row["predicted_label"] = predicted_label
        row["actual_label"] = actual_label
        row["evaluated_at"] = now.isoformat()
        updated.append(row)

        result_lines.append(
            (
                f"{now.isoformat()} | market={market_id} | pred_prob={predicted_prob:.3f} "
                f"| base={base_prob:.4f} | future={future_prob:.4f} | move={actual_move:.4f} "
                f"| pred_label={predicted_label} | actual_label={actual_label}"
            )
        )

    _write_jsonl(SHADOW_PENDING_PATH, updated)
    if result_lines:
        with open(SHADOW_RESULTS_PATH, "a", encoding="utf-8") as f:
            for line in result_lines:
                f.write(line + "\n")
        logger.info("Shadow mode evaluated %d matured predictions.", len(result_lines))


def run_predictor_cycle(
    db: Optional[Session] = None,
    logger: Optional[logging.Logger] = None,
    retrain_every_hours: int = 12,
):
    logger = logger or logging.getLogger("predictor")
    own_db = False
    if db is None:
        db = SessionLocal()
        own_db = True

    try:
        model = _load_model()
        horizon_hours = int(model.get("horizon_hours", 6)) if model else 6
        should_train = True
        if model and model.get("trained_at"):
            trained_at = datetime.fromisoformat(model["trained_at"].replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600.0
            should_train = age_hours >= retrain_every_hours

        if should_train:
            train_and_evaluate(db=db, logger=logger, horizon_hours=6, volatility_multiplier=2.0)

        predictions = predict_current_markets(db=db, logger=logger, top_n=20)
        _append_shadow_pending(predictions=predictions, horizon_hours=horizon_hours)
        _evaluate_shadow_pending(db=db, logger=logger, horizon_hours=horizon_hours)
        _send_high_conviction_alerts(predictions=predictions, logger=logger)
    finally:
        if own_db:
            db.close()


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train and run social+market surge predictor.")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "cycle"],
        default="cycle",
        help="train: retrain model, predict: generate predictions, cycle: retrain if stale then predict",
    )
    parser.add_argument(
        "--vol-mult",
        type=float,
        default=2.0,
        help="Volatility multiplier for dynamic surge label (default: 2.0).",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [PREDICTOR] [%(levelname)s] - %(message)s",
    )
    logger = logging.getLogger("predictor")
    db = SessionLocal()
    try:
        if args.mode == "train":
            train_and_evaluate(db=db, logger=logger, horizon_hours=6, volatility_multiplier=args.vol_mult)
        elif args.mode == "predict":
            predict_current_markets(db=db, logger=logger, top_n=20)
        else:
            run_predictor_cycle(db=db, logger=logger, retrain_every_hours=12)
    finally:
        db.close()


if __name__ == "__main__":
    main()
