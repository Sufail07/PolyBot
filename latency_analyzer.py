import argparse
import csv
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from database import Market, MarketSnapshot, SessionLocal, SocialSignal

LATENCY_EVENTS_CSV_PATH = "latency_events.csv"
LATENCY_REPORT_PATH = "latency_report.json"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    return _safe_float(os.getenv(name, default), default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _normalize_timestamp(ts: datetime) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _latest_snapshot_before(snapshots: List[MarketSnapshot], ts: datetime) -> Optional[MarketSnapshot]:
    ts = _normalize_timestamp(ts)
    prior = [s for s in snapshots if _normalize_timestamp(s.timestamp) <= ts]
    if not prior:
        return None
    return max(prior, key=lambda s: _normalize_timestamp(s.timestamp))


def _liquidity_bucket(volume: float) -> str:
    if volume < 10_000:
        return "low"
    if volume < 100_000:
        return "mid"
    return "high"


def _rolling_signal_zscore(
    signals: List[SocialSignal],
    signal_ts: datetime,
    current_value: float,
    window_hours: int,
) -> float:
    window_start = signal_ts - timedelta(hours=window_hours)
    past_values = []
    for s in signals:
        ts = _normalize_timestamp(s.timestamp)
        if ts < signal_ts and ts >= window_start:
            past_values.append(_safe_float(s.weighted_score))
    if len(past_values) < 3:
        return 0.0
    avg = float(mean(past_values))
    variance = float(mean([(x - avg) ** 2 for x in past_values]))
    std = math.sqrt(max(variance, 0.0))
    if std < 1e-8:
        return 0.0
    return (_safe_float(current_value) - avg) / std


def _rolling_prob_std_before(snapshots: List[MarketSnapshot], ts: datetime, lookback_points: int) -> float:
    history = [s for s in snapshots if _normalize_timestamp(s.timestamp) <= ts]
    if len(history) < 3:
        return 0.0
    history = history[-lookback_points:]
    probs = [_safe_float(s.probability) for s in history]
    deltas = [probs[i] - probs[i - 1] for i in range(1, len(probs))]
    if len(deltas) < 2:
        return 0.0
    avg = float(mean(deltas))
    variance = float(mean([(x - avg) ** 2 for x in deltas]))
    return math.sqrt(max(variance, 0.0))


def _seconds_between(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if not a or not b:
        return None
    return (b - a).total_seconds()


def _aggregate_rows(rows: List[dict]) -> dict:
    windows = [r["time_to_50pct_adjust_minutes"] for r in rows if r.get("time_to_50pct_adjust_minutes") is not None]
    first_moves = [r["time_to_first_move_seconds"] for r in rows if r.get("time_to_first_move_seconds") is not None]
    first_move_lags = [r["first_move_lag_seconds"] for r in rows if r.get("first_move_lag_seconds") is not None]
    reversions = [1 if r.get("reverted_within_window") else 0 for r in rows]
    return {
        "count": len(rows),
        "median_time_to_50pct_adjust_minutes": float(median(windows)) if windows else None,
        "median_time_to_first_move_seconds": float(median(first_moves)) if first_moves else None,
        "median_first_move_lag_seconds": float(median(first_move_lags)) if first_move_lags else None,
        "reversion_rate": (sum(reversions) / len(reversions)) if reversions else None,
    }


def run_latency_analysis(db: Session = None, logger: logging.Logger = None) -> dict:
    logger = logger or logging.getLogger("latency")
    created_db = False
    if db is None:
        db = SessionLocal()
        created_db = True

    spike_z_threshold = _env_float("LATENCY_SPIKE_Z_THRESHOLD", 2.3)
    spike_lookback_hours = _env_int("LATENCY_SPIKE_LOOKBACK_HOURS", 72)
    move_sigma_multiplier = _env_float("LATENCY_MOVE_SIGMA_MULT", 1.0)
    min_move_abs = _env_float("LATENCY_MIN_MOVE_ABS", 0.01)
    horizon_minutes = _env_int("LATENCY_HORIZON_MINUTES", 180)
    reversion_ratio = _env_float("LATENCY_REVERSION_RATIO", 0.5)
    pre_vol_lookback_points = _env_int("LATENCY_PREVOL_LOOKBACK_POINTS", 48)

    min_required_events = _env_int("FEASIBILITY_MIN_LATENCY_EVENTS", 30)
    min_median_window_minutes = _env_float("FEASIBILITY_MIN_MEDIAN_WINDOW_MINUTES", 10.0)

    events = []

    try:
        markets = db.query(Market).all()
        for market in markets:
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
            if len(signals) < 4 or len(snapshots) < 4:
                continue

            category = market.category or "Unknown"
            volume = _safe_float(market.volume, 0.0)
            liquidity = _liquidity_bucket(volume)

            for signal in signals:
                spike_ts = _normalize_timestamp(signal.timestamp)
                z = _rolling_signal_zscore(
                    signals=signals,
                    signal_ts=spike_ts,
                    current_value=_safe_float(signal.weighted_score),
                    window_hours=spike_lookback_hours,
                )
                if z < spike_z_threshold:
                    continue

                base_snap = _latest_snapshot_before(snapshots, spike_ts)
                if not base_snap:
                    continue

                base_prob = _safe_float(base_snap.probability)
                pre_vol_std = _rolling_prob_std_before(snapshots, spike_ts, pre_vol_lookback_points)
                move_threshold = max(min_move_abs, pre_vol_std * move_sigma_multiplier)
                window_end = spike_ts + timedelta(minutes=horizon_minutes)
                window_start = spike_ts - timedelta(minutes=horizon_minutes)
                future_snaps = [
                    s for s in snapshots
                    if _normalize_timestamp(s.timestamp) > spike_ts
                    and _normalize_timestamp(s.timestamp) <= window_end
                ]
                if not future_snaps:
                    continue

                abs_moves = [abs(_safe_float(s.probability) - base_prob) for s in future_snaps]
                max_abs_move = max(abs_moves)
                full_idx = abs_moves.index(max_abs_move)
                full_snap = future_snaps[full_idx]
                full_move = _safe_float(full_snap.probability) - base_prob

                first_idx = next((i for i, move in enumerate(abs_moves) if move >= move_threshold), None)
                first_snap = future_snaps[first_idx] if first_idx is not None else None

                window_snaps = [
                    s for s in snapshots
                    if _normalize_timestamp(s.timestamp) >= window_start
                    and _normalize_timestamp(s.timestamp) <= window_end
                ]
                lag_candidates = []
                for snap in window_snaps:
                    snap_ts = _normalize_timestamp(snap.timestamp)
                    if abs(_safe_float(snap.probability) - base_prob) >= move_threshold:
                        lag_candidates.append((snap_ts - spike_ts).total_seconds())
                first_move_lag_seconds = None
                if lag_candidates:
                    lag_candidates = sorted(lag_candidates, key=lambda x: (abs(x), 0 if x < 0 else 1))
                    first_move_lag_seconds = lag_candidates[0]

                half_target = 0.5 * abs(full_move)
                half_idx = next((i for i, move in enumerate(abs_moves) if move >= half_target), None) if half_target > 0 else None
                half_snap = future_snaps[half_idx] if half_idx is not None else None

                reverted = False
                if abs(full_move) > 0:
                    direction = 1.0 if full_move >= 0 else -1.0
                    full_dir_move = direction * (_safe_float(full_snap.probability) - base_prob)
                    for snap in future_snaps[full_idx + 1:]:
                        dir_move = direction * (_safe_float(snap.probability) - base_prob)
                        retrace = full_dir_move - dir_move
                        if retrace >= reversion_ratio * abs(full_move):
                            reverted = True
                            break

                events.append(
                    {
                        "market_id": market.market_id,
                        "category": category,
                        "liquidity_bucket": liquidity,
                        "signal_time": spike_ts.isoformat(),
                        "signal_weighted_score": _safe_float(signal.weighted_score),
                        "signal_spike_z": z,
                        "base_probability": base_prob,
                        "move_threshold": move_threshold,
                        "first_move_probability": _safe_float(first_snap.probability) if first_snap else None,
                        "full_adjust_probability": _safe_float(full_snap.probability),
                        "time_to_first_move_seconds": _seconds_between(spike_ts, _normalize_timestamp(first_snap.timestamp)) if first_snap else None,
                        "first_move_lag_seconds": first_move_lag_seconds,
                        "time_to_50pct_adjust_minutes": (
                            _seconds_between(spike_ts, _normalize_timestamp(half_snap.timestamp)) / 60.0
                            if half_snap else None
                        ),
                        "time_to_full_adjust_minutes": _seconds_between(spike_ts, _normalize_timestamp(full_snap.timestamp)) / 60.0,
                        "full_adjust_move": full_move,
                        "reverted_within_window": reverted,
                    }
                )

        by_category = defaultdict(list)
        by_liquidity = defaultdict(list)
        for row in events:
            by_category[row["category"]].append(row)
            by_liquidity[row["liquidity_bucket"]].append(row)

        overall = _aggregate_rows(events)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "params": {
                "spike_z_threshold": spike_z_threshold,
                "spike_lookback_hours": spike_lookback_hours,
                "move_sigma_multiplier": move_sigma_multiplier,
                "min_move_abs": min_move_abs,
                "horizon_minutes": horizon_minutes,
                "reversion_ratio": reversion_ratio,
            },
            "overall": overall,
            "by_category": {k: _aggregate_rows(v) for k, v in by_category.items()},
            "by_liquidity_bucket": {k: _aggregate_rows(v) for k, v in by_liquidity.items()},
        }

        median_window = overall.get("median_time_to_50pct_adjust_minutes")
        gate_pass = (
            overall.get("count", 0) >= min_required_events
            and median_window is not None
            and median_window >= min_median_window_minutes
        )
        report["feasibility_gate"] = {
            "name": "latency_window_gate",
            "min_required_events": min_required_events,
            "min_median_window_minutes": min_median_window_minutes,
            "actual_event_count": overall.get("count", 0),
            "actual_median_window_minutes": median_window,
            "pass": bool(gate_pass),
        }

        with open(LATENCY_EVENTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            if events:
                writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
                writer.writeheader()
                for row in events:
                    writer.writerow(row)
            else:
                f.write("")

        with open(LATENCY_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(
            "Latency analysis complete. events=%d median_50pct_window_min=%s",
            len(events),
            f"{median_window:.2f}" if median_window is not None else "N/A",
        )
        return report
    finally:
        if created_db:
            db.close()


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Analyze social-to-price latency and reversion metrics.")
    parser.add_argument(
        "--run-once",
        action="store_true",
        default=True,
        help="Run analysis once and write report files.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [LATENCY] [%(levelname)s] - %(message)s",
    )
    logger = logging.getLogger("latency")
    run_latency_analysis(logger=logger)


if __name__ == "__main__":
    main()
