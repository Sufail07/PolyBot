import os
import logging
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
from sqlalchemy import desc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database import Market, MarketSnapshot, ProcessedSubmission, SessionLocal, SocialSignal
from social_monitor import _parse_google_news_rss, clean_market_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram-check")
PREDICTIONS_PATH = ROOT_DIR / "predictions_latest.json"
SHADOW_RESULTS_PATH = ROOT_DIR / "shadow_results.log"
LATENCY_REPORT_PATH = ROOT_DIR / "latency_report.json"
SHADOW_SUMMARY_PATH = ROOT_DIR / "shadow_summary.json"


def _mask(value: str, keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * (len(value) - keep) + value[-keep:]


@st.cache_resource
def ensure_backend_worker():
    """
    Start backend.py once per Streamlit process so ingestion runs continuously.
    """
    if os.getenv("DISABLE_BACKGROUND_BACKEND", "0") == "1":
        return None, "disabled"

    process = subprocess.Popen(
        [sys.executable, str(ROOT_DIR / "backend.py")],
        cwd=str(ROOT_DIR),
        stdout=None,
        stderr=None,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return process, "stdout/stderr (HF runtime logs)"


def _send_telegram_message(text: str) -> tuple[bool, str]:
    gh_token = os.getenv("GITHUB_RELAY_TOKEN")
    gh_repo = os.getenv("GITHUB_REPO")

    if not gh_token or not gh_repo:
        return False, "Missing GITHUB_RELAY_TOKEN or GITHUB_REPO"

    try:
        r = requests.post(
            f"https://api.github.com/repos/{gh_repo}/dispatches",
            headers={
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"event_type": "send_telegram", "client_payload": {"text": text}},
            timeout=15,
        )
        if r.status_code in (200, 204):
            return True, "Message queued via GitHub relay"
        return False, f"GitHub relay error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Exception: {e}"


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    except Exception:
        return str(value)


@st.cache_data(ttl=30)
def get_market_runtime_health() -> dict:
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        ten_min_ago = now - timedelta(minutes=10)
        one_hour_ago = now - timedelta(hours=1)

        total_markets = db.query(Market).count()
        latest_snapshot = db.query(MarketSnapshot).order_by(desc(MarketSnapshot.timestamp)).first()
        latest_signal = db.query(SocialSignal).order_by(desc(SocialSignal.timestamp)).first()
        snapshots_last_10m = db.query(MarketSnapshot).filter(MarketSnapshot.timestamp >= ten_min_ago).count()
        snapshots_last_hour = db.query(MarketSnapshot).filter(MarketSnapshot.timestamp >= one_hour_ago).count()
        signals_last_hour = db.query(SocialSignal).filter(SocialSignal.timestamp >= one_hour_ago).count()

        recent_snapshots = (
            db.query(MarketSnapshot)
            .order_by(desc(MarketSnapshot.timestamp))
            .limit(10)
            .all()
        )

        recent_rows = [
            {
                "market_id": s.market_id,
                "snapshot_utc": _fmt_dt(s.timestamp),
                "probability": float(s.probability or 0.0),
                "volume": float(s.volume or 0.0),
            }
            for s in recent_snapshots
        ]

        return {
            "ok": True,
            "total_markets_tracked": total_markets,
            "latest_snapshot_utc": _fmt_dt(getattr(latest_snapshot, "timestamp", None)),
            "latest_signal_utc": _fmt_dt(getattr(latest_signal, "timestamp", None)),
            "snapshots_last_10m": snapshots_last_10m,
            "snapshots_last_hour": snapshots_last_hour,
            "signals_last_hour": signals_last_hour,
            "recent_snapshots": recent_rows,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


@st.cache_data(ttl=30)
def get_latest_predictions() -> dict:
    if not PREDICTIONS_PATH.exists():
        return {"ok": False, "error": "predictions_latest.json not found yet"}
    try:
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        top = data.get("top_predictions", [])[:10]
        table = [
            {
                "market_id": p.get("market_id"),
                "pred_surge_prob": round(float(p.get("predicted_surge_probability", 0.0)), 4),
                "latest_weighted_score": round(float(p.get("latest_weighted_score", 0.0)), 3),
                "latest_raw_count": int(p.get("latest_raw_count", 0)),
                "base_probability": round(float(p.get("base_probability", 0.0)), 4),
                "prediction_time": p.get("prediction_time"),
            }
            for p in top
        ]
        return {
            "ok": True,
            "generated_at": data.get("generated_at", "n/a"),
            "model": data.get("model", "ml"),
            "rows": table,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data(ttl=30)
def get_shadow_evaluations(last_n: int = 20) -> list[dict]:
    if not SHADOW_RESULTS_PATH.exists():
        return []
    rows = []
    try:
        with open(SHADOW_RESULTS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines[-last_n:]:
            rows.append({"evaluation": line})
        return rows[::-1]
    except Exception:
        return []


@st.cache_data(ttl=30)
def get_latency_report() -> dict:
    if not LATENCY_REPORT_PATH.exists():
        return {"ok": False, "error": "latency_report.json not found yet"}
    try:
        with open(LATENCY_REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data(ttl=30)
def get_shadow_summary() -> dict:
    if not SHADOW_SUMMARY_PATH.exists():
        return {"ok": False, "error": "shadow_summary.json not found yet"}
    try:
        with open(SHADOW_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _segment_table(section: dict) -> list[dict]:
    if not isinstance(section, dict):
        return []
    rows = []
    for key, metrics in section.items():
        rows.append(
            {
                "segment": key,
                "count": int(metrics.get("count", 0) or 0),
                "gross_edge_mean": round(float(metrics.get("gross_edge_mean", 0.0) or 0.0), 6)
                if metrics.get("gross_edge_mean") is not None else None,
                "net_edge_mean": round(float(metrics.get("net_edge_mean", 0.0) or 0.0), 6)
                if metrics.get("net_edge_mean") is not None else None,
                "gross_hit_rate": round(float(metrics.get("gross_hit_rate", 0.0) or 0.0), 4)
                if metrics.get("gross_hit_rate") is not None else None,
                "net_hit_rate": round(float(metrics.get("net_hit_rate", 0.0) or 0.0), 4)
                if metrics.get("net_hit_rate") is not None else None,
                "net_edge_std": round(float(metrics.get("net_edge_std", 0.0) or 0.0), 6)
                if metrics.get("net_edge_std") is not None else None,
                "net_edge_p25": round(float(metrics.get("net_edge_p25", 0.0) or 0.0), 6)
                if metrics.get("net_edge_p25") is not None else None,
                "net_edge_p75": round(float(metrics.get("net_edge_p75", 0.0) or 0.0), 6)
                if metrics.get("net_edge_p75") is not None else None,
                "net_edge_worst_decile_mean": round(float(metrics.get("net_edge_worst_decile_mean", 0.0) or 0.0), 6)
                if metrics.get("net_edge_worst_decile_mean") is not None else None,
            }
        )
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


@st.cache_data(ttl=60)
def get_source_health() -> dict:
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(hours=24)

        rss_filter = ProcessedSubmission.submission_id.like("rss:%")

        total_rss_items = db.query(ProcessedSubmission).filter(rss_filter).count()
        rss_last_hour = (
            db.query(ProcessedSubmission)
            .filter(rss_filter, ProcessedSubmission.first_seen_at >= one_hour_ago)
            .count()
        )
        rss_last_day = (
            db.query(ProcessedSubmission)
            .filter(rss_filter, ProcessedSubmission.first_seen_at >= one_day_ago)
            .count()
        )
        last_rss_item = (
            db.query(ProcessedSubmission)
            .filter(rss_filter)
            .order_by(desc(ProcessedSubmission.first_seen_at))
            .first()
        )

        signals_last_hour = db.query(SocialSignal).filter(SocialSignal.timestamp >= one_hour_ago).count()
        signals_with_mentions_last_hour = (
            db.query(SocialSignal)
            .filter(SocialSignal.timestamp >= one_hour_ago, SocialSignal.raw_count > 0)
            .count()
        )
        latest_signal = db.query(SocialSignal).order_by(desc(SocialSignal.timestamp)).first()

        recent_signal_rows = (
            db.query(SocialSignal)
            .order_by(desc(SocialSignal.timestamp))
            .limit(8)
            .all()
        )
        recent_signals = [
            {
                "market_id": row.market_id,
                "ts_utc": _fmt_dt(row.timestamp),
                "raw_count": int(row.raw_count or 0),
                "weighted_score": float(row.weighted_score or 0.0),
                "organic": bool(getattr(row, "is_organic", False)),
            }
            for row in recent_signal_rows
        ]

        return {
            "ok": True,
            "google_rss": {
                "last_fetch_utc": _fmt_dt(getattr(last_rss_item, "first_seen_at", None)),
                "fetch_status": "ok" if last_rss_item else "no_data",
                "items_fetched_total": total_rss_items,
                "items_fetched_last_hour": rss_last_hour,
                "items_fetched_last_24h": rss_last_day,
                "new_items_saved_last_hour": rss_last_hour,
                "new_items_saved_last_24h": rss_last_day,
                "last_error": "n/a (not persisted in DB)",
            },
            "social_signals": {
                "last_signal_utc": _fmt_dt(getattr(latest_signal, "timestamp", None)),
                "signals_generated_last_hour": signals_last_hour,
                "signals_with_mentions_last_hour": signals_with_mentions_last_hour,
            },
            "recent_signals": recent_signals,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        db.close()


def probe_google_rss(limit: int = 5):
    db = SessionLocal()
    try:
        markets = db.query(Market).filter(Market.question.isnot(None)).limit(25).all()
        if not markets:
            return False, "No markets available for probe.", []

        question = next((m.question for m in markets if m.question), "")
        query = clean_market_query(question)
        if not query:
            return False, "Could not derive query from market question.", []

        mentions = _parse_google_news_rss(query=query, limit=limit)
        simplified = [{"title": m.get("title", ""), "source": m.get("source", "Unknown")} for m in mentions]
        return True, f"Probe query: {query}", simplified
    except Exception as e:
        return False, f"Probe failed: {e}", []
    finally:
        db.close()


@st.cache_resource
def startup_telegram_check() -> dict:
    msg = f"HF startup check OK at {datetime.now(timezone.utc).isoformat()}"
    ok, detail = _send_telegram_message(msg)
    if ok:
        logger.info("Telegram startup check passed")
        return {"ok": True, "detail": detail}
    logger.warning("Telegram startup check failed: %s", detail)
    return {"ok": False, "detail": detail}


st.set_page_config(page_title="Polymarket Bot Monitor", page_icon=":satellite:", layout="centered")
st.title("Polymarket Bot Monitor")

backend_proc, backend_log = ensure_backend_worker()
if backend_proc is None:
    st.warning("Background backend worker is disabled (DISABLE_BACKGROUND_BACKEND=1).")
elif backend_proc.poll() is None:
    st.caption(f"Backend worker: running (pid={backend_proc.pid}) | log: {backend_log}")
else:
    st.error(f"Backend worker exited (code={backend_proc.returncode}). Check {backend_log}.")

status = startup_telegram_check()
st.caption(f"Startup Telegram check: {'OK' if status['ok'] else 'FAILED'}")
if not status["ok"]:
    st.warning(status["detail"])

gh_token = os.getenv("GITHUB_RELAY_TOKEN")
gh_repo = os.getenv("GITHUB_REPO")
st.write(f"GitHub relay secret check: token={'set' if gh_token else 'missing'}, repo={gh_repo or '<missing>'}")
st.write(f"Token preview: {_mask(gh_token)}")

if st.button("Send Telegram test message"):
    ok, detail = _send_telegram_message(f"Manual test from HF at {datetime.now(timezone.utc).isoformat()}")
    if ok:
        st.success("Telegram test queued.")
    else:
        st.error(detail)

st.subheader("Live Operations")
runtime = get_market_runtime_health()
if not runtime.get("ok"):
    st.error(f"Runtime health unavailable: {runtime.get('error', 'unknown error')}")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Tracked Markets", runtime["total_markets_tracked"])
    col2.metric("Snapshots (10m)", runtime["snapshots_last_10m"])
    col3.metric("Signals (1h)", runtime["signals_last_hour"])
    st.caption(
        f"Latest snapshot: {runtime['latest_snapshot_utc']} | Latest signal: {runtime['latest_signal_utc']}"
    )
    st.caption("Recent market snapshots")
    st.dataframe(runtime["recent_snapshots"], width="stretch")

st.subheader("Latest Evaluation")
preds = get_latest_predictions()
if not preds.get("ok"):
    st.warning(preds.get("error", "Prediction output not ready yet"))
else:
    st.caption(f"Predictions generated at: {preds['generated_at']} | Model: {preds['model']}")
    st.dataframe(preds["rows"], width="stretch")
    shadow_rows = get_shadow_evaluations(last_n=20)
    st.caption("Recent shadow evaluation results")
    if shadow_rows:
        st.dataframe(shadow_rows, width="stretch")
    else:
        st.write("No shadow evaluation rows yet.")

st.subheader("Feasibility Stats")
latency = get_latency_report()
shadow_summary = get_shadow_summary()

if not latency.get("ok"):
    st.warning(latency.get("error", "Latency report unavailable"))
else:
    ldata = latency["data"]
    lgate = ldata.get("feasibility_gate", {})
    loverall = ldata.get("overall", {})
    lcol1, lcol2, lcol3, lcol4 = st.columns(4)
    lcol1.metric("Latency Gate", "PASS" if lgate.get("pass") else "FAIL")
    lcol2.metric("Latency Events", int(loverall.get("count", 0) or 0))
    lcol3.metric(
        "Median 50% Adjust (min)",
        "n/a" if loverall.get("median_time_to_50pct_adjust_minutes") is None
        else round(float(loverall.get("median_time_to_50pct_adjust_minutes")), 2),
    )
    lcol4.metric(
        "Median First Move (sec)",
        "n/a" if loverall.get("median_time_to_first_move_seconds") is None
        else round(float(loverall.get("median_time_to_first_move_seconds")), 2),
    )
    first_lag = loverall.get("median_first_move_lag_seconds")
    st.caption(
        "Signed first-move lag (sec): "
        + ("n/a" if first_lag is None else str(round(float(first_lag), 2)))
        + " (negative means price moved before social spike)"
    )
    st.caption(
        "Latency gate: "
        f"min_events={lgate.get('min_required_events')} | "
        f"min_median_window_min={lgate.get('min_median_window_minutes')} | "
        f"actual_median_window_min={lgate.get('actual_median_window_minutes')}"
    )
    st.caption("Latency by category")
    st.dataframe(_segment_table(ldata.get("by_category", {})), width="stretch")
    st.caption("Latency by liquidity bucket")
    st.dataframe(_segment_table(ldata.get("by_liquidity_bucket", {})), width="stretch")

if not shadow_summary.get("ok"):
    st.warning(shadow_summary.get("error", "Shadow summary unavailable"))
else:
    sdata = shadow_summary["data"]
    sgate = sdata.get("feasibility_gate", {})
    soverall = sdata.get("overall", {})
    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Post-Cost Gate", "PASS" if sgate.get("pass") else "FAIL")
    scol2.metric("Evaluated", int(sdata.get("count_evaluated", 0) or 0))
    scol3.metric(
        "Net Edge Mean",
        "n/a" if soverall.get("net_edge_mean") is None else round(float(soverall.get("net_edge_mean")), 6),
    )
    scol4.metric(
        "Net Hit Rate",
        "n/a" if soverall.get("net_hit_rate") is None else round(float(soverall.get("net_hit_rate")), 4),
    )
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    dcol1.metric(
        "Net Edge Std",
        "n/a" if soverall.get("net_edge_std") is None else round(float(soverall.get("net_edge_std")), 6),
    )
    dcol2.metric(
        "Net Edge P25",
        "n/a" if soverall.get("net_edge_p25") is None else round(float(soverall.get("net_edge_p25")), 6),
    )
    dcol3.metric(
        "Net Edge P75",
        "n/a" if soverall.get("net_edge_p75") is None else round(float(soverall.get("net_edge_p75")), 6),
    )
    dcol4.metric(
        "Worst Decile Mean",
        "n/a" if soverall.get("net_edge_worst_decile_mean") is None else round(float(soverall.get("net_edge_worst_decile_mean")), 6),
    )
    st.caption(
        "Post-cost gate: "
        f"min_net_edge={sgate.get('min_net_edge')} | "
        f"min_net_hit_rate={sgate.get('min_net_hit_rate')} | "
        f"min_positive_categories={sgate.get('min_positive_categories')} | "
        f"min_positive_month_ratio={sgate.get('min_positive_month_ratio')}"
    )
    st.caption("Shadow performance by category")
    st.dataframe(_segment_table(sdata.get("by_category", {})), width="stretch")
    st.caption("Shadow performance by liquidity bucket")
    st.dataframe(_segment_table(sdata.get("by_liquidity_bucket", {})), width="stretch")
    st.caption("Shadow performance by month")
    st.dataframe(_segment_table(sdata.get("by_month", {})), width="stretch")

    st.caption("Red flags")
    red_flags = []
    if first_lag is not None and float(first_lag) < 0:
        red_flags.append("A: Social follows price (median signed first-move lag < 0s).")

    positive_cats = int(sgate.get("actual_positive_categories", 0) or 0)
    positive_liq = int(sgate.get("actual_positive_liquidity_buckets", 0) or 0)
    if positive_cats <= 1 or positive_liq <= 1:
        red_flags.append(
            "B: Edge appears concentrated in a tiny slice. Treat as specialization, not generalization."
        )

    gross_edge = soverall.get("gross_edge_mean")
    net_edge = soverall.get("net_edge_mean")
    if gross_edge is not None and net_edge is not None and float(gross_edge) > 0 and float(net_edge) <= 0:
        red_flags.append("C: Edge disappears after costs (gross positive, net non-positive).")

    if red_flags:
        for item in red_flags:
            st.error(item)
    else:
        st.success("No major red flags triggered by current feasibility diagnostics.")

    policy = sdata.get("evaluation_policy", {})
    if policy.get("freeze_active"):
        st.warning(
            "Threshold freeze active: do not retune filters/thresholds during this window. "
            f"Freeze ends at {policy.get('freeze_end_at')}."
        )
    else:
        st.info(
            "Threshold freeze recommendation: keep rules fixed for a full evaluation window "
            "(default 30 days) to avoid data snooping."
        )

st.subheader("Source Health")
source_health = get_source_health()
if not source_health.get("ok"):
    st.error(f"Source health unavailable: {source_health.get('error', 'unknown error')}")
else:
    st.json(source_health["google_rss"])
    st.json(source_health["social_signals"])
    st.caption("Recent social signal rows")
    st.dataframe(source_health["recent_signals"], width="stretch")

if st.button("Run Google RSS probe now"):
    ok, detail, rows = probe_google_rss(limit=5)
    if ok:
        st.success(detail)
        st.dataframe(rows, width="stretch")
    else:
        st.warning(detail)
