import os
import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
from sqlalchemy import desc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from database import Market, ProcessedSubmission, SessionLocal, SocialSignal
from social_monitor import _parse_google_news_rss, clean_market_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram-check")


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

st.subheader("Source Health")
source_health = get_source_health()
if not source_health.get("ok"):
    st.error(f"Source health unavailable: {source_health.get('error', 'unknown error')}")
else:
    st.json(source_health["google_rss"])
    st.json(source_health["social_signals"])
    st.caption("Recent social signal rows")
    st.dataframe(source_health["recent_signals"], use_container_width=True)

if st.button("Run Google RSS probe now"):
    ok, detail, rows = probe_google_rss(limit=5)
    if ok:
        st.success(detail)
        st.dataframe(rows, use_container_width=True)
    else:
        st.warning(detail)
