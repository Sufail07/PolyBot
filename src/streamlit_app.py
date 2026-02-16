import os
import logging
from datetime import datetime, timezone

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram-check")


def _mask(value: str, keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep:
        return "*" * len(value)
    return "*" * (len(value) - keep) + value[-keep:]


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
