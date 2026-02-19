import logging
import os
import re
import hashlib
import random
from urllib.parse import quote_plus
from datetime import datetime, timedelta, timezone
from statistics import mean
import xml.etree.ElementTree as ET
import requests

try:
    import nltk
except ImportError:
    nltk = None
from dotenv import load_dotenv
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None
from sqlalchemy.orm import Session

from database import Market, ProcessedSubmission, SessionLocal, SocialSignal
from http_utils import request_with_retry, jitter_sleep

NEWS_SOURCE_WEIGHTS = {
    "Reuters": 3.0,
    "Associated Press": 3.0,
    "AP News": 3.0,
    "Bloomberg": 2.5,
    "Financial Times": 2.5,
    "Wall Street Journal": 2.5,
    "BBC": 2.0,
    "CoinDesk": 1.8,
    "Cointelegraph": 1.6,
    "Yahoo": 1.2,
    "Bluesky": 1.1,
    "Unknown": 1.0,
}

GOOGLE_NEWS_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)
BLUESKY_SEARCH_API_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"

_GOOGLE_TRENDS_COOLDOWN_UNTIL = None
_GOOGLE_TRENDS_DISABLED_UNTIL = None
_BLUESKY_DISABLED_UNTIL = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "Too Many Requests" in text


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(min_value, int(raw))
    except ValueError:
        return default


def _get_logger(logger: logging.Logger = None) -> logging.Logger:
    if logger:
        return logger
    return logging.getLogger("social")


def clean_market_query(question: str) -> str:
    if not question:
        return ""
    query = question.strip()
    query = re.sub(r"^\s*will\s+", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+by\s+[^?]+", "", query, flags=re.IGNORECASE)
    query = query.replace("?", "").strip()
    return query


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_weight(source_name: str) -> float:
    if not source_name:
        return NEWS_SOURCE_WEIGHTS["Unknown"]
    for known, weight in NEWS_SOURCE_WEIGHTS.items():
        if known.lower() in source_name.lower():
            return weight
    return NEWS_SOURCE_WEIGHTS["Unknown"]


def _source_from_title(title: str) -> str:
    if not title:
        return "Unknown"
    if " - " in title:
        return title.split(" - ")[-1].strip()
    return "Unknown"


def _dedup_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256((value or "").encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _parse_google_news_rss(query: str, limit: int = 25):
    url = GOOGLE_NEWS_RSS_TEMPLATE.format(query=quote_plus(query))
    response = request_with_retry("GET", url, timeout=20, max_retries=5)
    root = ET.fromstring(response.text)

    items = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = _source_from_title(title)
        items.append(
            {
                "id": _dedup_id("rss", link or title),
                "title": title,
                "text": title,
                "source": source,
                "weight": _source_weight(source),
                "author_name": source,
                "author_quality": _source_weight(source) * 1000.0,
                "published_at": pub_date,
            }
        )
    return items


def _bluesky_query(query: str) -> str:
    text = re.sub(r"\s+", " ", query or "").strip()
    max_chars = _env_int("BLUESKY_QUERY_MAX_CHARS", 120, min_value=20)
    return text[:max_chars]


def _bluesky_author_quality(post: dict) -> float:
    like_count = _safe_float(post.get("likeCount"))
    repost_count = _safe_float(post.get("repostCount"))
    reply_count = _safe_float(post.get("replyCount"))
    quote_count = _safe_float(post.get("quoteCount"))
    # Keep this bounded so one viral post does not dominate the market score.
    return max(
        10.0,
        50.0 + (2.0 * repost_count) + like_count + (1.5 * reply_count) + (2.0 * quote_count),
    )


def _parse_bluesky_posts(query: str, limit: int, logger: logging.Logger):
    if not _env_bool("BLUESKY_ENABLED", True):
        return []

    q = _bluesky_query(query)
    if not q:
        return []

    params = {
        "q": q,
        "limit": max(1, min(100, limit)),
        "sort": os.getenv("BLUESKY_SEARCH_SORT", "latest"),
    }
    lang = os.getenv("BLUESKY_LANG")
    if lang:
        params["lang"] = lang

    response = request_with_retry(
        "GET",
        BLUESKY_SEARCH_API_URL,
        params=params,
        timeout=_env_int("BLUESKY_HTTP_TIMEOUT_SECONDS", 20, min_value=5),
        max_retries=_env_int("BLUESKY_HTTP_MAX_RETRIES", 4, min_value=1),
        logger=logger,
        headers={
            "Accept": "application/json",
            "User-Agent": "PolymarketSocialMonitor/1.0",
        },
    )
    payload = response.json()
    posts = payload.get("posts") or []

    items = []
    for post in posts:
        author = post.get("author") or {}
        record = post.get("record") or {}
        text = (record.get("text") or "").strip()
        if not text:
            continue
        handle = (author.get("handle") or author.get("did") or "unknown").strip()
        uri = post.get("uri") or ""
        created_at = record.get("createdAt") or post.get("indexedAt") or ""
        source = "Bluesky"
        items.append(
            {
                "id": _dedup_id("bsky", uri or f"{handle}:{created_at}:{text[:120]}"),
                "title": text[:280],
                "text": text,
                "source": source,
                "weight": _source_weight(source),
                "author_name": handle,
                "author_quality": _bluesky_author_quality(post),
                "published_at": created_at,
            }
        )
    return items


def _market_priority_score(market: Market, now_utc: datetime) -> float:
    volume_score = _safe_float(market.volume)
    momentum_score = max(0.0, _safe_float(getattr(market, "momentum", 0.0))) * 500.0

    urgency_score = 0.0
    resolution_date = getattr(market, "resolution_date", None)
    if resolution_date:
        if resolution_date.tzinfo is None:
            resolution_date = resolution_date.replace(tzinfo=timezone.utc)
        hours_to_resolution = (resolution_date - now_utc).total_seconds() / 3600.0
        if hours_to_resolution <= 0:
            urgency_score = 2000.0
        else:
            urgency_score = max(0.0, 2000.0 - (hours_to_resolution * 4.0))

    return volume_score + momentum_score + urgency_score


def _select_deep_tracked_market_ids(markets, logger: logging.Logger) -> set:
    deep_cap = _env_int("SOCIAL_DEEP_MAX_MARKETS", 50, min_value=1)
    wildcard_cap = _env_int("SOCIAL_WILDCARD_MARKETS", 10, min_value=0)

    if len(markets) <= deep_cap:
        return {m.market_id for m in markets}

    now_utc = datetime.now(timezone.utc)
    ranked = sorted(markets, key=lambda m: _market_priority_score(m, now_utc), reverse=True)
    selected = ranked[:deep_cap]
    selected_ids = {m.market_id for m in selected}

    remaining = [m for m in ranked[deep_cap:] if m.market_id not in selected_ids]
    wildcard_count = min(wildcard_cap, len(remaining))
    if wildcard_count > 0:
        wildcard_picks = random.sample(remaining, wildcard_count)
        selected_ids.update(m.market_id for m in wildcard_picks)

    logger.info(
        "Deep social tracking enabled for %d/%d markets (priority cap=%d, wildcards=%d).",
        len(selected_ids),
        len(markets),
        deep_cap,
        wildcard_count,
    )
    return selected_ids


def _google_trends_boost(query: str, logger: logging.Logger) -> float:
    if not _env_bool("GOOGLE_TRENDS_ENABLED", True):
        return 0.0

    global _GOOGLE_TRENDS_COOLDOWN_UNTIL, _GOOGLE_TRENDS_DISABLED_UNTIL
    now = datetime.now(timezone.utc)
    if _GOOGLE_TRENDS_DISABLED_UNTIL and now < _GOOGLE_TRENDS_DISABLED_UNTIL:
        return 0.0
    if _GOOGLE_TRENDS_COOLDOWN_UNTIL and now < _GOOGLE_TRENDS_COOLDOWN_UNTIL:
        return 0.0

    try:
        from pytrends.request import TrendReq
    except Exception:
        logger.warning("pytrends is unavailable; skipping Google Trends boost.")
        return 0.0

    max_attempts = max(1, int(os.getenv("GOOGLE_TRENDS_MAX_ATTEMPTS", "3")))
    cooldown_seconds = max(60, int(os.getenv("GOOGLE_TRENDS_COOLDOWN_SECONDS", "1800")))
    for attempt in range(max_attempts):
        try:
            pytrends = TrendReq(
                hl="en-US",
                tz=0,
                retries=1,
                backoff_factor=0.0,
                requests_args={
                    "headers": {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/121.0.0.0 Safari/537.36"
                        )
                    }
                },
            )
            pytrends.build_payload([query], timeframe="now 1-d", geo="US")
            df = pytrends.interest_over_time()
            if df.empty or query not in df:
                return 0.0
            series = df[query].astype(float)
            if len(series) < 2:
                return 0.0
            latest = float(series.iloc[-1])
            baseline = float(series.iloc[:-1].mean()) if len(series) > 1 else latest
            accel = max(0.0, latest - baseline)
            return accel / 10.0
        except Exception as exc:
            # pytrends/urllib3 compatibility failure: disable for a while to avoid log spam.
            if "method_whitelist" in str(exc):
                disable_minutes = max(30, int(os.getenv("GOOGLE_TRENDS_DISABLE_MINUTES", "720")))
                _GOOGLE_TRENDS_DISABLED_UNTIL = datetime.now(timezone.utc) + timedelta(minutes=disable_minutes)
                logger.warning(
                    "Google Trends disabled for %d minute(s) due to pytrends/urllib3 incompatibility: %s",
                    disable_minutes,
                    exc,
                )
                return 0.0
            is_rate_limited = _is_rate_limited_error(exc)
            last_attempt = attempt == max_attempts - 1
            if is_rate_limited and last_attempt:
                _GOOGLE_TRENDS_COOLDOWN_UNTIL = datetime.now(timezone.utc) + timedelta(
                    seconds=cooldown_seconds
                )
                cooldown_minutes = cooldown_seconds // 60
                logger.warning(
                    "Google Trends returned HTTP 429 repeatedly. "
                    "Disabling Trends boost for %d minute(s).",
                    cooldown_minutes,
                )
                return 0.0
            if last_attempt:
                logger.warning(f"Google Trends boost failed for query '{query}': {exc}")
                return 0.0

            delay = min(12.0, (2 ** attempt)) + random.uniform(0.1, 0.6)
            jitter_sleep(delay, delay + 0.8)
    return 0.0


def _load_vader(logger: logging.Logger):
    if SentimentIntensityAnalyzer is None:
        logger.warning("nltk/VADER not installed; sentiment defaults to 0.0.")
        return None
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        if nltk is None:
            logger.warning("nltk not available for vader_lexicon download; sentiment defaults to 0.0.")
            return None
        logger.info("Downloading vader_lexicon for sentiment analysis.")
        try:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as exc:
            logger.warning(f"VADER unavailable after download attempt: {exc}")
            return None


def _compute_trust_score(unique_authors: int, avg_karma: float, crossover_bonus: float) -> float:
    return unique_authors * avg_karma * crossover_bonus


def _is_organic_signal(raw_count: int, unique_author_count: int, encountered_sources: set) -> bool:
    """
    Organic proxy for non-Reddit mode:
    - Enough total mentions
    - Healthy author/source diversity
    """
    if raw_count < 3:
        return False
    diversity = (unique_author_count / raw_count) if raw_count else 0.0
    if diversity < 0.6:
        return False
    if len(encountered_sources) < 2:
        return False
    return True


def _is_http_status(exc: Exception, statuses: set[int]) -> bool:
    if not isinstance(exc, requests.RequestException):
        return False
    response = getattr(exc, "response", None)
    return response is not None and response.status_code in statuses


def run_social_monitor(db: Session = None, logger: logging.Logger = None):
    logger = _get_logger(logger)
    load_dotenv()

    created_db = False
    if db is None:
        db = SessionLocal()
        created_db = True

    analyzer = _load_vader(logger)

    markets = db.query(Market).all()
    if not markets:
        logger.info("No markets in database. Social signal run skipped.")
        if created_db:
            db.close()
        return

    processed_ids = {sid for (sid,) in db.query(ProcessedSubmission.submission_id).all()}
    logger.info(f"Starting social monitor for {len(markets)} markets.")
    deep_tracked_market_ids = _select_deep_tracked_market_ids(markets=markets, logger=logger)

    records_created = 0

    global _BLUESKY_DISABLED_UNTIL
    for market in markets:
        query = clean_market_query(market.question or "")
        if not query:
            continue

        raw_count = 0
        sentiments = []
        unique_authors = set()
        author_karmas = []
        encountered_weights = []
        encountered_sources = set()
        market_seen_ids = set()
        mentions = []

        if market.market_id in deep_tracked_market_ids:
            google_news_limit = _env_int("SOCIAL_GOOGLE_NEWS_LIMIT", 40, min_value=1)
            bluesky_limit = _env_int("SOCIAL_BLUESKY_LIMIT", 20, min_value=1)
            try:
                mentions.extend(_parse_google_news_rss(query=query, limit=google_news_limit))
            except Exception as exc:
                logger.warning(f"Google News RSS failed for market {market.market_id}: {exc}")

            if _BLUESKY_DISABLED_UNTIL and datetime.now(timezone.utc) < _BLUESKY_DISABLED_UNTIL:
                pass
            else:
                try:
                    mentions.extend(_parse_bluesky_posts(query=query, limit=bluesky_limit, logger=logger))
                except Exception as exc:
                    if _is_http_status(exc, {401, 403}):
                        disable_minutes = max(15, _env_int("BLUESKY_DISABLE_MINUTES_ON_403", 60, min_value=1))
                        _BLUESKY_DISABLED_UNTIL = datetime.now(timezone.utc) + timedelta(minutes=disable_minutes)
                        logger.warning(
                            "Bluesky API access denied (status 401/403). Disabling Bluesky fetches for %d minute(s).",
                            disable_minutes,
                        )
                    else:
                        logger.warning(f"Bluesky search failed for market {market.market_id}: {exc}")

        for mention in mentions:
            mention_id = mention["id"]
            if mention_id in processed_ids or mention_id in market_seen_ids:
                continue

            market_seen_ids.add(mention_id)
            title_text = mention.get("title", "")
            body_text = mention.get("text", "")
            compound = 0.0
            if analyzer:
                compound = analyzer.polarity_scores(f"{title_text}\n{body_text}")["compound"]

            raw_count += 1
            sentiments.append(compound)
            unique_authors.add(mention.get("author_name"))
            author_karmas.append(_safe_float(mention.get("author_quality")))
            encountered_weights.append(_safe_float(mention.get("weight"), 1.0))
            encountered_sources.add(mention.get("source", "Unknown"))

            db.add(
                ProcessedSubmission(
                    submission_id=mention_id,
                    market_id=market.market_id,
                    first_seen_at=datetime.now(timezone.utc),
                )
            )
            processed_ids.add(mention_id)

        avg_sentiment = mean(sentiments) if sentiments else 0.0
        unique_author_count = len({a for a in unique_authors if a})
        avg_karma = mean(author_karmas) if author_karmas else 0.0
        crossover_bonus = max(encountered_weights) if encountered_weights else 1.0
        trust_score = _compute_trust_score(unique_author_count, avg_karma, crossover_bonus)
        trend_boost = 0.0
        if market.market_id in deep_tracked_market_ids:
            trend_boost = _google_trends_boost(query=query, logger=logger)
        weighted_score = trust_score + trend_boost
        is_organic = _is_organic_signal(
            raw_count=raw_count,
            unique_author_count=unique_author_count,
            encountered_sources=encountered_sources,
        )

        db.add(
            SocialSignal(
                market_id=market.market_id,
                raw_count=raw_count,
                weighted_score=weighted_score,
                avg_sentiment=avg_sentiment,
                unique_authors=unique_author_count,
                is_organic=is_organic,
                timestamp=datetime.now(timezone.utc),
            )
        )
        records_created += 1

    db.commit()
    logger.info(f"Social monitor complete. Inserted {records_created} social signal rows.")

    if created_db:
        db.close()
