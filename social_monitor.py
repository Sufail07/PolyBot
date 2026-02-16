import logging
import os
import re
import hashlib
from urllib.parse import quote_plus
from datetime import datetime, timezone
from statistics import mean
import xml.etree.ElementTree as ET

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
from http_utils import request_with_retry

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
    "Unknown": 1.0,
}

GOOGLE_NEWS_RSS_TEMPLATE = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)


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


def _google_trends_boost(query: str, logger: logging.Logger) -> float:
    try:
        from pytrends.request import TrendReq
    except Exception:
        logger.warning("pytrends is unavailable; skipping Google Trends boost.")
        return 0.0

    try:
        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload([query], timeframe="now 1-d", geo="US")
        df = pytrends.interest_over_time()
        if df.empty:
            return 0.0
        series = df[query].astype(float)
        if len(series) < 2:
            return 0.0
        latest = float(series.iloc[-1])
        baseline = float(series.iloc[:-1].mean()) if len(series) > 1 else latest
        accel = max(0.0, latest - baseline)
        return accel / 10.0
    except Exception as exc:
        logger.warning(f"Google Trends boost failed for query '{query}': {exc}")
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

    records_created = 0

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

        try:
            mentions = _parse_google_news_rss(query=query, limit=40)
        except Exception as exc:
            logger.warning(f"Google News RSS failed for market {market.market_id}: {exc}")
            mentions = []

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
