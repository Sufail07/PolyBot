import logging
import time
import json
import os
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler

from database import SessionLocal, Market, MarketSnapshot, download_db, upload_db
from polymarket_scraper import fetch_markets, store_markets
from social_monitor import run_social_monitor
from signal_predictor import run_predictor_cycle
from http_utils import request_with_retry, jitter_sleep
import nltk
import sys
# This forces Python to print immediately rather than waiting
sys.stdout.reconfigure(line_buffering=True)

nltk.download('vader_lexicon')
nltk.download('stopwords')

# --- 1. Logging Setup ---
# Custom formatter to add a gear name to the logs
class GearFilter(logging.Filter):
    def __init__(self, gear=''):
        super().__init__()
        self.gear = gear

    def filter(self, record):
        record.gear = self.gear
        return True


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

# Create two loggers, one for each gear
formatter = logging.Formatter("%(asctime)s [%(gear)s] [%(levelname)s] - %(message)s")

# General file handler
file_handler = logging.FileHandler("gemini_system.log")
file_handler.setFormatter(formatter)

# Stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Tracker logger
tracker_logger = logging.getLogger('tracker')
tracker_logger.setLevel(logging.INFO)
tracker_logger.addFilter(GearFilter('TRACKER'))
tracker_logger.addHandler(file_handler)
tracker_logger.addHandler(stream_handler)
tracker_logger.propagate = False

# Sentry logger
sentry_logger = logging.getLogger('sentry')
sentry_logger.setLevel(logging.INFO)
sentry_logger.addFilter(GearFilter('SENTRY'))
sentry_logger.addHandler(file_handler)
sentry_logger.addHandler(stream_handler)
sentry_logger.propagate = False

# System logger for general info
system_logger = logging.getLogger('system')
system_logger.setLevel(logging.INFO)
system_logger.addFilter(GearFilter('SYSTEM'))
system_logger.addHandler(file_handler)
system_logger.addHandler(stream_handler)
system_logger.propagate = False

# Social logger
social_logger = logging.getLogger('social')
social_logger.setLevel(logging.INFO)
social_logger.addFilter(GearFilter('SOCIAL'))
social_logger.addHandler(file_handler)
social_logger.addHandler(stream_handler)
social_logger.propagate = False

# Predictor logger
predictor_logger = logging.getLogger('predictor')
predictor_logger.setLevel(logging.INFO)
predictor_logger.addFilter(GearFilter('PREDICTOR'))
predictor_logger.addHandler(file_handler)
predictor_logger.addHandler(stream_handler)
predictor_logger.propagate = False


# Polymarket API base URL for a single market
POLIMARKET_SINGLE_MARKET_API_URL = "https://gamma-api.polymarket.com/markets/{}"

# --- 2. Helper Functions ---
def fetch_single_market_data(market_id: str):
    """Fetches real-time data for a single market."""
    try:
        response = request_with_retry(
            "GET",
            POLIMARKET_SINGLE_MARKET_API_URL.format(market_id),
            timeout=20,
            max_retries=5,
            logger=system_logger,
        )
        return response.json()
    except json.JSONDecodeError as e:
        system_logger.error(f"JSON Decode Error for market {market_id}: {e}")
        return None
    except Exception as e:
        system_logger.error(f"API Error for market {market_id}: {e}")
        return None


def extract_primary_probability(market_data: dict) -> float:
    """Extract outcome[0] probability from list or JSON-encoded list."""
    raw_prices = market_data.get("outcomePrices", [])
    prices = raw_prices
    if isinstance(raw_prices, str):
        prices = json.loads(raw_prices)
    if not prices or prices[0] is None:
        raise ValueError("Missing outcomePrices[0]")
    return float(prices[0])


def create_baseline_snapshot(market_id: str, db: Session):
    """Creates a single, immediate snapshot for a new market."""
    market_data = fetch_single_market_data(market_id)
    if not market_data:
        sentry_logger.warning(f"Could not fetch data for baseline snapshot of {market_id}.")
        return

    try:
        probability = extract_primary_probability(market_data)
        volume = float(market_data.get("volumeNum", 0))

        snapshot = MarketSnapshot(
            market_id=market_id,
            probability=probability,
            volume=volume,
            timestamp=datetime.now(timezone.utc)
        )
        db.add(snapshot)
        db.commit()
        sentry_logger.info(f"Baseline snapshot created for new market {market_id}.")
    except (IndexError, TypeError, ValueError) as e:
        sentry_logger.error(f"Data parsing error for baseline snapshot of {market_id}: {e}")
        db.rollback()


# --- 3. Gear 1: The Tracker ---
def run_tracker():
    """
    Fetches the current state of all tracked markets and stores snapshots.
    """
    tracker_logger.info("Starting half-hourly market snapshot capture.")
    db = SessionLocal()
    try:
        market_ids = [m.market_id for m in db.query(Market.market_id).all()]
        if not market_ids:
            tracker_logger.warning("No markets to track. Skipping run.")
            return

        tracker_logger.info(f"Tracking {len(market_ids)} markets.")

        for market_id in market_ids:
            market_data = fetch_single_market_data(market_id)
            if not market_data:
                tracker_logger.warning(f"Could not fetch data for market {market_id}. Skipping.")
                continue

            try:
                probability = extract_primary_probability(market_data)
                volume = float(market_data.get("volumeNum", 0))
                snapshot = MarketSnapshot(
                    market_id=market_id, probability=probability, volume=volume,
                    timestamp=datetime.now(timezone.utc)
                )
                db.add(snapshot)
            except (IndexError, TypeError, ValueError) as e:
                tracker_logger.error(f"Data parsing error for market {market_id}: {e}")
                continue
        
        db.commit()
        tracker_logger.info("Snapshot capture complete.")
        upload_db()

    except Exception as e:
        tracker_logger.error(f"An unexpected error occurred in the tracker: {e}")
        db.rollback()
    finally:
        db.close()


# --- 4. Gear 2: The Sentry ---
def run_sentry():
    """
    Scans all Polymarket markets, finds new ones, and adds them to the database.
    """
    sentry_logger.info("Starting 6-hour full market scan.")
    db = SessionLocal()
    try:
        existing_market_ids = {m.market_id for m in db.query(Market.market_id).all()}
        sentry_logger.info(f"Found {len(existing_market_ids)} existing markets in DB.")
        
        all_live_markets = []
        limit = 200
        offset = 0
        pages = 0
        max_pages = max(1, int(os.getenv("SENTRY_MAX_PAGES", "40")))
        
        while True:
            sentry_logger.info(f"Fetching markets with offset={offset}...")
            markets_batch = fetch_markets(limit=limit, offset=offset)
            if not markets_batch:
                break
            all_live_markets.extend(markets_batch)
            pages += 1
            if len(markets_batch) < limit:
                break
            if pages >= max_pages:
                sentry_logger.warning(f"Reached SENTRY_MAX_PAGES={max_pages}; stopping scan early.")
                break
            offset += limit
            jitter_sleep(1.0, 5.0)

        sentry_logger.info(f"Fetched a total of {len(all_live_markets)} markets from API.")
        
        # Apply "Relaxed Filters"
        new_markets_to_add = []
        seen_market_ids = set(existing_market_ids)
        for market in all_live_markets:
            market_id = market.get("id")
            if not market_id:
                continue

            # Check if it's new
            if market_id not in seen_market_ids:
                # Check filters: Active and Volume > $2000
                if market.get("closed") is False and _safe_float(market.get("volumeNum", 0)) > 2000:
                    new_markets_to_add.append(market)
                    seen_market_ids.add(market_id)

        if not new_markets_to_add:
            sentry_logger.info("No new markets found matching criteria.")
            return

        sentry_logger.info(f"Found {len(new_markets_to_add)} new markets to add.")
        
        # Store new markets and create baseline snapshots
        store_markets(new_markets_to_add, db)
        sentry_logger.info(f"Successfully stored {len(new_markets_to_add)} new markets.")
        
        for market_data in new_markets_to_add:
            create_baseline_snapshot(market_data['id'], db)

        upload_db() 
        sentry_logger.info("Database backed up after Sentry run.")

    except Exception as e:
        sentry_logger.error(f"An unexpected error occurred in the sentry: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def run_social():
    """Runs the social-signal monitor and stores aggregate metrics."""
    social_logger.info("Starting 2-hour social signal collection run.")
    db = SessionLocal()
    try:
        run_social_monitor(db=db, logger=social_logger)
        upload_db()
    except Exception as e:
        social_logger.error(f"An unexpected error occurred in social monitor: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def run_predictor():
    """Runs model retraining (if stale) and generates latest surge predictions."""
    predictor_logger.info("Starting predictor cycle run.")
    db = SessionLocal()
    try:
        run_predictor_cycle(db=db, logger=predictor_logger, retrain_every_hours=12)
        upload_db()
    except Exception as e:
        predictor_logger.error(f"An unexpected error occurred in predictor: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


# --- 5. Scheduler Setup ---
if __name__ == "__main__":
    system_logger.info("Initializing Gemini Backend Service...")
    download_db()
    
    scheduler = BackgroundScheduler(timezone="UTC")
    
    # Schedule Tracker (Gear 1)
    scheduler.add_job(run_tracker, 'interval', minutes=30)
    
    # Schedule Sentry (Gear 2)
    scheduler.add_job(run_sentry, 'interval', hours=6)

    # Schedule Social Monitor (Gear 3)
    scheduler.add_job(run_social, 'interval', hours=2)

    # Schedule Predictor (Gear 4)
    scheduler.add_job(run_predictor, 'interval', hours=1)
    
    scheduler.start()
    system_logger.info("Scheduler started. Jobs are now running on their respective intervals.")
    
    # Initial "catch-up" runs on startup
    system_logger.info("Performing initial catch-up runs...")
    run_sentry()
    run_tracker()
    run_social()
    run_predictor()
    system_logger.info("Initial runs complete. Service is now in standard operation.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        system_logger.info("Shutting down Gemini Backend Service.")
        scheduler.shutdown()

