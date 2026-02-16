import requests
import json
from datetime import datetime, timedelta, timezone
from dateutil.parser import isoparse
from sqlalchemy.orm import Session
from database import SessionLocal, Market, engine, Base
import numpy as np
from http_utils import request_with_retry

# Polymarket API endpoint
POLIMARKET_API_URL = "https://gamma-api.polymarket.com/markets"

def fetch_markets(limit: int, offset: int, end_date_min: datetime = None, end_date_max: datetime = None, volume_num_min: int = None):
    """Fetches all markets from the Polymarket API with specified query parameters."""
    params = {
        "closed": "false",
        "limit": limit,
        "offset": offset
    }
    if end_date_min:
        params["end_date_min"] = end_date_min.isoformat()
    if end_date_max:
        params["end_date_max"] = end_date_max.isoformat()
    if volume_num_min:
        params["volume_num_min"] = volume_num_min

    response = request_with_retry(
        "GET",
        POLIMARKET_API_URL,
        params=params,
        timeout=20,
        max_retries=5,
    )
    return response.json()

def score_and_filter_markets(markets: list) -> list:
    """
    Scores and filters markets based on a hard liquidity floor, bid-ask spread, and probability thresholds.
    """
    
    HARD_MIN_VOLUME_USD = 2000
    MAX_SPREAD = 0.10 # 10%
    MIN_PROBABILITY = 0.10 # 10%
    
    ALLOWED_CATEGORIES = ["Technology", "Crypto", "Politics", "Science", "Business", "News"] 

    filtered_markets = []

    for market_data in markets:
        
        volume = market_data.get("volumeNum", 0)
        best_bid = market_data.get("bestBid")
        best_ask = market_data.get("bestAsk")
        spread = float(best_ask) - float(best_bid) if best_bid is not None and best_ask is not None else None
        market_data['bid_ask_spread'] = spread

        current_prob = 0
        outcome_prices = market_data.get("outcomePrices")
        if outcome_prices:
            try:
                # Handle cases where outcomePrices is a stringified list '["0.1", "0.9"]' or a direct list
                prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                if prices and len(prices) > 0 and prices[0] is not None:
                    current_prob = float(prices[0])
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                current_prob = 0

        category = market_data.get("category")
        question = market_data.get("question", "").lower()
        
        print(f"--- Checking Market: {market_data.get('id')} ---")
        print(f"Volume: {volume}, Spread: {spread}, Prob: {current_prob}, Category: {category}")

        if volume < HARD_MIN_VOLUME_USD:
            print("Failed Volume Filter")
            continue

        if spread is None or spread > MAX_SPREAD:
            print("Failed Spread Filter")
            continue
            
        if current_prob < MIN_PROBABILITY:
            print("Failed Probability Filter")
            continue

        in_category = False
        if category and category in ALLOWED_CATEGORIES:
            in_category = True
        elif any(cat.lower() in question for cat in ALLOWED_CATEGORIES):
            in_category = True
        
        if not in_category:
            print("Failed Category Filter")
            continue
            
        # --- Scoring (only for markets that pass filters) ---
        print("Market Passed All Filters!")
        momentum = abs(market_data.get("oneDayPriceChange", 0) or 0)
        
        normalized_momentum = momentum
        normalized_volume = min(volume / 1_000_000, 1.0)
        
        alpha = 0.6
        beta = 0.4
        
        score = (alpha * normalized_momentum) + (beta * normalized_volume)

        market_data['momentum'] = momentum
        market_data['score'] = score

        filtered_markets.append(market_data)
    
    return filtered_markets


def store_markets(markets_data: list, db: Session):
    """Stores scored and filtered market data into the database."""
    for market_data in markets_data:
        # Prepare data for the Market model
        market_id = market_data.get("id")
        
        # Check if market already exists
        existing_market = db.query(Market).filter(Market.market_id == market_id).first()

        outcomes_json = json.dumps(market_data.get("outcomes")) if market_data.get("outcomes") else None
        current_probabilities_json = json.dumps(market_data.get("outcomePrices")) if market_data.get("outcomePrices") else None

        resolution_date_str = market_data.get("endDate")
        resolution_date = isoparse(resolution_date_str) if resolution_date_str else None

        if existing_market:
            # Update existing market
            existing_market.question = market_data.get("question")
            existing_market.category = market_data.get("category")
            existing_market.outcomes = outcomes_json
            existing_market.current_probabilities = current_probabilities_json
            existing_market.volume = market_data.get("volumeNum")
            existing_market.resolution_date = resolution_date
            existing_market.bid_ask_spread = market_data.get('bid_ask_spread')
            existing_market.momentum = market_data.get('momentum')
            existing_market.score = market_data.get('score')
        else:
            # Add new market
            market = Market(
                market_id=market_id,
                question=market_data.get("question"),
                category=market_data.get("category"),
                outcomes=outcomes_json,
                current_probabilities=current_probabilities_json,
                volume=market_data.get("volumeNum"),
                resolution_date=resolution_date,
                bid_ask_spread=market_data.get('bid_ask_spread'),
                momentum=market_data.get('momentum'),
                score=market_data.get('score'),
            )
            db.add(market)
    db.commit()


def generate_summary_report(markets: list, all_markets_count: int):
    """Generates and prints a summary report of the scored and filtered markets."""
    total_markets_fetched = all_markets_count
    total_markets_filtered = len(markets)

    all_scores = [m.get('score', 0) for m in markets if m.get('score') is not None]
    
    if all_scores:
        score_stats = {
            "min": min(all_scores),
            "max": max(all_scores),
            "avg": sum(all_scores) / len(all_scores)
        }
    else:
        score_stats = {"min": "N/A", "max": "N/A", "avg": "N/A"}

    # Top 5 candidates by score
    top_candidates = sorted(markets, key=lambda m: m.get('score', 0), reverse=True)[:5]

    print("\n--- Market Analysis Report ---")
    print(f"Total markets fetched: {total_markets_fetched}")
    print(f"Total markets after filtering: {total_markets_filtered}")
    
    print("\nScore Distribution Statistics:")
    print(f"  - Minimum Score: {score_stats['min']:.4f}" if isinstance(score_stats['min'], float) else f"  - Minimum Score: {score_stats['min']}")
    print(f"  - Maximum Score: {score_stats['max']:.4f}" if isinstance(score_stats['max'], float) else f"  - Maximum Score: {score_stats['max']}")
    print(f"  - Average Score: {score_stats['avg']:.4f}" if isinstance(score_stats['avg'], float) else f"  - Average Score: {score_stats['avg']}")

    print("\nTop 5 Candidate Markets by Score:")
    for i, market in enumerate(top_candidates, 1):
        print(f"  {i}. {market.get('question')} (Score: {market.get('score', 0):.4f})")
    print("--- End of Report ---\n")


def main():
    Base.metadata.create_all(bind=engine) # Ensure tables are created
    db = SessionLocal()
    try:
        print("Fetching all active markets with pagination...")
        
        all_fetched_markets = []
        limit = 200
        offset = 0
        
        # Set the minimum end date to now to only get active markets
        end_date_min = datetime.now(timezone.utc)
        
        seen_market_ids = set()
        while True:
            print(f"Fetching markets with limit={limit} and offset={offset}...")
            markets = fetch_markets(limit=limit, offset=offset, end_date_min=end_date_min)
            if not markets:
                break
            
            for market in markets:
                market_id = market.get('id')
                if market_id and market_id not in seen_market_ids:
                    all_fetched_markets.append(market)
                    seen_market_ids.add(market_id)
            
            offset += limit

        print(f"Fetched a total of {len(all_fetched_markets)} unique markets.")

        print("Scoring and filtering markets...")
        scored_and_filtered_markets = score_and_filter_markets(all_fetched_markets)
        
        print(f"Found {len(scored_and_filtered_markets)} markets after scoring and filtering.")

        print("Storing markets in the database...")
        store_markets(scored_and_filtered_markets, db)
        print("Markets stored successfully.")
        
        generate_summary_report(scored_and_filtered_markets, len(all_fetched_markets))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Polymarket API: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
