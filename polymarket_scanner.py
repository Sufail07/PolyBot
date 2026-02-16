import requests
import json
from datetime import datetime, timedelta
from dateutil.parser import isoparse # For robust ISO 8601 date parsing
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError

# --- Configuration ---
DATABASE_URL = "sqlite:///polymarket_markets.db"
POLYMARKET_API_URL = "https://api.polymarket.com/v2/markets"

# Filtering criteria
MIN_RESOLUTION_DAYS = 14
MAX_RESOLUTION_DAYS = 90
MIN_VOLUME_USD = 100000
MIN_PROBABILITY = 0.30  # 30%
MAX_PROBABILITY = 0.70  # 70%

# --- Database Setup ---
Base = declarative_base()

class Market(Base):
    __tablename__ = 'markets'
    market_id = Column(String, primary_key=True)
    question = Column(String, nullable=False)
    category = Column(String)
    # Storing outcomes as a JSON string of titles, e.g., '["Yes", "No"]'
    outcomes_json = Column(Text, nullable=False)
    # Storing current probabilities as a JSON string of floats, e.g., '[0.65, 0.35]'
    current_probabilities_json = Column(Text, nullable=False)
    volume = Column(Float)
    resolution_date = Column(DateTime)
    # bid_ask_spread is often not directly available from the basic market endpoint, default to None
    bid_ask_spread = Column(Float, nullable=True)

    def __repr__(self):
        return (f"<Market(id='{self.market_id}', question='{self.question[:50]}...', "
                f"volume=${self.volume:,.2f}, resolution_date={self.resolution_date})>")

# --- API Functions ---

def fetch_active_polymarket_markets():
    """
    Fetches all active markets from the Polymarket API.
    Returns a list of market dictionaries.
    """
    print(f"Fetching markets from {POLYMARKET_API_URL}...")
    try:
        response = requests.get(POLYMARKET_API_URL, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if 'markets' in data:
            print(f"Successfully fetched {len(data['markets'])} markets.")
            return data['markets']
        else:
            print("API response did not contain 'markets' key.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching markets: {e}")
        return []

# --- Filtering Functions ---

def is_market_eligible(market_data):
    """
    Applies the filtering criteria to a single market.
    Returns True if the market is eligible, along with extracted probabilities
    and bid-ask spread (or None).
    """
    # 1. Resolution between 14–90 days
    try:
        resolution_dt = isoparse(market_data.get('end_date'))
        time_to_resolution = resolution_dt - datetime.utcnow()
        days_to_resolution = time_to_resolution.days

        if not (MIN_RESOLUTION_DAYS <= days_to_resolution <= MAX_RESOLUTION_DAYS):
            return False, [], None, None # Added None for bid_ask_spread_value
    except (TypeError, ValueError):
        # If end_date is missing or malformed, skip
        return False, [], None, None # Added None for bid_ask_spread_value

    # 2. Volume > $100k
    volume = market_data.get('volume_usd', 0)
    if volume <= MIN_VOLUME_USD:
        return False, [], None, None # Added None for bid_ask_spread_value

    # 3. Probability between 30–70% (for each outcome)
    outcomes = market_data.get('outcomes', [])
    if not outcomes:
        return False, [], None, None # Added None for bid_ask_spread_value

    current_probabilities = []
    outcome_titles = []
    for outcome in outcomes:
        price = outcome.get('price') # Polymarket uses 'price' for current probability
        if price is None:
            return False, [], None, None # Missing probability for an outcome, added None for bid_ask_spread_value

        if not (MIN_PROBABILITY <= price <= MAX_PROBABILITY):
            return False, [], None, None # Any probability outside range makes market ineligible, added None for bid_ask_spread_value
        current_probabilities.append(price)
        outcome_titles.append(outcome.get('title', 'Unknown'))

    # Bid-ask spread (if available) - not typically in this API endpoint, so default to None
    # For Phase 1, we'll assume it's not directly in the market_data unless specified by user.
    bid_ask_spread_value = market_data.get('bid_ask_spread', None) # Placeholder if API provides it

    return True, outcome_titles, current_probabilities, bid_ask_spread_value

# --- Database Functions ---

def save_market_data(session, market_data, outcome_titles, probabilities, bid_ask_spread_value):
    """
    Saves or updates market data in the database.
    """
    market_id = market_data.get('id')
    
    # Check if market already exists
    existing_market = session.query(Market).filter_by(market_id=market_id).first()

    # Convert lists to JSON strings for storage
    outcomes_json_str = json.dumps(outcome_titles)
    probabilities_json_str = json.dumps(probabilities)

    if existing_market:
        # Update existing market
        existing_market.question = market_data.get('question')
        existing_market.category = market_data.get('category')
        existing_market.outcomes_json = outcomes_json_str
        existing_market.current_probabilities_json = probabilities_json_str
        existing_market.volume = market_data.get('volume_usd')
        existing_market.resolution_date = isoparse(market_data.get('end_date'))
        existing_market.bid_ask_spread = bid_ask_spread_value
        print(f"Updated market: {market_id}")
    else:
        # Create new market
        new_market = Market(
            market_id=market_id,
            question=market_data.get('question'),
            category=market_data.get('category'),
            outcomes_json=outcomes_json_str,
            current_probabilities_json=probabilities_json_str,
            volume=market_data.get('volume_usd'),
            resolution_date=isoparse(market_data.get('end_date')),
            bid_ask_spread=bid_ask_spread_value
        )
        session.add(new_market)
        print(f"Added new market: {market_id}")

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        print(f"Error saving market {market_id}, likely a race condition or constraint violation. Rolling back.")


def get_candidate_markets(session, min_volume=100000, min_prob=0.3, max_prob=0.7):
    """
    Queries the database to get candidate markets based on specific criteria.
    Example: Retrieve markets with volume > $100k where all probabilities are within 30-70%.
    """
    print("\nQuerying candidate markets from the database:")
    # For more complex JSON querying, one might use database-specific JSON functions
    # or process after retrieval. Here, we'll filter on volume and retrieve all.
    # The probability filtering is done during initial ingestion.

    candidate_markets = session.query(Market).filter(
        Market.volume > min_volume
        # Add other filters if needed, e.g., on resolution_date, category etc.
    ).all()

    if candidate_markets:
        for market in candidate_markets:
            print(f"- ID: {market.market_id}, Question: {market.question[:70]}..., "
                  f"Volume: ${market.volume:,.2f}, Probabilities: {market.current_probabilities_json}")
    else:
        print("No candidate markets found matching the criteria.")
    return candidate_markets


# --- Main Execution ---

if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine) # Create tables if they don't exist

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Fetch, filter, and store markets
        markets_data = fetch_active_polymarket_markets()
        processed_count = 0
        saved_count = 0

        for market in markets_data:
            eligible, outcome_titles, probabilities, spread = is_market_eligible(market)
            if eligible:
                save_market_data(session, market, outcome_titles, probabilities, spread)
                saved_count += 1
            processed_count += 1
        
        print(f"\nScan complete. Processed {processed_count} markets, saved {saved_count} eligible markets.")

        # Example of querying the database
        get_candidate_markets(session, min_volume=100000)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        session.rollback()
    finally:
        session.close()
