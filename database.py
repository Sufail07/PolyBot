from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Integer, ForeignKey, Boolean, text
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

DATABASE_URL = "sqlite:///./polymarket.db"

Base = declarative_base()

class Market(Base):
    __tablename__ = "markets"

    market_id = Column(String, primary_key=True, index=True)
    question = Column(String, index=True)
    category = Column(String)
    outcomes = Column(Text)  # JSON string
    current_probabilities = Column(Text)  # JSON string
    volume = Column(Float)
    resolution_date = Column(DateTime)
    bid_ask_spread = Column(Float, nullable=True)
    momentum = Column(Float, nullable=True)
    score = Column(Float, nullable=True)

    snapshots = relationship("MarketSnapshot", back_populates="market")
    social_signals = relationship("SocialSignal", back_populates="market")
    processed_submissions = relationship("ProcessedSubmission", back_populates="market")

    def __repr__(self):
        return f"<Market(id='{self.market_id}', question='{self.question}')>"

class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(String, ForeignKey("markets.market_id"))
    probability = Column(Float)
    volume = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    market = relationship("Market", back_populates="snapshots")

    def __repr__(self):
        return f"<MarketSnapshot(market_id='{self.market_id}', probability={self.probability}, timestamp='{self.timestamp}')>"


class SocialSignal(Base):
    __tablename__ = "social_signals"

    id = Column(Integer, primary_key=True, index=True)
    market_id = Column(String, ForeignKey("markets.market_id"), index=True)
    raw_count = Column(Integer, default=0)
    weighted_score = Column(Float, default=0.0)
    avg_sentiment = Column(Float, default=0.0)
    unique_authors = Column(Integer, default=0)
    is_organic = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    market = relationship("Market", back_populates="social_signals")

    def __repr__(self):
        return (
            f"<SocialSignal(market_id='{self.market_id}', raw_count={self.raw_count}, "
            f"weighted_score={self.weighted_score}, timestamp='{self.timestamp}')>"
        )


class ProcessedSubmission(Base):
    __tablename__ = "processed_submissions"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(String, unique=True, index=True, nullable=False)
    market_id = Column(String, ForeignKey("markets.market_id"), index=True, nullable=False)
    first_seen_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    market = relationship("Market", back_populates="processed_submissions")

    def __repr__(self):
        return f"<ProcessedSubmission(submission_id='{self.submission_id}', market_id='{self.market_id}')>"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def _ensure_schema_compatibility():
    """
    Lightweight runtime migration for SQLite to add newly introduced columns.
    """
    with engine.connect() as conn:
        table_info = conn.execute(text("PRAGMA table_info(social_signals)")).fetchall()
        column_names = {row[1] for row in table_info}
        if "is_organic" not in column_names:
            conn.execute(text("ALTER TABLE social_signals ADD COLUMN is_organic BOOLEAN DEFAULT 0"))
            conn.commit()


_ensure_schema_compatibility()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
