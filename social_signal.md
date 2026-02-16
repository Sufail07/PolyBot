**Task: Build the 'Reddit Signal Gear' with Anti-Manipulation Filters for Gemini.**

**Objective:** Create `social_monitor.py` to detect genuine narrative shifts while filtering out low-quality noise and coordinated pumps.

**Requirements:**

1. **Data Ingestion (PRAW):** >    * Search `r/Polymarket`, `r/CryptoCurrency`, `r/Politics`, and `r/WorldNews`.
   * For each market in the `markets` table, use the title (cleaned of "Will" and "by [Date]") as the query.
2. **Account Verification (The "Anti-Pump" Layer):**
   * For every mention found, fetch the author’s **Account Age** and  **Total Karma** .
   * **Filter:** Ignore mentions from accounts < 30 days old OR with < 500 total karma.
3. **Crossover Weighting (The "Organic" Signal):**
   * Assign a `source_weight` to mentions based on the subreddit:
     * `r/Polymarket`: 1.0 (Baseline)
     * `r/CryptoCurrency`: 1.5 (High interest)
     * `r/Politics` / `r/WorldNews`: 3.0 (Narrative Breakout/Mainstream)
4. **Sentiment & Scoring:**
   * Use `nltk.sentiment.vader` for sentiment analysis (Compound score).
   * Calculate a `Trust_Score` for each market session:
     $$
     Trust\_Score = (Unique\_Authors \times Avg\_Karma \times Crossover\_Bonus)
     $$
5. **Database (New Table):**
   * Create `social_signals`: `(id, market_id, raw_count, weighted_score, avg_sentiment, unique_authors, timestamp)`.
6. **Integration:** >    * Add this as **Gear 3** to the `BackgroundScheduler` in `backend.py`, running every 2 hours.
   * Implement a "Deduplication" check using `submission.id` to ensure the same post isn't counted in consecutive runs.
