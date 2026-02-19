## **Project Name:**

**Gemini** – Real-Time Predictive Market Signal Engine (Polymarket Focus)

---

## **1. Project Aim**

Gemini aims to detect emerging market trends  **before they are priced in** , using real-time social and market data. The initial focus is on **Polymarket markets** (prediction markets) to identify opportunities where social signals or narrative shifts precede changes in market probabilities.

The long-term vision is to create a **predictive SaaS platform** for traders and analysts that provides actionable insights on emerging trends across markets.

---

## **2. Problem Statement**

* Traders often rely on raw market data or visible price movements.
* Current tools track historical price changes or institutional activity but do  **not predict movement based on narrative or social signals** .
* Prediction markets (e.g., Polymarket) have slower resolution cycles, giving the opportunity to identify **signal-driven probability changes** before outcomes are priced in.

**Gemini solves this by:**

* Identifying candidate markets with meaningful volume and volatility.
* Tracking social discussion and narrative spikes around these markets.
* Linking social activity to subsequent probability changes.

---

## **3. Project Scope**

### **Phase 1 – Market Scanning & Storage**

* **Goal:** Identify eligible Polymarket markets and store metadata in a database.
* **Inputs:** Polymarket API (or mock data if API access unavailable).
* **Outputs:** DB table of candidate markets with:
  * `market_id`
  * `question`
  * `category`
  * `outcomes`
  * `current_probabilities`
  * `volume`
  * `resolution_date`
  * Optional: `bid_ask_spread`
* **Filters:**
  * Resolution 14–90 days
  * Volume > $100k
  * Probability per outcome between 30–70%

---

### **Phase 2 – Probability Time Series Tracking**

* **Goal:** Capture hourly snapshots of market probabilities and volume.
* **Outputs:** Time series table:
  * `market_id`
  * `timestamp`
  * `outcome_id`
  * `probability`
  * `hourly_volume`

---

### **Phase 3 – Social Signal Detection**

* **Sources:** Reddit + X (Twitter)
* **Goal:** Track posts mentioning market-relevant keywords.
* **Process:**
  * Extract relevant posts/tweets per market
  * Filter noise (spam, low engagement, unrelated posts)
  * Compute “social shocks” based on bursts in mentions (z-score or rolling mean)
* **Outputs:** Table of social shock events:
  * `market_id`
  * `timestamp`
  * `z_score`
  * `mention_count`

---

### **Phase 4 – Event Study & Correlation**

* **Goal:** Quantify the impact of social signals on probability movement.
* **Process:**
  * Align social shocks with market probability time series
  * Measure probability drift at 1h, 6h, 12h, 24h after shock
  * Compare to baseline (no-shock periods)
* **Output:** Market-event correlation table for ranking signals

---

### **Phase 5 – Scoring & Ranking**

* Rank markets by predictive signal strength and liquidity-adjusted potential.
* Prioritize top markets for deeper analysis or automated alerts.

---

## **4. Requirements**

### **Functional Requirements**

1. Fetch Polymarket market data (API or mock).
2. Filter markets based on resolution, volume, probability.
3. Store filtered markets in a relational database (SQLite/Postgres).
4. Capture hourly market probability snapshots.
5. Track and filter social signals (Reddit + X).
6. Compute social shocks and correlate with market movements.
7. Provide candidate market ranking based on predictive signal strength.

### **Non-Functional Requirements**

1. **Modular architecture:** Each phase is independent, allows swapping sources or expanding.
2. **Scalable MVP:** Begin with 10–20 markets; expand to hundreds later.
3. **Low-cost testing:** Can operate initially with minimal compute (no GPU needed).
4. **Data integrity:** Handle missing fields, malformed dates, API errors.
5. **Extensibility:** Add new social sources, other prediction platforms, or automated alerts in future phases.

---

## **5. MVP Deliverables**

1. Database of candidate markets with metadata.
2. Hourly probability snapshots for each market.
3. Social signal tracking pipeline with filtering.
4. Event analysis showing correlation between social spikes and probability drift.
5. Simple ranking of markets by signal strength.

---

## **6. Technical Stack**

* **Python:** `requests`, `pandas`, `SQLAlchemy`
* **DB:** SQLite for MVP, Postgres for scale
* **APIs:** Polymarket, Reddit (Pushshift/official), X (Twitter API)
* **Scheduling:** Cron jobs or Python scheduler (`APScheduler`)
* **Optional:** NLP libraries for sentiment/noise filtering (`transformers`, `vaderSentiment`)
*
