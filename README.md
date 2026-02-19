# Polymarket Movement Predictor

A real-time predictive market signal engine that combines social media monitoring with Polymarket market data to detect emerging trends before they're priced in.

## **Problem Statement**

* Traders often rely on raw market data or visible price movements.
* Current tools track historical price changes or institutional activity but do **not predict movement based on narrative or social signals**.
* Prediction markets (e.g., Polymarket) have slower resolution cycles, giving the opportunity to identify **signal-driven probability changes** before outcomes are priced in.

**Gemini solves this by:**

* Identifying candidate markets with meaningful volume and volatility.
* Tracking social discussion and narrative spikes around these markets.
* Linking social activity to subsequent probability changes.

## **Features**

- **Social Signal Detection**: Monitors Google News, Bluesky, and Reddit for market-relevant content
- **Polymarket Integration**: Real-time market data via Gamma API
- **Prediction Engine**: Logistic regression model with dynamic surge detection
- **Alert System**: Telegram notifications for high-conviction signals
- **Dashboard**: Streamlit-based operational monitoring

## **Architecture**

- **Data Pipeline**: SQLite database with social signals, market snapshots, and predictions
- **Scheduler**: APScheduler for automated data collection and prediction
- **Modeling**: Rolling z-scores, volatility-adjusted surge thresholds
- **Alerting**: Configurable probability thresholds with cooldown periods

## **Getting Started**

```bash
pip install -r requirements.txt
python backend.py
```

## **Configuration**

- Edit `config.py` for API keys and thresholds
- Dashboard available at `http://localhost:8501`
- Telegram alerts configured via GitHub Actions relay

## **Market Coverage**

Currently tracking 8 active Polymarket markets with focus on political predictions.

## **License**

MIT

---

**Disclaimer:** This is a hypothetical project concept and not confirmed or implemented yet.