# Polymarket Movement Predictor

A real-time predictive market signal engine that combines social media monitoring with Polymarket market data to detect emerging trends before they're priced in.

## Features

- **Social Signal Detection**: Monitors Google News, Bluesky, and Reddit for market-relevant content
- **Polymarket Integration**: Real-time market data via Gamma API
- **Prediction Engine**: Logistic regression model with dynamic surge detection
- **Alert System**: Telegram notifications for high-conviction signals
- **Dashboard**: Streamlit-based operational monitoring

## Architecture

- **Data Pipeline**: SQLite database with social signals, market snapshots, and predictions
- **Scheduler**: APScheduler for automated data collection and prediction
- **Modeling**: Rolling z-scores, volatility-adjusted surge thresholds
- **Alerting**: Configurable probability thresholds with cooldown periods

## Getting Started

```bash
pip install -r requirements.txt
python backend.py
```

## Configuration

- Edit `config.py` for API keys and thresholds
- Dashboard available at `http://localhost:8501`
- Telegram alerts configured via GitHub Actions relay

## Market Coverage

Currently tracking 8 active Polymarket markets with focus on political predictions.

## License

MIT