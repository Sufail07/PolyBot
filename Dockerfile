FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/
COPY backend.py ./backend.py
COPY database.py ./database.py
COPY social_monitor.py ./social_monitor.py
COPY signal_predictor.py ./signal_predictor.py
COPY polymarket_scraper.py ./polymarket_scraper.py
COPY http_utils.py ./http_utils.py

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
