# my_project/common/config.py

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# ——— Load .env from project root ———
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)

# ——— Default RSS URL constants ———
DEFAULT_CRYPTO_RSS_URLS = ",".join([
    # … (eredeti lista változatlanul) …
])

DEFAULT_STOCK_RSS_URLS = ",".join([
    # … (eredeti lista változatlanul) …
])

# ——— Keyword mappings ———
STOCK_KEYWORDS = {
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AAPL": ["apple"]
}
CRYPTO_KEYWORDS = {
    "BTC": ["btc", "bitcoin"],
    "SOL": ["solana"],
    "XRP": ["xrp", "ripple"]
}

class Settings(BaseSettings):
    # — Primary DB —
    DB_HOST: str     = "localhost"
    DB_NAME: str     = "market_news_db"
    DB_USER: str     = "kraken_user"
    DB_PASSWORD: str = "9911_Leslie"

    # — Trading DB —
    TRADING_DB_HOST: str     = "localhost"
    TRADING_DB_NAME: str     = "trading_db"
    TRADING_DB_USER: str     = "kraken_user"
    TRADING_DB_PASSWORD: str = "9911_Leslie"

    # — Alpaca DB —
    ALPACA_DB_HOST: str = "localhost"
    ALPACA_DB_NAME: str = "alpaca_db"
    ALPACA_DB_USER: str = "kraken_user"
    ALPACA_DB_PASS: str = "9911_Leslie"

    # — Kraken DB —
    KRAKEN_DB_HOST: str = "localhost"
    KRAKEN_DB_NAME: str = "kraken_db"
    KRAKEN_DB_USER: str = "kraken_user"
    KRAKEN_DB_PASS: str = "9911_Leslie"

    # — RSS feeds (ENV-valued CSV) —
    STOCK_RSS_URLS: str  = Field(DEFAULT_STOCK_RSS_URLS, description="CSV list of stock RSS URLs")
    CRYPTO_RSS_URLS: str = Field(DEFAULT_CRYPTO_RSS_URLS, description="CSV list of crypto RSS URLs")

    # — Network & fetching —
    MAX_ENTRIES_PER_FEED:    int   = 10
    MAX_RETRIES:             int   = 3
    REQUEST_TIMEOUT:         int   = 30
    SSL_VERIFY:              bool  = True
    SSL_CIPHERS:             str   = "DEFAULT:!DH"
    DNS_SERVERS:             str   = "8.8.8.8,1.1.1.1"
    MAX_CONCURRENT_REQUESTS: int   = 10
    REQUESTS_PER_MINUTE:     int   = 60

    # — AWS Comprehend —
    AWS_REGION: str = "eu-central-1"

    # — Sentiment analysis tuning —
    SENTIMENT_BATCH_SIZE:    int   = 100
    RELEVANCE_THRESHOLD:     float = 2.5
    SYMBOL_DETECTION_THRESHOLD: int = 5
    FRESHNESS_DECAY_RATE:    float = 0.15
    MIN_SOURCE_REPUTATION:   float = 0.4

    # — Keyword mappings —
    STOCK_KEYWORDS:  dict = Field(STOCK_KEYWORDS, description="Stock symbol keywords")
    CRYPTO_KEYWORDS: dict = Field(CRYPTO_KEYWORDS, description="Crypto symbol keywords")

    class Config:
        env_file = dotenv_path
        extra    = "ignore"

settings = Settings()

class Config:
    """Central application configuration."""

    @staticmethod
    def _make_uri(user, pw, host, db):
        return f"mysql+pymysql://{user}:{pw}@{host}/{db}"

    # — Primary DB URIs —
    SQLALCHEMY_DATABASE_URI = _make_uri(
        settings.DB_USER, settings.DB_PASSWORD,
        settings.DB_HOST, settings.DB_NAME
    )
    NEWS_DB_URI    = SQLALCHEMY_DATABASE_URI
    TRADING_DB_URI = _make_uri(
        settings.TRADING_DB_USER, settings.TRADING_DB_PASSWORD,
        settings.TRADING_DB_HOST, settings.TRADING_DB_NAME
    )
    ALPACA_DB_URI  = _make_uri(
        settings.ALPACA_DB_USER, settings.ALPACA_DB_PASS,
        settings.ALPACA_DB_HOST, settings.ALPACA_DB_NAME
    )
    KRAKEN_DB_URI  = _make_uri(
        settings.KRAKEN_DB_USER, settings.KRAKEN_DB_PASS,
        settings.KRAKEN_DB_HOST, settings.KRAKEN_DB_NAME
    )

    # — RSS URLs as lists —
    STOCK_RSS_URLS  = [
        u.strip() for u in settings.STOCK_RSS_URLS.split(",") if u.strip()
    ]
    CRYPTO_RSS_URLS = [
        u.strip() for u in settings.CRYPTO_RSS_URLS.split(",") if u.strip()
    ]

    # — Network & fetching —
    MAX_ENTRIES_PER_FEED    = settings.MAX_ENTRIES_PER_FEED
    MAX_RETRIES             = settings.MAX_RETRIES
    REQUEST_TIMEOUT         = settings.REQUEST_TIMEOUT
    SSL_VERIFY              = settings.SSL_VERIFY
    SSL_CIPHERS             = settings.SSL_CIPHERS
    DNS_SERVERS             = [
        s.strip() for s in settings.DNS_SERVERS.split(",") if s.strip()
    ]
    MAX_CONCURRENT_REQUESTS = settings.MAX_CONCURRENT_REQUESTS
    REQUESTS_PER_MINUTE     = settings.REQUESTS_PER_MINUTE

    # — AWS —
    AWS_REGION              = settings.AWS_REGION

    # — Sentiment analysis params —
    SENTIMENT_BATCH_SIZE         = settings.SENTIMENT_BATCH_SIZE
    RELEVANCE_THRESHOLD          = settings.RELEVANCE_THRESHOLD
    SYMBOL_DETECTION_THRESHOLD   = settings.SYMBOL_DETECTION_THRESHOLD
    FRESHNESS_DECAY_RATE         = settings.FRESHNESS_DECAY_RATE
    MIN_SOURCE_REPUTATION        = settings.MIN_SOURCE_REPUTATION

    # — Keyword mappings —
    STOCK_KEYWORDS   = settings.STOCK_KEYWORDS
    CRYPTO_KEYWORDS  = settings.CRYPTO_KEYWORDS

# Legacy exports for backward compatibility
STOCK_KEYWORDS  = Config.STOCK_KEYWORDS
CRYPTO_KEYWORDS = Config.CRYPTO_KEYWORDS

