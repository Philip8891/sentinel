#!/usr/bin/env python3
from datetime import datetime, timezone, time
import concurrent.futures
import pytz
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from my_project.common.config import Config
from my_project.common.logger import logger
from sqlalchemy import text
from my_project.utils.db_connector import get_db_engine
import os

def in_stock_trading_hours():
    amsterdam_tz = pytz.timezone("Europe/Amsterdam")
    now_amsterdam = datetime.now(amsterdam_tz)
    ny_tz = pytz.timezone("America/New_York")
    now_ny = now_amsterdam.astimezone(ny_tz).time()
    return time(15, 30) <= now_ny <= time(22, 0)

if not in_stock_trading_hours():
    logger.info("Not within stock trading hours (New York time). Skipping Alpaca data collection.")
    exit(0)

SYMBOLS = os.getenv("SYMBOLS", "AAPL,TSLA,MSFT").split(",")

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
)

CALLS_PER_MINUTE = 200

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5))
def fetch_and_store(symbol: str):
    snapshot = api.get_snapshot(symbol)
    latest_trade = api.get_latest_trade(symbol)
    bars = list(api.get_bars(symbol, TimeFrame.Day, limit=5))
    
    data = {
        'timestamp': datetime.now(timezone.utc),
        'symbol': symbol,
        'price': getattr(latest_trade, 'price', 0.0),
        'open_price': getattr(snapshot.daily_bar, 'o', 0.0),
        'high': getattr(snapshot.daily_bar, 'h', 0.0),
        'low': getattr(snapshot.daily_bar, 'l', 0.0),
        'bid_price': getattr(snapshot.latest_quote, 'bid_price', 0.0),
        'ask_price': getattr(snapshot.latest_quote, 'ask_price', 0.0),
        'volume': getattr(snapshot.daily_bar, 'v', 0)
    }
    
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO market_data (timestamp, symbol, price, open_price, high, low, bid_price, ask_price, volume)
            VALUES (:timestamp, :symbol, :price, :open_price, :high, :low, :bid_price, :ask_price, :volume)
            ON DUPLICATE KEY UPDATE price = VALUES(price), volume = VALUES(volume);
        """), data)
        
        for bar in bars:
            conn.execute(text("""
                INSERT INTO market_price_history (date, symbol, open_price, close_price, high, low, volume)
                VALUES (:date, :symbol, :open_price, :close_price, :high, :low, :volume)
                ON DUPLICATE KEY UPDATE close_price = VALUES(close_price), volume = VALUES(volume);
            """), {
                'date': bar.t.date(),
                'symbol': symbol,
                'open_price': bar.o,
                'close_price': bar.c,
                'high': bar.h,
                'low': bar.l,
                'volume': bar.v
            })
        if snapshot.latest_quote:
            conn.execute(text("""
                INSERT INTO order_book (timestamp, symbol, top_bid_price, top_bid_volume, top_ask_price, top_ask_volume)
                VALUES (:timestamp, :symbol, :top_bid_price, :top_bid_volume, :top_ask_price, :top_ask_volume);
            """), {
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'top_bid_price': snapshot.latest_quote.bid_price,
                'top_bid_volume': getattr(snapshot.latest_quote, 'bid_size', 0),
                'top_ask_price': snapshot.latest_quote.ask_price,
                'top_ask_volume': getattr(snapshot.latest_quote, 'ask_size', 0)
            })
        conn.commit()
    logger.info(f"✅ {symbol} data stored.")

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(fetch_and_store, SYMBOLS)
    logger.info("🎉 Alpaca data collection complete.")
