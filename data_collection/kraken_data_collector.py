#!/usr/bin/env python3
import os
import pytz
from datetime import datetime
import concurrent.futures
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from sqlalchemy import text
from my_project.common.logger import logger
from my_project.common.config import Config
from my_project.utils.db_connector import get_db_engine

load_dotenv()

SERVER_TIMEZONE = pytz.timezone(os.getenv("SERVER_TIMEZONE", "UTC"))

def get_kraken_data(endpoint, params):
    url = f"{os.getenv('KRAKEN_API_URL')}/{endpoint}"
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5))
def fetch_and_save(symbol, kraken_symbol):
    engine = get_db_engine()
    with engine.connect() as conn:
        try:
            # Lekérjük az adatokat a Kraken API-ból
            ticker = get_kraken_data("Ticker", {"pair": kraken_symbol})
            ohlc = get_kraken_data("OHLC", {"pair": kraken_symbol, "interval": 1})
            depth = get_kraken_data("Depth", {"pair": kraken_symbol, "count": 1})
            
            ticker_data = ticker["result"][kraken_symbol]
            ohlc_data = ohlc["result"][kraken_symbol][-1]
            depth_data = depth["result"][kraken_symbol]
            
            timestamp = datetime.now(SERVER_TIMEZONE)
            
            # Mivel a market_data táblában a szimbólum VARCHAR-ként van tárolva,
            # közvetlenül a symbol változót használjuk, nem pedig egy symbol_id-t.
            conn.execute(text("""
                INSERT INTO market_data (timestamp, symbol, price, open_price, high, low, bid_price, ask_price, volume)
                VALUES (:timestamp, :symbol, :price, :open_price, :high, :low, :bid, :ask, :volume)
                ON DUPLICATE KEY UPDATE price = VALUES(price), volume = VALUES(volume);
            """), {
                'timestamp': timestamp,
                'symbol': symbol,
                'price': float(ticker_data['c'][0]),
                'open_price': float(ohlc_data[1]),
                'high': float(ohlc_data[2]),
                'low': float(ohlc_data[3]),
                'bid': float(ticker_data['b'][0]),
                'ask': float(ticker_data['a'][0]),
                'volume': float(ohlc_data[6])
            })
            
            conn.execute(text("""
                INSERT INTO order_book (timestamp, symbol, top_bid_price, top_bid_volume, top_ask_price, top_ask_volume)
                VALUES (:timestamp, :symbol, :bid_price, :bid_volume, :ask_price, :ask_volume);
            """), {
                'timestamp': timestamp,
                'symbol': symbol,
                'bid_price': float(depth_data['bids'][0][0]),
                'bid_volume': float(depth_data['bids'][0][1]),
                'ask_price': float(depth_data['asks'][0][0]),
                'ask_volume': float(depth_data['asks'][0][1])
            })
            
            daily_ohlc = get_kraken_data("OHLC", {"pair": kraken_symbol, "interval": 1440})
            daily_candle = daily_ohlc["result"][kraken_symbol][-2]
            candle_date = datetime.fromtimestamp(daily_candle[0], SERVER_TIMEZONE).date()
            
            conn.execute(text("""
                INSERT INTO market_price_history (date, symbol, open_price, close_price, high, low, volume)
                VALUES (:date, :symbol, :open_price, :close_price, :high, :low, :volume)
                ON DUPLICATE KEY UPDATE close_price = VALUES(close_price), volume = VALUES(volume);
            """), {
                'date': candle_date,
                'symbol': symbol,
                'open_price': float(daily_candle[1]),
                'close_price': float(daily_candle[4]),
                'high': float(daily_candle[2]),
                'low': float(daily_candle[3]),
                'volume': float(daily_candle[6])
            })
            conn.commit()
            logger.info(f"✅ {symbol} Kraken data stored.")
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Error processing Kraken data for {symbol}: {e}")

def main():
    SYMBOLS = {
        "BTCUSD": "XXBTZUSD",
        "SOLUSD": "SOLUSD",
        "XRPUSD": "XXRPZUSD"
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_and_save, sym, kr_sym) for sym, kr_sym in SYMBOLS.items()]
        for future in futures:
            future.result()
    logger.info("🎉 Kraken data collection complete.")

if __name__ == "__main__":
    main()
