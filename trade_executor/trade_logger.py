import os
import time
import logging
from datetime import datetime
from sqlalchemy import text
from utils.db_connector import get_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_trade_logs_table():
    """
    Létrehozza a trade_logs táblát, ha még nem létezik.
    A tábla tartalmazza a kereskedések részletes adatait: szimbólum, időbélyeg,
    megbízás típusa, mennyiség, előrejelzés, bizalmi szint, és a végrehajtás okát.
    """
    engine = get_db_engine()
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS trade_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(20),
        order_timestamp TIMESTAMP,
        order_side VARCHAR(10),
        quantity FLOAT,
        prediction VARCHAR(20),
        confidence FLOAT,
        reason TEXT,
        additional_info TEXT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    logging.info("Trade logs table ensured.")

def log_trade(trade_details):
    """
    Ment egy végrehajtott kereskedést a trade_logs táblába.
    
    Paraméter:
      - trade_details (dict): Tartalmazza az alábbi kulcsokat:
          • symbol: A kereskedési szimbólum (pl. "AAPL", "XBTUSD").
          • order_timestamp: A megbízás végrehajtásának időpontja (datetime objektum).
          • order_side: A megbízás iránya ("buy" vagy "sell").
          • quantity: A megbízás mennyisége.
          • prediction: Az ML modell által generált előrejelzés, amely alapján a megbízás történt.
          • confidence: Az előrejelzés bizalmi értéke.
          • reason: Szöveges magyarázat arra, miért került végrehajtásra a megbízás.
          • additional_info: (Opcionális) Egyéb részletek, pl. megbízás azonosító, hibaüzenet, stb.
    """
    engine = get_db_engine()
    insert_sql = """
    INSERT INTO trade_logs (symbol, order_timestamp, order_side, quantity, prediction, confidence, reason, additional_info)
    VALUES (:symbol, :order_timestamp, :order_side, :quantity, :prediction, :confidence, :reason, :additional_info)
    """
    with engine.connect() as conn:
        conn.execute(text(insert_sql), trade_details)
        conn.commit()
    logging.info(f"Trade logged for symbol {trade_details.get('symbol')} at {trade_details.get('order_timestamp')}.")

if __name__ == "__main__":
    # Példa: ha a bot végrehajt egy megbízást, akkor ezt a logoló függvényt hívjuk meg a részletekkel.
    create_trade_logs_table()
    sample_trade = {
        "symbol": "XBTUSD",
        "order_timestamp": datetime.now(),
        "order_side": "buy",
        "quantity": 0.001,
        "prediction": "long",
        "confidence": 0.78,
        "reason": "A modell előrejelzése 0.78 bizalommal jelezte a long pozíciót, így a küszöb felett végrehajtottuk a megbízást.",
        "additional_info": "Megbízás sikeresen végrehajtva a Kraken API segítségével."
    }
    log_trade(sample_trade)
