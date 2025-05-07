#!/usr/bin/env python3
import os
import time
import logging
from datetime import datetime
from sqlalchemy import text
from my_project.utils.db_connector import get_db_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def create_trade_logs_table():
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
    create_trade_logs_table()
    sample_trade = {
        "symbol": "XBTUSD",
        "order_timestamp": datetime.now(),
        "order_side": "buy",
        "quantity": 0.001,
        "prediction": "long",
        "confidence": 0.78,
        "reason": "Model predicted long position with confidence 0.78, executing trade.",
        "additional_info": "Order executed successfully via Kraken API."
    }
    log_trade(sample_trade)
