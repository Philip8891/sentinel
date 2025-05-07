#!/usr/bin/env python3
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from my_project.common.config import Config

# Győződj meg róla, hogy a .env megfelelően betöltődik (a run_all.py-ban már beállítjuk a PYTHONPATH-t, így itt csak importáljuk a Config-et)

def get_db_engine(db_uri=None):
    """
    Ha nem adunk meg kapcsolat stringet, alapértelmezésként a TRADING_DB_URI-t használjuk,
    mert az tartalmazza a market_data, integrated_market_data stb. táblákat.
    """
    if db_uri is None:
        db_uri = Config.TRADING_DB_URI
    engine = create_engine(db_uri)
    return engine

def load_market_data():
    engine = get_db_engine()  # Ez a TRADING_DB_URI-t fogja használni
    query = "SELECT * FROM market_data"
    df = pd.read_sql(query, engine)
    return df

def load_news_data():
    # Ha a hírek a Market News adatbázisban vannak, akkor itt Config.NEWS_DB_URI-t használd
    engine = create_engine(Config.NEWS_DB_URI)
    query = "SELECT * FROM market_news"
    df = pd.read_sql(query, engine)
    return df

def calculate_dynamic_qty(symbol):
    # Egyszerű, fix mennyiség példa
    return 100
