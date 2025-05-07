import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_db_engine():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    connection_string = f"postgresql://{user}:{password}@{host}/{db_name}"
    engine = create_engine(connection_string)
    return engine

def load_market_data():
    engine = get_db_engine()
    query = "SELECT * FROM market_data"
    df = pd.read_sql(query, engine)
    return df

def load_news_data():
    engine = get_db_engine()
    query = "SELECT * FROM market_news"
    df = pd.read_sql(query, engine)
    return df

def calculate_dynamic_qty(symbol):
    # Egyszerű példa: fix érték visszaadása
    return 100
