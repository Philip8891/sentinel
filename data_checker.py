# data_checker.py
import pandas as pd
from common.db import get_sync_engine
from common.config import Config
from common.logger import logger

def query_table(engine, table_name):
    query = f"SELECT * FROM {table_name}"
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        logger.error(f"Hiba a {table_name} tábla lekérdezése során: {e}")
        return pd.DataFrame()

def main():
    engines = {
        "Alpaca": get_sync_engine(Config.ALPACA_DB),
        "Kraken": get_sync_engine(Config.KRAKEN_DB),
        "Market News": get_sync_engine(Config.NEWS_DB),
        "Integrated": get_sync_engine(Config.ALPACA_DB)  # Feltételezzük, integrált adatok itt vannak
    }
    
    tables = {
        "Alpaca": ["market_data", "market_price_history", "order_book"],
        "Kraken": ["market_data", "market_price_history", "order_book"],
        "Market News": ["market_news"],
        "Integrated": ["integrated_market_data", "trading_predictions", "symbols"]
    }
    
    for source, engine in engines.items():
        print(f"\n--- {source} Adatbázis ---")
        for table in tables[source]:
            print(f"\nTáblanév: {table}")
            df = query_table(engine, table)
            if df.empty:
                print("Nincs adat.")
            else:
                print(df.head(10).to_string(index=False))
                
if __name__ == "__main__":
    main()
