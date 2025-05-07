#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from my_project.common.logger import logger
from my_project.utils.db_connector import get_db_engine
from my_project.data_processing.indicators import calculate_ta_indicators_by_symbol, save_to_db

def integrate_data():
    logger.info("🔄 Piaci adatok integrálása elkezdődött.")

    try:
        engine = get_db_engine()
        # A query-ben átnevezzük az oszlopokat
        query = """
        SELECT 
            timestamp,
            symbol,
            price AS close, 
            open_price AS open,
            high,
            low,
            volume
        FROM market_data
        """
        df_market = pd.read_sql(query, engine)
        logger.info("✅ Piaci adatok sikeresen betöltve az adatbázisból.")
    except Exception as e:
        logger.error(f"❌ Hiba a piaci adatok betöltésénél: {e}")
        return

    if df_market.empty:
        logger.error("❌ A 'market_data' tábla üres, nincs mit feldolgozni.")
        return

    # Meghívjuk a szimbólumonkénti indikátor-számítást
    try:
        df_with_indicators = calculate_ta_indicators_by_symbol(df_market)
    except Exception as e:
        logger.error(f"❌ Hiba az indikátorok számításánál: {e}")
        return

    # Az indikátorokat mentjük a technical_indicators táblába
    try:
        save_to_db(df_with_indicators)
    except Exception as e:
        logger.error(f"❌ Hiba az eredmények mentésénél: {e}")
        return

    logger.info("✅ Adatintegráció sikeresen befejezve.")

if __name__ == "__main__":
    integrate_data()
