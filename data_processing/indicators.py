#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
import ta
from my_project.common.logger import logger
from my_project.utils.db_connector import get_db_engine

REQUIRED_COLUMNS = ['close', 'open', 'high', 'low', 'volume']

def check_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"❌ Hiányzó oszlopok: {missing}")
        return False
    return True

def calculate_ta_indicators(df, symbol=None):
    """
    Egyetlen szimbólum adataira számítja ki az indikátorokat.
    `symbol` paraméter csak a logoláshoz kell, ha elérhető.
    """
    if not check_columns(df):
        logger.error(f"❌ [{symbol}] Kötelező oszlopok hiányoznak, kihagyom.")
        return df
    
    n = len(df)
    if symbol:
        logger.info(f"ℹ️ [{symbol}] {n} sor áll rendelkezésre.")

    # RSI
    if n < 14:
        logger.info(f"ℹ️ [{symbol}] RSI-hez legalább 14 adat kell, most {n}. Kihagyom.")
    else:
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        except Exception as e:
            logger.error(f"❌ [{symbol}] RSI hiba: {e}")

    # MACD
    if n < 26:
        logger.info(f"ℹ️ [{symbol}] MACD-hez legalább 26 adat kell, most {n}. Kihagyom.")
    else:
        try:
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        except Exception as e:
            logger.error(f"❌ [{symbol}] MACD hiba: {e}")

    # Stochastic (window=14)
    if n < 14:
        logger.info(f"ℹ️ [{symbol}] Stochastic-hoz legalább 14 adat kell, most {n}. Kihagyom.")
    else:
        try:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch'] = stoch.stoch()
            df['stoch_signal'] = stoch.stoch_signal()
        except Exception as e:
            logger.error(f"❌ [{symbol}] Stochastic hiba: {e}")

    # SMA, EMA (window=20)
    if n < 20:
        logger.info(f"ℹ️ [{symbol}] SMA/EMA-hoz legalább 20 adat kell, most {n}. Kihagyom.")
    else:
        try:
            df['sma'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        except Exception as e:
            logger.error(f"❌ [{symbol}] SMA hiba: {e}")
        try:
            df['ema'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        except Exception as e:
            logger.error(f"❌ [{symbol}] EMA hiba: {e}")

    # Bollinger (window=20)
    if n < 20:
        logger.info(f"ℹ️ [{symbol}] Bollinger-hez legalább 20 adat kell, most {n}. Kihagyom.")
    else:
        try:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_hband'] = bb.bollinger_hband()
            df['bb_lband'] = bb.bollinger_lband()
            df['bb_mavg'] = bb.bollinger_mavg()
        except Exception as e:
            logger.error(f"❌ [{symbol}] Bollinger hiba: {e}")

    # ATR (window=14)
    if n < 14:
        logger.info(f"ℹ️ [{symbol}] ATR-hez legalább 14 adat kell, most {n}. Kihagyom.")
    else:
        try:
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            df['atr'] = atr.average_true_range()
        except Exception as e:
            logger.error(f"❌ [{symbol}] ATR hiba: {e}")

    # ADX (window=14)
    if n < 14:
        logger.info(f"ℹ️ [{symbol}] ADX-hez legalább 14 adat kell, most {n}. Kihagyom.")
    else:
        try:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx.adx()
        except Exception as e:
            logger.error(f"❌ [{symbol}] ADX hiba: {e}")

    # OBV (nincs minimum)
    try:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    except Exception as e:
        logger.error(f"❌ [{symbol}] OBV hiba: {e}")

    # Supertrend placeholder
    df['supertrend'] = 0.0
    df['supertrend_trend'] = 'up'

    # Alligator
    try:
        df['alligator_jaw'] = df['close'].rolling(13).mean().shift(8)
        df['alligator_teeth'] = df['close'].rolling(8).mean().shift(5)
        df['alligator_lips'] = df['close'].rolling(5).mean().shift(3)
    except Exception as e:
        logger.error(f"❌ [{symbol}] Alligator hiba: {e}")

    # Pivot
    try:
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = (2 * df['pivot']) - df['low'].shift(1)
        df['s1'] = (2 * df['pivot']) - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
    except Exception as e:
        logger.error(f"❌ [{symbol}] Pivot hiba: {e}")

    # Fibonacci
    try:
        if not df.empty:
            diff = df['high'].max() - df['low'].min()
            df['fib_0'] = df['high'].max()
            df['fib_236'] = df['fib_0'] - 0.236 * diff
            df['fib_382'] = df['fib_0'] - 0.382 * diff
            df['fib_50'] = df['fib_0'] - 0.5 * diff
            df['fib_618'] = df['fib_0'] - 0.618 * diff
            df['fib_786'] = df['fib_0'] - 0.786 * diff
            df['fib_1'] = df['low'].min()
        else:
            logger.info(f"ℹ️ [{symbol}] Fibonacci: üres adat, kihagyom.")
    except Exception as e:
        logger.error(f"❌ [{symbol}] Fibonacci hiba: {e}")

    logger.info(f"✅ [{symbol}] Indikátorok számítva (amihez volt elég adat).")
    return df

def calculate_ta_indicators_by_symbol(df_all):
    """
    Csoportosítja a df_all-t symbol szerint,
    minden csoportra meghívja a calculate_ta_indicators függvényt,
    majd összefűzi az eredményeket egy DataFrame-be.
    """
    results = []
    for symbol, df_sym in df_all.groupby('symbol'):
        logger.info(f'🌀 Feldolgozás kezdete a(z) {symbol} szimbólumra.')
        df_sym_sorted = df_sym.sort_values('timestamp').reset_index(drop=True)
        df_out = calculate_ta_indicators(df_sym_sorted, symbol=symbol)
        # visszaírjuk a symbol mezőt:
        df_out['symbol'] = symbol
        results.append(df_out)

    if results:
        df_concat = pd.concat(results, ignore_index=True)
        logger.info(f'✅ Összesen {len(results)} szimbólum feldolgozva, {len(df_concat)} sor.')
        return df_concat
    else:
        logger.warning('⚠️ Nincs feldolgozható szimbólum.')
        return pd.DataFrame()

def save_to_db(df):
    engine = get_db_engine()
    try:
        df.to_sql('technical_indicators', engine, if_exists='replace', index=False)
        logger.info("✅ Indikátorok mentve az adatbázisba.")
    except Exception as e:
        logger.error(f"❌ Indikátorok mentési hiba: {e}")

if __name__ == "__main__":
    # Teszt futtatás: csoportonként számítás
    engine = get_db_engine()
    query = """
    SELECT 
        timestamp,
        symbol,
        price AS close,
        open_price AS open,
        high, low, volume
    FROM market_data
    """
    df_all = pd.read_sql(query, engine)
    df_all_ind = calculate_ta_indicators_by_symbol(df_all)
    save_to_db(df_all_ind)
