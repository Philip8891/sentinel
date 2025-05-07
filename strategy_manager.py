# strategy_manager.py
from common.logger import logger
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

def detect_market_regime(df):
    """
    Egyszerű KMeans-alapú piaci regime detektálás volatilitás alapján.
    Feltételezzük, hogy a df tartalmaz egy 'volatility' oszlopot.
    """
    if 'volatility' not in df.columns:
        df['volatility'] = df['close'].pct_change().rolling(window=14).std().fillna(0)
    
    km = KMeans(n_clusters=3, random_state=42)
    df['regime'] = km.fit_predict(df[['volatility']])
    logger.info("✅ Piaci regime detektálás kész.")
    return df

def strategy_decision(df):
    """
    Döntés a stratégia váltásról a piaci regime alapján.
    Visszaadja, hogy melyik stratégiát alkalmazza: trend_following, mean_reversion, vagy hedge.
    """
    df = detect_market_regime(df)
    regime_mode = df['regime'].mode()[0]
    if regime_mode == 0:
        decision = "trend_following"
    elif regime_mode == 1:
        decision = "mean_reversion"
    else:
        decision = "hedge"
    logger.info(f"✅ Stratégia döntés: {decision}")
    return decision

if __name__ == "__main__":
    sample_data = pd.DataFrame({'close': np.random.normal(100, 5, 200)})
    sample_data['volatility'] = sample_data['close'].pct_change().rolling(window=14).std().fillna(0)
    decision = strategy_decision(sample_data)
    print("Stratégia:", decision)
