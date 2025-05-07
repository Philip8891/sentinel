# ml_pipeline/features.py
import pandas as pd
from ta import add_all_ta_features

def add_features(df):
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            # Ha hiányzik, az open, high, low helyettesíthető a close értékkel, volume esetén 0-val
            df[col] = df['close'] if col in ['open', 'high', 'low'] else 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    return df
