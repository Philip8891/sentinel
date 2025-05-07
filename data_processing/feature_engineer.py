# data_processing/feature_engineer.py
from ta import add_all_ta_features

def engineer_features(df):
    # A TA csomag segítségével technikai indikátorok hozzáadása
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    # További egyedi feature-ek, ha szükséges
    return df
