#!/usr/bin/env python3
import joblib
import pandas as pd

# Assumes the model file is saved as ml_pipeline/xgb_model.joblib
model = joblib.load('ml_pipeline/xgb_model.joblib')

def make_prediction(df):
    features = ['rsi', 'macd', 'macd_signal']
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    df['prediction'] = model.predict(df[features])
    df['confidence'] = model.predict_proba(df[features]).max(axis=1)
    return df[['symbol', 'timestamp', 'prediction', 'confidence']]

if __name__ == "__main__":
    df_test = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'symbol': ['AAPL']*5,
        'rsi': [30, 40, 50, 60, 70],
        'macd': [0.1, 0.2, 0.0, -0.1, -0.2],
        'macd_signal': [0.05, 0.15, 0.05, -0.05, -0.15]
    })
    predictions = make_prediction(df_test)
    print(predictions)
