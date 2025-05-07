#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from my_project.common.logger import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from my_project.utils.db_connector import get_db_engine

def preprocess_data(X_train, X_test, n_components):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test

def build_model():
    return StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=5)),
            ('lgbm', LGBMClassifier(n_estimators=500, learning_rate=0.01, max_depth=5, verbose=-1))
        ],
        final_estimator=LogisticRegression(),
        stack_method='predict_proba'
    )

def add_features(df):
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            df[col] = df['close'] if col in ['open', 'high', 'low'] else 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df

def train_model():
    engine = get_db_engine()
    df = pd.read_sql("SELECT * FROM integrated_market_data", engine)
    
    if df.empty:
        logger.error("❌ No data available for training.")
        return
    
    missing = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in df.columns]
    if missing:
        logger.error(f"❌ Missing columns in training data: {missing}")
        return
    else:
        logger.info("✅ All required columns are present in training data.")
    
    if 'target' not in df.columns:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = add_features(df)
    logger.info(f"✅ Features added. Total records: {len(df)}")
    
    required_count = 100
    if len(df) < required_count:
        logger.warning(f"⚠️ Only {len(df)} records available, while {required_count} are required.")
    else:
        logger.info(f"✅ Sufficient data available ({len(df)} records).")
    
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = next(tscv.split(df))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    
    features = [col for col in train.columns if col not in ['timestamp', 'symbol', 'target']]
    X_train, y_train = train[features], train['target']
    X_test, y_test = test[features], test['target']
    
    X_train, X_test = preprocess_data(X_train, X_test, n_components=10)
    
    model = build_model()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    logger.info(f"Model evaluation:\n{report}\nROC-AUC: {roc_auc:.4f}")
    
    # A modell elmentése a ml_pipeline mappába
    model_path = os.path.join(os.path.dirname(__file__), 'xgb_model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"🚀 Model training completed and saved at {model_path}")

if __name__ == "__main__":
    train_model()
