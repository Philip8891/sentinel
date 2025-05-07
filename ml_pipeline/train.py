# ml_pipeline/train.py
import pandas as pd
import io
import sys
import collections
from sklearn.model_selection import TimeSeriesSplit
from common.config import Config, MLConfig
from common.db import get_sync_engine
from common.logger import logger
from ml_pipeline.features import add_features
from ml_pipeline.model import build_model
from ml_pipeline.preprocessing import preprocess_data
from ml_pipeline.evaluation import evaluate_model

def main():
    engine = get_sync_engine(Config.ALPACA_DB)  # Az integrált adatokat tartalmazza az Alpaca adatbázis
    raw_data = pd.read_sql("SELECT * FROM integrated_market_data", engine)
    
    if raw_data.empty:
        logger.error("❌ Nincs adat a tréninghez.")
        return
    
    missing = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in raw_data.columns]
    if missing:
        logger.error(f"❌ Hiányzó oszlopok a tréning adatokból: {missing}")
        return
    else:
        logger.info("✅ Minden szükséges oszlop megtalálható a tréning adatokban.")
    
    # Ha hiányzik a 'target' oszlop, számítsuk ki: 1, ha a következő záróár magasabb, egyébként 0.
    if 'target' not in raw_data.columns:
        raw_data['target'] = (raw_data['close'].shift(-1) > raw_data['close']).astype(int)
    
    raw_data = add_features(raw_data)
    logger.info(f"✅ Technikai indikátorok számítása kész. Összes rekord: {len(raw_data)}")
    
    required_count = 100
    if len(raw_data) < required_count:
        logger.warning(f"⚠️ Csak {len(raw_data)} rekord áll rendelkezésre, míg {required_count} szükséges.")
    else:
        logger.info(f"✅ Elég adat áll rendelkezésre az indikátor számításhoz ({len(raw_data)} rekord).")
    
    tscv = TimeSeriesSplit(n_splits=MLConfig.TRAINING_SPLITS)
    train_idx, test_idx = next(tscv.split(raw_data))
    train, test = raw_data.iloc[train_idx], raw_data.iloc[test_idx]
    
    # Csak numerikus jellemzők kiválasztása (timestamp, symbol, target kivételével)
    features = [col for col in train.columns if col not in ['timestamp', 'symbol', 'target']]
    numeric_features = train[features].select_dtypes(include=["number"]).columns.tolist()
    
    X_train, y_train = train[numeric_features], train['target']
    X_test, y_test = test[numeric_features], test['target']
    
    X_train, X_test = preprocess_data(X_train, X_test, MLConfig.PCA_COMPONENTS)
    
    model = build_model()
    
    # Capture a LightGBM hiba/figyelmeztetéseket (stderr kimenetét)
    stderr_capture = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture
    try:
        model.fit(X_train, y_train)
    finally:
        sys.stderr = old_stderr
    warnings_text = stderr_capture.getvalue()
    # Csak azokat a sorokat nézzük, amelyek a [LightGBM] szöveget tartalmazzák
    lgbm_lines = [line for line in warnings_text.splitlines() if "[LightGBM]" in line]
    if lgbm_lines:
        counter = collections.Counter(lgbm_lines)
        for message, count in counter.items():
            logger.warning(f"[LightGBM] {message} occurred {count} times.")
    
    logger.info("🚀 Modell tréning sikeresen befejeződött.")
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    returns = test['close'].pct_change().shift(-1).fillna(0)
    
    eval_details = evaluate_model(y_test, preds, probs, returns)
    
    results = test[['timestamp', 'symbol']].copy()
    results['prediction'] = preds
    results['probability'] = probs
    results.to_sql('trading_predictions', engine, if_exists='append', index=False)
    
    logger.info("🚀 Modellezési eredmények sikeresen mentve.")
    logger.info(f"ML Pipeline részletek:\n{eval_details}")

if __name__ == "__main__":
    main()
