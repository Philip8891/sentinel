from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from common.config import MLConfig

def build_model():
    # Frissítjük a modell paramétereket, hogy a LightGBM ne írjon verbose kimenetet.
    lgbm_params = MLConfig.MODEL_PARAMS.get('lgbm', {}).copy()
    lgbm_params['verbose'] = -1  # kikapcsolja a LightGBM figyelmeztetéseit

    return StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(**MLConfig.MODEL_PARAMS.get('xgb', {}))),
            ('lgbm', LGBMClassifier(**lgbm_params))
        ],
        final_estimator=LogisticRegression(),
        stack_method='predict_proba'
    )
