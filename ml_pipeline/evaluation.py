# ml_pipeline/evaluation.py
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_model(y_test, preds, probs, returns):
    report = classification_report(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    details = f"Classification Report:\n{report}\nROC-AUC: {roc_auc:.4f}\n"
    print(details)
    
    strategy_returns = (preds == 1) * returns
    cumulative = (1 + strategy_returns).cumprod()
    
    plt.figure(figsize=(10,5))
    plt.plot(cumulative, label="Stratégia Hozam")
    plt.title("Kumulatív Hozam")
    plt.legend()
    plt.show()
    
    return details
