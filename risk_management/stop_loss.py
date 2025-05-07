# risk_management/stop_loss.py
def calculate_stop_loss(close, atr, multiplier=2):
    """
    Dinamikus stop loss számítás ATR alapján.
    Példa: ha a close = 150 és az ATR = 5, multiplier = 2, akkor SL = 150 - (2*5) = 140.
    """
    return close - multiplier * atr
