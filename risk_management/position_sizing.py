# risk_management/position_sizing.py
def calculate_position_size(probability, current_capital, risk_percentage=0.02):
    """
    Kiszámolja a pozíció méretét a kockázati százalék és a predikció valószínűsége alapján.
    Példa: Ha a kockázat 2% és az aktuális tőke 100,000 USD, akkor a maximális kockázat 2000 USD.
    A probability alapján súlyozás történik.
    """
    base_risk = current_capital * risk_percentage
    if probability < 0.6:
        multiplier = 0.5
    elif probability < 0.8:
        multiplier = 1.0
    else:
        multiplier = 1.5
    return base_risk * multiplier
