# execution/trading_bot.py
from common.logger import logger

def execute_trade(trade_signal, symbol, position_size, stop_loss):
    """
    Végrehajtja a tranzakciót a bróker API segítségével.
    trade_signal: "long" vagy "short"
    """
    if trade_signal == "long":
        logger.info(f"[Trading] Long pozíció nyitása: {symbol}, Méret: {position_size}, Stop Loss: {stop_loss}")
    elif trade_signal == "short":
        logger.info(f"[Trading] Short pozíció nyitása: {symbol}, Méret: {position_size}, Stop Loss: {stop_loss}")
    else:
        logger.info(f"[Trading] Nincs kereskedési jelzés: {symbol}")

def main():
    # Példa hívás statikus értékekkel
    execute_trade("long", "AAPL", 2000, 145.0)

if __name__ == "__main__":
    main()
