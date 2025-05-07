#!/usr/bin/env python3
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

api = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    api_version='v2'
)

def execute_trade(symbol, qty, side):
    # Csak produkciós környezetben hajtsuk végre a tranzakciót
    if os.getenv("ENV", "test") != "prod":
        print("Test mode: Tranzakció végrehajtása ki van kapcsolva.")
        return None
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='gtc'
    )
    return order

if __name__ == "__main__":
    # Teszt futtatás: csak produkciós környezetben fut
    result = execute_trade("AAPL", 10, "buy")
    print(result)
