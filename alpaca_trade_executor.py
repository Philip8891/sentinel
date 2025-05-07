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
    order = api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='market',
        time_in_force='gtc'
    )
    return order

if __name__ == "__main__":
    result = execute_trade("AAPL", 10, "buy")
    print(result)
