import pytz
from datetime import datetime, time

def in_stock_trading_hours():
    amsterdam_tz = pytz.timezone("Europe/Amsterdam")
    now_amsterdam = datetime.now(amsterdam_tz)
    ny_tz = pytz.timezone("America/New_York")
    now_ny = now_amsterdam.astimezone(ny_tz).time()
    return time(15, 30) <= now_ny <= time(22, 0)
