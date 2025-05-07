# monitor.py
import time
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from common.logger import logger

# Példa metrikák
trade_counter = Counter('trades_executed', 'Number of trades executed')
pnl_gauge = Gauge('current_pnl', 'Current profit and loss')
drawdown_gauge = Gauge('current_drawdown', 'Current drawdown')

def monitor_loop():
    start_http_server(8000)  # Prometheus export port
    logger.info("✅ Monitor server elindult a 8000-es porton.")
    while True:
        # Itt lehet lekérni és frissíteni a metrikákat
        time.sleep(10)

if __name__ == "__main__":
    monitor_loop()
