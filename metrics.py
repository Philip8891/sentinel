from prometheus_client import start_http_server, Gauge
import time

pnl_gauge = Gauge('current_pnl', 'Current profit and loss')
drawdown_gauge = Gauge('current_drawdown', 'Current drawdown')

def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"Metrics server started on port {port}")

def update_metrics(pnl, drawdown):
    pnl_gauge.set(pnl)
    drawdown_gauge.set(drawdown)

if __name__ == "__main__":
    start_metrics_server()
    while True:
        update_metrics(1000, 50)
        time.sleep(10)
