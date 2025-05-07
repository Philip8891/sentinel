import subprocess
import time
from datetime import datetime, time as dtime
from apscheduler.schedulers.blocking import BlockingScheduler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import sys, os
import pandas as pd
import pytz
from common.db import get_sync_engine
from common.config import Config
from ml_pipeline.predictive_model import make_prediction
from trade_executor.alpaca_trade_executor import execute_trade
from data_processing.data_integrator import integrate_data
from utils.db_connector import load_market_data, load_news_data, calculate_dynamic_qty
from common.logger import logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
console = Console()

REQUIRED_ALPACA_COUNT = 100
REQUIRED_KRAKEN_COUNT = 50
REQUIRED_NEWS_COUNT = 50
REQUIRED_INTEGRATED_COUNT = 100

def in_stock_trading_hours():
    amsterdam_tz = pytz.timezone("Europe/Amsterdam")
    now_amsterdam = datetime.now(amsterdam_tz)
    ny_tz = pytz.timezone("America/New_York")
    now_ny = now_amsterdam.astimezone(ny_tz).time()
    return dtime(15, 30) <= now_ny <= dtime(22, 0)

def check_data_availability():
    availability = {}
    try:
        engine = get_sync_engine(Config.ALPACA_DB)
        df = pd.read_sql("SELECT COUNT(*) as cnt FROM market_data", engine)
        availability['alpaca'] = df['cnt'].iloc[0]
    except Exception:
        availability['alpaca'] = 0
    try:
        engine = get_sync_engine(Config.KRAKEN_DB)
        df = pd.read_sql("SELECT COUNT(*) as cnt FROM market_data", engine)
        availability['kraken'] = df['cnt'].iloc[0]
    except Exception:
        availability['kraken'] = 0
    try:
        engine = get_sync_engine(Config.NEWS_DB)
        df = pd.read_sql("SELECT COUNT(*) as cnt FROM market_news", engine)
        availability['news'] = df['cnt'].iloc[0]
    except Exception:
        availability['news'] = 0
    try:
        engine = get_sync_engine(Config.ALPACA_DB)
        df = pd.read_sql("SELECT COUNT(*) as cnt FROM integrated_market_data", engine)
        availability['integrated'] = df['cnt'].iloc[0]
    except Exception:
        availability['integrated'] = 0
    return availability

def run_predictions_and_trades():
    df_market = load_market_data()
    df_news = load_news_data()
    df_integrated = integrate_data(df_market, df_news)
    predictions = make_prediction(df_integrated)
    for _, row in predictions.iterrows():
        if row['confidence'] > 0.75:
            qty = calculate_dynamic_qty(row['symbol'])
            order = execute_trade(row['symbol'], qty, row['prediction'])
            logger.info(f"Executed trade for {row['symbol']} with order: {order}")
        else:
            logger.info(f"Trade not executed for {row['symbol']} due to low confidence ({row['confidence']})")

def run_all_tasks():
    console.print(Panel("\U0001F4A1 Checking data availability...", style="bold blue"))
    availability = check_data_availability()
    console.print(f"Data counts: {availability}")

    wait_time = 60
    max_wait_iterations = 5
    iterations = 0

    while ((in_stock_trading_hours() and availability.get('alpaca', 0) < REQUIRED_ALPACA_COUNT) or
           (availability.get('kraken', 0) < REQUIRED_KRAKEN_COUNT) or
           (availability.get('news', 0) < REQUIRED_NEWS_COUNT) or
           (availability.get('integrated', 0) < REQUIRED_INTEGRATED_COUNT)) and iterations < max_wait_iterations:
        missing_info = []
        if in_stock_trading_hours() and availability.get('alpaca', 0) < REQUIRED_ALPACA_COUNT:
            missing_info.append(f"Alpaca: {REQUIRED_ALPACA_COUNT - availability.get('alpaca', 0)} missing")
        if availability.get('kraken', 0) < REQUIRED_KRAKEN_COUNT:
            missing_info.append(f"Kraken: {REQUIRED_KRAKEN_COUNT - availability.get('kraken', 0)} missing")
        if availability.get('news', 0) < REQUIRED_NEWS_COUNT:
            missing_info.append(f"News: {REQUIRED_NEWS_COUNT - availability.get('news', 0)} missing")
        if availability.get('integrated', 0) < REQUIRED_INTEGRATED_COUNT:
            missing_info.append(f"Integrated: {REQUIRED_INTEGRATED_COUNT - availability.get('integrated', 0)} missing")
        remaining_time = (max_wait_iterations - iterations) * wait_time
        console.print(f"[yellow]Missing data: {', '.join(missing_info)}. Waiting {remaining_time} more seconds.[/yellow]")
        time.sleep(wait_time)
        availability = check_data_availability()
        iterations += 1
        console.print(f"Updated data counts: {availability}")

    if in_stock_trading_hours() and availability.get('alpaca', 0) < REQUIRED_ALPACA_COUNT:
        console.print("[red]Not enough Alpaca data during trading hours. Tasks will not run.[/red]")
        return
    if (availability.get('kraken', 0) < REQUIRED_KRAKEN_COUNT or
        availability.get('news', 0) < REQUIRED_NEWS_COUNT or
        availability.get('integrated', 0) < REQUIRED_INTEGRATED_COUNT):
        console.print("[red]Not enough data from Kraken, News, or Integrated sources. Tasks will not run.[/red]")
        return

    console.print(Panel("✅ Sufficient data available. Running tasks...", style="bold green"))

    tasks = [
        ("🔵 Alpaca Data Check", ["python", "data_collection/alpaca_data_check.py"], True),
        ("🔵 Alpaca Data Collection", ["python", "data_collection/alpaca_data.py"], True),
        ("🔵 Kraken Data Collection", ["python", "data_collection/kraken_data.py"], False),
        ("🔵 News Fetching", ["python", "data_collection/news_fetcher.py"], False),
        ("🔵 Data Integration & Indicators", ["python", "data_processing/data_integrator.py"], False),
        ("🔵 ML Pipeline Training & Evaluation", ["python", "ml_pipeline/train.py"], False),
    ]

    results = []
    console.print(Panel("\U0001F4A1 Running tasks...", style="bold blue"))
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    for name, cmd, trading_only in tasks:
        if trading_only and not in_stock_trading_hours():
            message = f"ℹ️ {name}: Not trading hours, skipping."
            console.print(f"[yellow]{message}[/yellow]")
            results.append((name, "ℹ️ Skipped", "-", message))
            continue
        start_time = time.time()
        try:
            console.print(f"▶️  [bold]{name}[/bold] starting...")
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
            duration = time.time() - start_time
            output = completed.stdout.strip() if completed.stdout.strip() else "No details."
            results.append((name, "✅ Success", f"{duration:.2f} s", output))
            console.print(f"[green]✅ {name} succeeded! (Duration: {duration:.2f} s)[/green]")
            console.print(f"[green]Output: {output}[/green]")
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            err_output = e.stderr.strip() if e.stderr.strip() else "No error message."
            results.append((name, "❌ Error", f"{duration:.2f} s", err_output))
            console.print(f"[red]❌ Error occurred in {name}! (Duration: {duration:.2f} s)[/red]")
            console.print(f"[red]Return Code: {e.returncode}[/red]")
            console.print(f"[red]Error: {err_output}[/red]")

    table = Table(title="Central Scheduler - Task Status")
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Duration", justify="right", style="green")
    table.add_column("Details", style="white")
    for name, status, duration, details in results:
        table.add_row(name, status, duration, details)
    console.print(table)

    run_predictions_and_trades()

def run_scheduler():
    scheduler = BlockingScheduler()
    scheduler.add_job(run_all_tasks, 'interval', minutes=5)
    console.print(Panel("⏰ Scheduler started. Tasks run every 5 minutes.", style="bold blue"))
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        console.print(Panel("⏹️ Scheduler shutting down...", style="bold yellow"))

if __name__ == "__main__":
    run_all_tasks()
    run_scheduler()
