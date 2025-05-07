#!/usr/bin/env python3
import os
import sys
import subprocess

modules_to_run = [
    "my_project.data_collection.alpaca_data_collector",
    "my_project.data_collection.kraken_data_collector",
    "my_project.data_collection.news_fetcher",
    "my_project.data_processing.indicators",
    "my_project.data_processing.data_integrator",
    "my_project.ml_pipeline.model_trainer",
    "my_project.ml_pipeline.predictive_model",
    "my_project.trade_executor.alpaca_trade_executor",
    "my_project.trade_executor.kraken_trade_executor",
    "my_project.trade_executor.kraken_trade_logger",
    "my_project.monitoring.metrics"
]

def run_module(module_name):
    print(f"\nFuttatom: {module_name}")
    command = [sys.executable, "-m", module_name]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Sikeresen lefutott: {module_name}")
        if result.stdout.strip():
            print("Kimenet:")
            print(result.stdout)
    else:
        print(f"Hiba történt a {module_name} futtatása során!")
        print("Hibakimenet:")
        print(result.stderr)
    # A hiba ellenére folytatjuk a következő modullal.

if __name__ == '__main__':
    # A run_all.py a projekt gyökerében van, ezért a PYTHONPATH-ot a szülőkönyvtárra (pl. /root) állítjuk be
    project_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.environ["PYTHONPATH"] = project_parent

    print("----- RUN_ALL START -----")
    for module in modules_to_run:
        try:
            run_module(module)
        except Exception as e:
            print(f"Váratlan hiba a {module} futtatása során: {e}")
    print("----- RUN_ALL VÉGE -----")
