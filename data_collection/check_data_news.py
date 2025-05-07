#!/usr/bin/env python3
import sys
import os
from rich.console import Console
from rich.table import Table

# Projekt gyökérkönyvtárának meghatározása és hozzáadása a sys.path-hez
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug: győződj meg róla, hogy a projekt gyökér megjelenik a sys.path-ben
#print("Project root:", project_root)
#print("sys.path:", sys.path)

from my_project.common.config import Config
from my_project.utils.db_connector import get_db_engine

console = Console()

def main():
    engine = get_db_engine(Config.SQLALCHEMY_DATABASE_URI)
    with engine.connect() as conn:
        result = conn.execute(
            "SELECT id, symbol, source, title, published FROM market_news ORDER BY published DESC"
        )
        rows = result.fetchall()

    table = Table(title="Letöltött hírek")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Szimbólum", justify="center", style="magenta")
    table.add_column("Forrás", justify="center", style="green")
    table.add_column("Cím", justify="left", style="yellow")
    table.add_column("Publikálva", justify="center", style="blue")

    for row in rows:
        table.add_row(str(row[0]), row[1], row[2], row[3], str(row[4]))

    console.print(table)

if __name__ == "__main__":
    main()
