"""Orchestrator: download -> backtest -> report -> charts -> portfolio.

Edit MODE below and run `python bt_run.py` (no arguments needed).
A CLI arg, if provided, overrides MODE for that one invocation.
"""

import sys

# ===================================================================
# WHICH STAGE(S) TO RUN
# ===================================================================
# Options:
#   "download"  - Refresh parquet data only (slow; ~25 min for 473 syms)
#   "backtest"  - Run the simulation against existing parquet data
#                 (writes trades.csv, signals.csv, etc., but does NOT
#                 update the HTML reports)
#   "report"    - Rebuild analysis.html from existing trades.csv
#   "charts"    - Rebuild trade_charts.html from existing trades.csv
#   "portfolio" - Rebuild current_portfolio.html from existing trades.csv
#   "rerun"     - backtest + report + charts + portfolio (skips download).
#                 Use this when iterating on config: refreshes everything
#                 you need to see, never re-downloads.
#   "full"      - download -> backtest -> report -> charts -> portfolio
MODE = "full"


def _dispatch():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else MODE

    if mode in ("download", "full"):
        from bt_download import main as dl_main
        dl_main()

    if mode in ("backtest", "rerun", "full"):
        from bt_backtest import main as bt_main
        bt_main()

    if mode in ("report", "rerun", "full"):
        from bt_report import main as rpt_main
        rpt_main()

    if mode in ("charts", "rerun", "full"):
        from bt_trade_charts import main as charts_main
        charts_main()

    if mode in ("portfolio", "rerun", "full"):
        from bt_current_portfolio import main as port_main
        port_main()


# Required for Windows multiprocessing (spawn) used by bt_backtest's
# ProcessPoolExecutor: workers re-import this module on spawn, and
# without the guard they would re-execute the dispatch and fork-bomb.
if __name__ == "__main__":
    _dispatch()
