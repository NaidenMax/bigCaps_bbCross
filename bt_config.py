"""Backtest configuration for BB 5-Min Cross strategy."""

import os
from datetime import date, timedelta

# ===================================================================
# POLYGON API
# ===================================================================
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "zGBzzTzmJJGiA5QzsvpUbeq85RYZL_fP")
RATE_LIMIT_SLEEP = 0.10          # seconds between API calls
MAX_RETRIES = 3

# ===================================================================
# DATE RANGE
# ===================================================================
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365)
WARMUP_DAILY_BARS = 220          # extra daily bars before START_DATE for EMA warmup
WARMUP_INTRADAY_DAYS = 23        # extra trading days for 30-min indicator warmup

# ===================================================================
# PATHS
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WATCHLIST = os.path.join(BASE_DIR, "watchlist.txt")

# ===================================================================
# STRATEGY — SCREENING (30-min Bollinger Band)
# ===================================================================
BB_PERIOD = 20
BB_STD = 2.0
SCREENING_TF = "30min"

# ===================================================================
# STRATEGY — TREND FILTER (30-min EMA crossover)
# ===================================================================
TREND_FAST_EMA = 100
TREND_SLOW_EMA = 260

# How far back (in 30-min bars) to look for a recent BULLISH cross of
# TREND_FAST_EMA up through TREND_SLOW_EMA. 130 bars ≈ 10 trading days
# (13 30-min bars/day). Used as a "fresh trend" tag on every trade and,
# optionally, as a hard entry filter (see ENABLE_RECENT_30M_CROSS_FILTER).
HAD_30M_BULLISH_CROSS_LOOKBACK_BARS = 130

# When False (default): the boolean is recorded on every trade for
# analysis only — no entries are rejected. When True: entries are
# skipped whenever no bullish 30-min EMA cross occurred inside the
# lookback window (skip_reason = "no_recent_30m_cross").
ENABLE_RECENT_30M_CROSS_FILTER = False

# ===================================================================
# STRATEGY — UPTREND FILTER (daily EMA crossover)
# ===================================================================
UPTREND_FAST_EMA = 8
UPTREND_SLOW_EMA = 21

# ===================================================================
# STRATEGY — DAILY MACD HISTOGRAM FILTER
# ===================================================================
ENABLE_DAILY_MACD_FILTER = True
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9                 # entry requires MACD_HIST > 0 on latest daily bar

# ===================================================================
# STRATEGY — INDEX MACD HISTOGRAM FILTER (benchmark)
# ===================================================================
# When enabled, NEW long entries on any stock are only allowed if the
# daily MACD histogram on INDEX_SYMBOL is > 0 on the latest completed
# daily bar. Reuses MACD_FAST / MACD_SLOW / MACD_SIGNAL above.
ENABLE_INDEX_MACD_FILTER = False
INDEX_SYMBOL = "SPY"

# ===================================================================
# STRATEGY — DAILY ROC RANK FILTER
# ===================================================================
# Master on/off for the ROC-percentile entry gate.
#   False -> no ROC filter; every signal that passes other gates fires.
#   True  -> the gate is active; the SPECIFIC logic depends on
#            ROC_RANK_MODE below.
ENABLE_ROC_RANK_FILTER = True

# Which ranking logic to use when the filter is enabled.
#   "Simple_Rank"          : (current behavior, default)
#       Per day, compute ROC(ROC_PERIOD) for every symbol in the
#       trading watchlist. Keep entries only if the symbol's ROC is in
#       the top ROC_TOP_PCT % of THAT watchlist on the entry day.
#       Universe = trading watchlist (biased but cheap).
#   "TC2000_Complex_Rank"  : (NEW, mirrors screenConditionEOD.txt)
#       Per day, against an EXTERNAL universe (e.g. Russell 1000),
#       rank ROC over each period in ROC_COMPLEX_PERIODS, mark a
#       symbol "hot" if it sits in the top ROC_COMPLEX_TOP_PCT % on
#       ANY of those periods, and allow entry whenever the symbol was
#       hot at least once within the trailing ROC_COMPLEX_LOOKBACK_DAYS
#       days. Additionally enforces per-symbol gates (avg volume, min
#       price, ATR/price, MACD positive). The trading symbol does NOT
#       need to be in the universe -- its own ROC is just compared
#       against the universe-wide cutoffs.
ROC_RANK_MODE = "TC2000_Complex_Rank"   # "Simple_Rank" | "TC2000_Complex_Rank"

assert ROC_RANK_MODE in ("Simple_Rank", "TC2000_Complex_Rank"), \
    f"ROC_RANK_MODE must be 'Simple_Rank' or 'TC2000_Complex_Rank', got {ROC_RANK_MODE!r}"

# -------------------------------------------------------------------
# Simple_Rank knobs (ROC_RANK_MODE == "Simple_Rank")
# -------------------------------------------------------------------
ROC_PERIOD = 60                 # daily lookback for Rate of Change
ROC_TOP_PCT = 75.0              # only allow entries on the top X% of watchlist by daily ROC

# -------------------------------------------------------------------
# TC2000_Complex_Rank knobs (ROC_RANK_MODE == "TC2000_Complex_Rank")
# Defaults mirror screenConditionEOD.txt.
# -------------------------------------------------------------------
ROC_COMPLEX_PERIODS = [5, 10, 30]   # ranked independently; OR'd
ROC_COMPLEX_TOP_PCT = 10         # top X% per ROC period (TC2000: 100 - 94.59)
ROC_COMPLEX_LOOKBACK_DAYS = 10      # "Within Bars 20" -- hot ANY day in last N

# Universe file (one ticker per line; '#' or blank lines ignored).
# The downloader fetches daily-only bars for any symbol here that
# isn't already in WATCHLIST. R1000 implicitly satisfies cap > $2B
# so we don't need historical shares-outstanding data.
ROC_COMPLEX_UNIVERSE_FILE = os.path.join(BASE_DIR, "watchlist R1000.txt")

# Per-symbol gates from the TC2000 screen, evaluated on the entry
# day's most recent completed daily bar. Set to None / False to
# disable an individual sub-check.
ROC_COMPLEX_REQUIRE_AVGV10     = 300_000   # rolling mean(volume, 10)
ROC_COMPLEX_REQUIRE_MIN_PRICE  = 3.0       # close
ROC_COMPLEX_REQUIRE_ATR10_PCT  = 3.0       # ATR(10) / close * 100
ROC_COMPLEX_REQUIRE_ATR25_PCT  = 3.0       # ATR(25) / close * 100
ROC_COMPLEX_REQUIRE_MACD_POS   = True      # daily MACD_HIST > 0
                                            # (overlaps with ENABLE_DAILY_MACD_FILTER)

# ===================================================================
# STRATEGY — DAILY ATR
# ===================================================================
ATR_PERIOD = 7

# ===================================================================
# STRATEGY — VOLATILITY REGIME AT ENTRY (audit + bracket analysis)
# ===================================================================
# 30m Bollinger Band width is a direct volatility proxy. We persist
# two scale-free ratios on every signal / trade for the analysis HTML:
#   bb_squeeze_ratio_30m = current 30m BB width % / trailing-mean BB width %
#   bb_vs_atr_ratio      = current 30m BB width % / daily ATR %
# Lookback for the trailing mean is in days; converted internally to
# 30-min bars via BB_WIDTH_BARS_PER_DAY (13 bars in a regular 09:30->16:00
# session). Audit-only: no entry filter, just two new bracket sections in
# analysis.html plus columns in signals.csv / trades.csv.
BB_WIDTH_AVG_LOOKBACK_DAYS = 10
BB_WIDTH_BARS_PER_DAY      = 13

# ===================================================================
# STRATEGY — ENTRY
# ===================================================================
ENTRY_TF = "5min"
ENTRY_TIME_AFTER = "10:00"       # HH:MM — reject entries at/before this clock time
ENTRY_TIME_BEFORE = "14:00"       # HH:MM — reject entries after this clock time

# Entry cross detection mode:
#   "same_bar"   -> strict: 5m bar OPENS below 30m lower BB AND CLOSES above
#                   it (single-bar traversal of the band).
#   "prev_close" -> classic crossover: PREV 5m close < prev lower BB AND
#                   CURRENT 5m close > current lower BB (catches crosses
#                   spanning two 5m bars and 30m band recomputes).
ENTRY_CROSS_MODE = "prev_close"

# ---- Additional pullback filter (stacks on top of TREND_FAST_EMA>TREND_SLOW_EMA) ----
ENABLE_BELOW_30M_EMA = False     # require bar_close <= EMA(BELOW_30M_EMA_PERIOD) on 30-min
BELOW_30M_EMA_PERIOD = 72      # must equal TREND_FAST_EMA (reuses that column)

# ===================================================================
# STRATEGY — EXIT: STOP LOSS (per-trade, OFF by default in accumulation mode)
# ===================================================================
ENABLE_STOP = False
STOP_TYPE = "session_low"        # session_low = LOD - offset
STOP_OFFSET = 0.01

# ===================================================================
# STRATEGY — EXIT: TAKE PROFIT
# ===================================================================
ENABLE_TARGET = True
TARGET_TYPE = "swing_high"       # highest 30-min high over N bars
TARGET_LOOKBACK = 130            # number of 30-min bars

# ===================================================================
# STRATEGY — EXIT: END-OF-DAY / DAILY EMA
# ===================================================================
ENABLE_EOD_FLATTEN = False       # flatten all open positions at last 5-min bar of day
ENABLE_EMA_EXIT = True           # flatten a symbol when its completed daily bar closes below EMA
EMA_EXIT_PERIOD = 21             # EMA period used for the daily-close exit (single-leg / fallback)

# ===================================================================
# STRATEGY — TRIM AND TRAIL (multi-leg exit)
# ===================================================================
# When enabled, each entry is split into 3 sub-positions ("legs") sharing a
# common parent_id:
#   Leg 1 -> exit at swing-high target (existing 130 30-min bar lookback)
#   Leg 2 -> exit at 127.2% Fib extension of the same lookback's high-low
#   Leg 3 -> trail; exit when EOD daily close < EMA(period)
# When Leg 1 fills, sibling stops are moved to breakeven and the EMA trail
# tightens from PRE_LEG1 (21) to POST_LEG1 (8) to lock in profit.
# When the Fib target can't be computed (warmup or fib <= swing_high), Leg 2's
# shares are folded into Leg 1 (sell at swing high) so total share count is
# preserved.
ENABLE_TRIM_AND_TRAIL = True

# Leg sizes — fractions of total share count for one entry.
# Should sum to 1.0; the rounding remainder always goes to Leg 3.
LEG1_PCT = 0.33
LEG2_PCT = 0.33
LEG3_PCT = 0.34

# Leg 2 target = swing_low + EXT * (swing_high - swing_low).
# Same TARGET_LOOKBACK 30-min window as Leg 1 -> fib_target > swing_high by construction.
LEG2_FIB_EXTENSION = 1.272

# Behaviour after Leg 1 fills
MOVE_STOPS_TO_BE_AFTER_LEG1 = False
EMA_EXIT_PERIOD_PRE_LEG1  = 21
EMA_EXIT_PERIOD_POST_LEG1 = 8

# ===================================================================
# SHORTLISTING (pre-filter bars to skip symbols unlikely to trigger)
# ===================================================================
ENABLE_SHORTLIST = True
SHORTLIST_REQUIRE_RED = True     # skip bars where bar_close >= day_open (stock is green)
SHORTLIST_MAX_DIST_PCT = 5.0     # skip bars where bar_low > lower_bb * (1 + pct/100)

# ===================================================================
# POSITION SIZING
# ===================================================================
INITIAL_CAPITAL = 100_000
SIZING_TYPE = "atr_risk"         # atr_risk | fixed_dollar | percent_equity
SIZING_AMOUNT = 500.0            # used only when SIZING_TYPE != "atr_risk"
RISK_PCT_PER_TRADE = 0.3        # % of equity risked per entry (atr_risk mode)
ATR_RISK_MULTIPLIER = 2.0        # risk-per-share = multiplier * ATR(period)
ATR_RISK_PERIOD = 7              # daily ATR lookback for sizing
MAX_POSITIONS = 100000               # total open sub-positions across all symbols 
MAX_ENTRIES_PER_SYMBOL_PER_DAY = 10
ENTRY_COOLDOWN_MINUTES = 5      # min minutes between entries on same symbol
USE_COMPOUNDING = False

# ===================================================================
# COMMISSION
# ===================================================================
COMMISSION_PER_SHARE = 0.003

# ===================================================================
# PARALLEL EXECUTION
# ===================================================================
# Run the backtest with one worker process per symbol.  Set to 1 to
# force the legacy single-process path (useful for debugging or for
# byte-identical parity comparisons).  Auto-falls back to 1 (with a
# warning) when USE_COMPOUNDING=True, because compounded sizing
# depends on the cross-symbol equity curve and cannot be parallelised
# per-symbol without behavioural drift.
import os as _os
NUM_WORKERS = max(1, (_os.cpu_count() or 2) - 1)

# ===================================================================
# TRADE CHARTS (trade_charts.html)
# ===================================================================
# Controls which parent entries get rendered as interactive charts.
# Each card shows 30-min candles (BB + per-leg entry/exit markers) stacked
# on daily candles with the EMA trail that the engine actually used.
TRADE_CHARTS_RECENT_DAYS = 30      # always include entries from the last N calendar days
TRADE_CHARTS_TOP_N = 5             # plus top-N winners (by parent % return) from older trades
TRADE_CHARTS_BOTTOM_N = 5          # plus bottom-N losers (by parent % return) from older trades
TRADE_CHARTS_CONTEXT_PAD_DAYS = 5  # calendar-day pad before entry / after last exit
SHOW_CLOSED_TRADES_ONLY = False     # True = only render parents whose every leg has exited
TRADE_CHARTS_FILENAME = "trade_charts.html"

# ===================================================================
# REPORT — what-if target sweep (analysis.html)
# ===================================================================
# Multipliers applied to the swing-high target to ask: "what if the target had
# been entry_price * (1 + (m - 1) * realised_range)?"  Each trade's MFE is
# used as a proxy for whether the level was reached.
WHATIF_TARGET_MULTIPLIERS = [1.0, 1.1, 1.2, 1.272, 1.5, 1.618, 2.0]
