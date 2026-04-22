"""Core backtest engine for BB 5-Min Cross strategy (accumulation mode).

Loads daily/30min/5min Parquet, computes indicators, and simulates a multi-entry
accumulation strategy:

- Each BB cross that passes filters opens its own sub-position (stop/target tracked
  independently).
- Positions persist across days (no EOD flatten by default).
- A per-symbol cooldown limits how often new entries may fire.
- A shortlist pre-filter skips bars unlikely to trigger (symbol green on the day
  or price too far from the lower BB).
- At end of day, if the symbol's daily close is below the configured EMA, all of
  that symbol's open positions are flattened at the daily close.
"""

from __future__ import annotations
import math, os
from datetime import time as dtime
import numpy as np
import pandas as pd
from tqdm import tqdm
import bt_config as cfg
from bt_download import load_watchlist

MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# ===================================================================
# DATA LOADING
# ===================================================================

def _load(symbol: str, tf: str) -> pd.DataFrame:
    path = os.path.join(cfg.DATA_DIR, f"{symbol}_{tf}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("US/Eastern")
    # Intraday bars: drop anything outside RTH.  Polygon timestamps bars at the
    # start of the period, so RTH is [09:30, 16:00) → 13 × 30-min bars, 78 × 5-min.
    # Keeping only RTH means, e.g., EMA(273) on 30-min ≈ EMA(21) on daily (273 = 13×21).
    if tf in ("30min", "5min") and "date" in df.columns and not df.empty:
        t = df["date"].dt.time
        df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].reset_index(drop=True)
    return df

# ===================================================================
# INDICATOR COMPUTATION
# ===================================================================

def _compute_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[f"EMA_{cfg.UPTREND_FAST_EMA}"] = df["close"].ewm(span=cfg.UPTREND_FAST_EMA, adjust=False).mean()
    df[f"EMA_{cfg.UPTREND_SLOW_EMA}"] = df["close"].ewm(span=cfg.UPTREND_SLOW_EMA, adjust=False).mean()
    # EMA used for the daily-close exit (may or may not equal the uptrend slow EMA)
    exit_col = f"EMA_{cfg.EMA_EXIT_PERIOD}"
    if exit_col not in df.columns:
        df[exit_col] = df["close"].ewm(span=cfg.EMA_EXIT_PERIOD, adjust=False).mean()
    # Pre/post Leg-1 EMA-trail columns for trim-and-trail.  Computed
    # unconditionally (cheap, idempotent) so the EOD exit can read each leg's
    # own period without re-running the EMA each day.
    for _p in {cfg.EMA_EXIT_PERIOD_PRE_LEG1, cfg.EMA_EXIT_PERIOD_POST_LEG1}:
        _col = f"EMA_{_p}"
        if _col not in df.columns:
            df[_col] = df["close"].ewm(span=_p, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df[f"ATR_{cfg.ATR_PERIOD}"] = tr.ewm(alpha=1/cfg.ATR_PERIOD, adjust=False).mean()
    # Separate ATR column used for risk-based position sizing (typically a faster period)
    risk_col = f"ATR_{cfg.ATR_RISK_PERIOD}"
    if risk_col not in df.columns:
        df[risk_col] = tr.ewm(alpha=1/cfg.ATR_RISK_PERIOD, adjust=False).mean()
    # MACD (daily)
    fast_ema = df["close"].ewm(span=cfg.MACD_FAST, adjust=False).mean()
    slow_ema = df["close"].ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    df["MACD"] = fast_ema - slow_ema
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    # ROC (daily, %)
    df[f"ROC_{cfg.ROC_PERIOD}"] = (df["close"] / df["close"].shift(cfg.ROC_PERIOD) - 1) * 100
    # Complex_Rank extras -- only when that mode is active. Cheap, but
    # kept conditional so Simple_Rank backtests stay byte-identical.
    if getattr(cfg, "ROC_RANK_MODE", "Simple_Rank") == "TC2000_Complex_Rank":
        for _p in cfg.ROC_COMPLEX_PERIODS:
            _col = f"ROC_{_p}"
            if _col not in df.columns:
                df[_col] = (df["close"] / df["close"].shift(_p) - 1) * 100
        if "AVGV10" not in df.columns and "volume" in df.columns:
            df["AVGV10"] = df["volume"].rolling(10).mean()
        for _p in (10, 25):
            _col = f"ATR_{_p}"
            if _col not in df.columns:
                df[_col] = tr.ewm(alpha=1/_p, adjust=False).mean()
    return df


def _compute_30min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # NO-LOOK-AHEAD CONTRACT: every derived column below is shifted by 1 bar so
    # that row k holds the indicator value as it was KNOWN AT THE START of bar
    # k (= as of the END of bar k-1, the most recent fully closed 30-min bar).
    # This pairs with the worker's `idx_30m_arr = searchsorted(side='left') - 1`
    # mapping so an intraday 5-min bar at time t reads only closed-bar values.
    df[f"EMA_{cfg.TREND_FAST_EMA}"] = df["close"].ewm(span=cfg.TREND_FAST_EMA, adjust=False).mean().shift(1)
    df[f"EMA_{cfg.TREND_SLOW_EMA}"] = df["close"].ewm(span=cfg.TREND_SLOW_EMA, adjust=False).mean().shift(1)
    ma  = df["close"].rolling(cfg.BB_PERIOD).mean()
    std = df["close"].rolling(cfg.BB_PERIOD).std()
    df["BB_LOWER"] = (ma - cfg.BB_STD * std).shift(1)
    df["BB_UPPER"] = (ma + cfg.BB_STD * std).shift(1)
    # BB_WIDTH_PCT must normalize by the SAME closed-bar close used for the BB
    # itself; using the current bar's close would re-introduce look-ahead.
    df["BB_WIDTH_PCT"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["close"].shift(1) * 100.0
    _n = max(1, int(getattr(cfg, "BB_WIDTH_AVG_LOOKBACK_DAYS", 10))
                * int(getattr(cfg, "BB_WIDTH_BARS_PER_DAY", 13)))
    df["BB_WIDTH_PCT_AVG"] = df["BB_WIDTH_PCT"].rolling(_n, min_periods=1).mean()
    df["BB_SQUEEZE_RATIO"] = df["BB_WIDTH_PCT"] / df["BB_WIDTH_PCT_AVG"]
    return df

# ===================================================================
# HELPERS
# ===================================================================

def _swing_high(bars_30m: pd.DataFrame, end_idx: int, lookback: int) -> float:
    start = max(0, end_idx - lookback + 1)
    if start > end_idx or end_idx < 0:
        return float("nan")
    return float(bars_30m["high"].iloc[start:end_idx + 1].max())


def _swing_low(bars_30m: pd.DataFrame, end_idx: int, lookback: int) -> float:
    start = max(0, end_idx - lookback + 1)
    if start > end_idx or end_idx < 0:
        return float("nan")
    return float(bars_30m["low"].iloc[start:end_idx + 1].min())


def _get_last_closed_30m_idx(bar_time: pd.Timestamp, bars_30m: pd.DataFrame) -> int:
    """Index of the latest 30-min bar whose period ended on or before bar_time.

    Kept for backward compatibility (used by bt_report). The hot path in
    run_backtest() uses a vectorised np.searchsorted instead.
    """
    mask = bars_30m["date"] < bar_time
    if not mask.any():
        return -1
    return int(mask.values.nonzero()[0][-1])


def _position_size(equity: float, price: float, atr_risk: float = float("nan")) -> int:
    """Compute share count for a new entry.

    atr_risk_mode: shares = (equity * RISK_PCT) / (ATR_MULT * ATR_period).
    Returns 0 (skip) if the ATR is not yet warmed up or non-positive.
    """
    if price <= 0:
        return 0
    if cfg.SIZING_TYPE == "atr_risk":
        if atr_risk is None or math.isnan(atr_risk) or atr_risk <= 0:
            return 0
        risk_dollars = equity * (cfg.RISK_PCT_PER_TRADE / 100.0)
        shares = int(risk_dollars / (cfg.ATR_RISK_MULTIPLIER * atr_risk))
        return max(0, shares)
    if cfg.SIZING_TYPE == "fixed_dollar":
        amount = cfg.SIZING_AMOUNT
    else:
        amount = equity * (cfg.SIZING_AMOUNT / 100.0)
    return max(1, int(amount / price))


def _format_held(total_minutes: int) -> str:
    """Humanize a duration in minutes to "{d}d {h}h" or "{h}h {m}m"."""
    if total_minutes <= 0:
        return "0m"
    days, rem = divmod(total_minutes, 24 * 60)
    hours, mins = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _close_position(position: dict, exit_time, exit_price: float, reason: str,
                    trades: list[dict]) -> float:
    """Record a trade exit and return the net P&L."""
    pnl_gross = (exit_price - position["entry_price"]) * position["shares"]
    comm = cfg.COMMISSION_PER_SHARE * position["shares"] * 2
    pnl_net = pnl_gross - comm
    # Held duration for analysis / report. Stored in minutes for sorting and as
    # a humanized string for the chart table.
    try:
        held_td = pd.Timestamp(exit_time) - pd.Timestamp(position["entry_time"])
        held_minutes = max(0, int(held_td.total_seconds() // 60))
    except Exception:
        held_minutes = 0
    trades.append({**position, "exit_time": str(exit_time),
                   "exit_price": round(float(exit_price), 4),
                   "exit_reason": reason,
                   "gross_pnl": pnl_gross,
                   "commission": comm,
                   "net_pnl": pnl_net,
                   "period_held_minutes": held_minutes,
                   "period_held": _format_held(held_minutes)})
    return pnl_net

# ===================================================================
# MAIN BACKTEST
# ===================================================================

# ===================================================================
# CROSS-SYMBOL INPUT BUILDERS
# ===================================================================
# These need data from EVERY symbol, so they run once in the main
# process before workers are dispatched.  The resulting dicts are
# small (a few KB) and pickle cheaply into worker tasks.

def _build_roc_cutoffs(daily_only: dict[str, pd.DataFrame]) -> dict:
    """For each trading day, the (100 - ROC_TOP_PCT) percentile of ROC_N
    across all symbols with a valid (non-NaN) ROC on that day.  A symbol's
    entry is allowed only if its ROC >= cutoff.
    """
    if not cfg.ENABLE_ROC_RANK_FILTER:
        return {}
    roc_col = f"ROC_{cfg.ROC_PERIOD}"
    pieces = []
    for sym, ddf in daily_only.items():
        if roc_col not in ddf.columns:
            continue
        sub = ddf[["date", roc_col]].copy()
        sub["day"] = sub["date"].dt.date
        sub = sub.rename(columns={roc_col: "roc"}).dropna(subset=["roc"])
        sub["symbol"] = sym
        pieces.append(sub[["day", "symbol", "roc"]])
    if not pieces:
        return {}
    roc_panel = pd.concat(pieces, ignore_index=True)
    q = (100.0 - cfg.ROC_TOP_PCT) / 100.0
    cutoffs = roc_panel.groupby("day")["roc"].quantile(q)
    return cutoffs.to_dict()


def _build_complex_rank_cutoffs() -> dict[int, dict]:
    """Per-day percentile cutoffs across the ROC ranking universe for
    each period in cfg.ROC_COMPLEX_PERIODS.

    Returns {period: {date -> cutoff_value (float)}}.

    A trading symbol's ROC_p is "hot" on day d iff
        ROC_p[d] >= cutoffs[p][d].

    Returns {} when not in Complex mode (caller short-circuits) or
    when the universe file is empty / missing.
    """
    if not cfg.ENABLE_ROC_RANK_FILTER:
        return {}
    if getattr(cfg, "ROC_RANK_MODE", "Simple_Rank") != "TC2000_Complex_Rank":
        return {}

    from bt_download import load_universe
    universe = load_universe()
    if not universe:
        print("WARN: ROC_COMPLEX_UNIVERSE_FILE empty/missing; "
              "Complex_Rank gate will reject ALL entries.")
        return {p: {} for p in cfg.ROC_COMPLEX_PERIODS}

    out: dict[int, dict] = {}
    q = (100.0 - cfg.ROC_COMPLEX_TOP_PCT) / 100.0
    panels: dict[int, list[pd.DataFrame]] = {p: [] for p in cfg.ROC_COMPLEX_PERIODS}
    n_loaded = 0
    for sym in tqdm(universe, desc="Complex_Rank universe"):
        ddf = _compute_daily(_load(sym, "daily"))
        if ddf.empty:
            continue
        n_loaded += 1
        sub_day = ddf["date"].dt.date.values
        for p in cfg.ROC_COMPLEX_PERIODS:
            col = f"ROC_{p}"
            if col not in ddf.columns:
                continue
            piece = pd.DataFrame({
                "day": sub_day,
                "roc": ddf[col].values,
            })
            panels[p].append(piece.dropna(subset=["roc"]))
    print(f"Complex_Rank universe: loaded {n_loaded}/{len(universe)} "
          f"symbols' daily data")
    for p, pieces in panels.items():
        if not pieces:
            out[p] = {}
            continue
        panel = pd.concat(pieces, ignore_index=True)
        out[p] = panel.groupby("day")["roc"].quantile(q).to_dict()
    return out


def _build_index_macd() -> dict:
    """Per-day MACD histogram on the benchmark INDEX_SYMBOL (e.g. SPY).
    Entry is only allowed when histogram > 0 on the latest completed daily.
    """
    if not cfg.ENABLE_INDEX_MACD_FILTER:
        return {}
    idx_df = _compute_daily(_load(cfg.INDEX_SYMBOL, "daily"))
    if idx_df.empty or "MACD_HIST" not in idx_df.columns:
        print(f"WARN: no daily data for INDEX_SYMBOL={cfg.INDEX_SYMBOL}; "
              "index MACD filter will block ALL entries until data is downloaded.")
        return {}
    sub = idx_df[["date", "MACD_HIST"]].dropna(subset=["MACD_HIST"]).copy()
    sub["day"] = sub["date"].dt.date
    return dict(zip(sub["day"], sub["MACD_HIST"].astype(float)))


def _build_trading_days(symbols: list[str]) -> list:
    """Union of all unique 5-min bar dates across the watchlist, clipped
    to [START_DATE, END_DATE].  Each worker receives the full list and
    iterates only over days where its symbol has bars.
    """
    all_dates: set = set()
    for sym in symbols:
        df5 = _load(sym, "5min")
        if df5.empty:
            continue
        all_dates.update(df5["date"].dt.date.unique())
    return sorted(d for d in all_dates
                  if cfg.START_DATE <= d <= cfg.END_DATE)


# ===================================================================
# PER-SYMBOL WORKER
# ===================================================================
# Module-level so it pickles for ProcessPoolExecutor on Windows spawn.
# Each worker reloads its own parquet (cheaper than pickling DataFrames
# across process boundaries) and simulates one symbol end-to-end.
#
# The `equity` column on each trade is the per-symbol running balance
# starting at INITIAL_CAPITAL; the orchestrator overwrites it after
# merging across all symbols (sorted by exit_time).  This is fine
# because USE_COMPOUNDING=False -> sizing uses INITIAL_CAPITAL, not
# the running equity, so the per-symbol simulation is fully independent.

def _simulate_one_symbol(args):
    """Simulate one symbol across all trading days.  Returns dict with
    trades / signals / shortlist_rows lists and a `skipped` flag."""
    (sym, trading_days, roc_cutoff_by_day, index_macd_by_day,
     complex_cutoffs_by_period,
     entry_after_time, entry_before_time, cooldown_td) = args

    daily = _compute_daily(_load(sym, "daily"))
    m30 = _compute_30min(_load(sym, "30min"))
    m5 = _load(sym, "5min")
    if daily.empty or m30.empty or m5.empty:
        return {"sym": sym, "trades": [], "signals": [],
                "shortlist_rows": [], "skipped": True}

    m30_time_ns = m30["date"].values.astype("datetime64[ns]")
    daily_dates = daily["date"].dt.date.values
    m5_dates = m5["date"].dt.date.values

    roc_col = f"ROC_{cfg.ROC_PERIOD}"
    atr_risk_col = f"ATR_{cfg.ATR_RISK_PERIOD}"

    # ---- Precompute Complex_Rank screen pass per day --------------------
    # screen_pass_by_day[date] = True iff the symbol is allowed to
    # trade on that day under TC2000_Complex_Rank rules:
    #   hot_within (top X% on ROC{5,10,30} during last N days)
    #   AND gates_pass (AVGV10 / price / ATR/C / MACD>0).
    # Computed once per symbol; the per-bar entry gate is a dict lookup.
    # Empty dict in Simple_Rank mode -- the entry gate short-circuits to
    # day_screen_pass=True so behaviour is identical.
    screen_pass_by_day: dict = {}
    if (cfg.ENABLE_ROC_RANK_FILTER
            and getattr(cfg, "ROC_RANK_MODE", "Simple_Rank")
                == "TC2000_Complex_Rank"
            and complex_cutoffs_by_period):
        _N = max(1, int(cfg.ROC_COMPLEX_LOOKBACK_DAYS))
        _days = daily["date"].dt.date.values
        _close = daily["close"].values.astype(float, copy=False)
        _vol_avg = (daily["AVGV10"].values
                    if "AVGV10" in daily.columns
                    else np.full(len(_days), np.nan))
        _macd = (daily["MACD_HIST"].values
                 if "MACD_HIST" in daily.columns
                 else np.full(len(_days), np.nan))
        _atr10 = (daily["ATR_10"].values
                  if "ATR_10" in daily.columns
                  else np.full(len(_days), np.nan))
        _atr25 = (daily["ATR_25"].values
                  if "ATR_25" in daily.columns
                  else np.full(len(_days), np.nan))

        # hot_any[i] -- top X% on ANY of ROC{5,10,30} on day i
        hot_any = np.zeros(len(_days), dtype=bool)
        for p in cfg.ROC_COMPLEX_PERIODS:
            cuts = complex_cutoffs_by_period.get(p, {})
            if not cuts:
                continue
            roc_vals = (daily[f"ROC_{p}"].values
                        if f"ROC_{p}" in daily.columns
                        else np.full(len(_days), np.nan))
            cut_arr = np.array([cuts.get(d, np.nan) for d in _days],
                               dtype=float)
            valid = ~(np.isnan(roc_vals) | np.isnan(cut_arr))
            hot_p = np.zeros_like(hot_any)
            hot_p[valid] = roc_vals[valid] >= cut_arr[valid]
            hot_any |= hot_p

        # hot_within[i] -- OR of hot_any over [i-N+1, i]
        # cumulative-sum trick (same pattern as had_cross_window below).
        if _N > 0 and len(hot_any) > 0:
            csum = np.concatenate(([0], np.cumsum(hot_any.astype(np.int32))))
            idx = np.arange(1, len(csum))
            lo = np.maximum(0, idx - _N)
            hot_within = (csum[idx] - csum[lo]) > 0
        else:
            hot_within = hot_any.copy()

        # gates_pass[i] -- per-symbol AVGV/price/ATR/MACD checks
        gates_pass = np.ones(len(_days), dtype=bool)
        if cfg.ROC_COMPLEX_REQUIRE_AVGV10:
            with np.errstate(invalid="ignore"):
                gates_pass &= (_vol_avg >= cfg.ROC_COMPLEX_REQUIRE_AVGV10)
        if cfg.ROC_COMPLEX_REQUIRE_MIN_PRICE:
            with np.errstate(invalid="ignore"):
                gates_pass &= (_close >= cfg.ROC_COMPLEX_REQUIRE_MIN_PRICE)
        if cfg.ROC_COMPLEX_REQUIRE_ATR10_PCT:
            with np.errstate(divide="ignore", invalid="ignore"):
                gates_pass &= ((_atr10 / _close * 100.0)
                               >= cfg.ROC_COMPLEX_REQUIRE_ATR10_PCT)
        if cfg.ROC_COMPLEX_REQUIRE_ATR25_PCT:
            with np.errstate(divide="ignore", invalid="ignore"):
                gates_pass &= ((_atr25 / _close * 100.0)
                               >= cfg.ROC_COMPLEX_REQUIRE_ATR25_PCT)
        if cfg.ROC_COMPLEX_REQUIRE_MACD_POS:
            with np.errstate(invalid="ignore"):
                gates_pass &= (_macd > 0)
        gates_pass &= ~np.isnan(_close)

        screen_pass_arr = hot_within & gates_pass
        screen_pass_by_day = {d: bool(p) for d, p in
                              zip(_days, screen_pass_arr)}

    # ---- Precompute "had a bullish 30m EMA cross in the last N bars" ----
    # Bullish cross on bar k = (fast[k] > slow[k]) AND (fast[k-1] <= slow[k-1]).
    # had_cross_window[k] = True iff at least one bullish cross occurred in
    # the half-open window (k - N, k] (i.e. the most recent N bars,
    # *including* bar k itself). Built via cumulative-sum trick so the
    # whole symbol is O(len(m30)).
    _fast_30 = m30[f"EMA_{cfg.TREND_FAST_EMA}"].values.astype(float, copy=False)
    _slow_30 = m30[f"EMA_{cfg.TREND_SLOW_EMA}"].values.astype(float, copy=False)
    cross_up = np.zeros(len(_fast_30), dtype=bool)
    if len(_fast_30) > 1:
        valid_pair = ~(np.isnan(_fast_30[1:]) | np.isnan(_slow_30[1:])
                       | np.isnan(_fast_30[:-1]) | np.isnan(_slow_30[:-1]))
        cross_up[1:] = valid_pair & (_fast_30[1:] > _slow_30[1:]) \
                                   & (_fast_30[:-1] <= _slow_30[:-1])
    _N = int(getattr(cfg, "HAD_30M_BULLISH_CROSS_LOOKBACK_BARS", 130))
    if len(cross_up) > 0 and _N > 0:
        _csum = np.concatenate(([0], np.cumsum(cross_up.astype(np.int32))))
        # had_cross_window[k] = (_csum[k+1] - _csum[max(0, k+1-N)]) > 0
        _idx = np.arange(1, len(_csum))
        _lo  = np.maximum(0, _idx - _N)
        had_cross_window = (_csum[_idx] - _csum[_lo]) > 0
    else:
        had_cross_window = np.zeros(len(_fast_30), dtype=bool)

    # ---- Per-symbol persistent state (was cross-symbol dicts in main) ----
    open_positions: list[dict] = []
    last_entry_time = None
    prev_close = float("nan")
    prev_lower_bb = float("nan")

    trades: list[dict] = []
    signals: list[dict] = []
    shortlist_rows: list[dict] = []
    equity = cfg.INITIAL_CAPITAL  # local; orchestrator rebuilds the cross-symbol curve

    for day in trading_days:
        day_net = 0.0

        # --- daily uptrend filter (gates NEW entries only) ---
        d_idx = int(np.searchsorted(daily_dates, day, side="right")) - 1
        if d_idx < 0:
            drow = None
            uptrend_ok = False
            ema_fast = ema_slow = float("nan")
            atr_risk_val = float("nan")
            macd_hist = float("nan")
            roc_val = float("nan")
            roc_cut = float("nan")
        else:
            drow = daily.iloc[d_idx]
            ema_fast = float(drow.get(f"EMA_{cfg.UPTREND_FAST_EMA}", float("nan")))
            ema_slow = float(drow.get(f"EMA_{cfg.UPTREND_SLOW_EMA}", float("nan")))
            uptrend_ok = (not math.isnan(ema_fast) and not math.isnan(ema_slow)
                          and ema_fast > ema_slow)
            atr_risk_val = float(drow.get(atr_risk_col, float("nan")))
            macd_hist = float(drow.get("MACD_HIST", float("nan")))
            roc_val = float(drow.get(roc_col, float("nan")))
            roc_cut = float(roc_cutoff_by_day.get(daily_dates[d_idx], float("nan"))) \
                if cfg.ENABLE_ROC_RANK_FILTER else float("nan")

        index_macd_hist = float(index_macd_by_day.get(day, float("nan"))) \
            if cfg.ENABLE_INDEX_MACD_FILTER else float("nan")

        # Per-day Complex_Rank screen result (True everywhere else, so
        # Simple_Rank short-circuits past the new gate).
        day_screen_pass = (screen_pass_by_day.get(day, False)
                           if (cfg.ENABLE_ROC_RANK_FILTER
                               and getattr(cfg, "ROC_RANK_MODE",
                                           "Simple_Rank")
                                   == "TC2000_Complex_Rank")
                           else True)

        # --- Slice today's 5-min bars via searchsorted ---
        i0 = int(np.searchsorted(m5_dates, day, side="left"))
        i1 = int(np.searchsorted(m5_dates, day, side="right"))
        if i0 == i1:
            continue
        day_5m = m5.iloc[i0:i1]

        # --- Vectorised 30-min index for every 5-min bar of the day ---
        bar_times_ns = day_5m["date"].values.astype("datetime64[ns]")
        idx_30m_arr = np.searchsorted(m30_time_ns, bar_times_ns, side="left") - 1

        # ---- Pre-extract today's OHLC + timestamps as numpy arrays ----
        n_bars = len(day_5m)
        o_arr = day_5m["open"].values.astype(float, copy=False)
        h_arr = day_5m["high"].values.astype(float, copy=False)
        l_arr = day_5m["low"].values.astype(float, copy=False)
        c_arr = day_5m["close"].values.astype(float, copy=False)
        t_list = day_5m["date"].tolist()

        # ---- Pre-resolve 30-min indicator values for every 5-min bar ----
        m30_bb_lower_full = m30["BB_LOWER"].values
        m30_ema_fast_full = m30[f"EMA_{cfg.TREND_FAST_EMA}"].values
        m30_ema_slow_full = m30[f"EMA_{cfg.TREND_SLOW_EMA}"].values
        m30_bb_width_pct_full = m30["BB_WIDTH_PCT"].values
        m30_bb_width_avg_full = m30["BB_WIDTH_PCT_AVG"].values
        m30_bb_squeeze_full   = m30["BB_SQUEEZE_RATIO"].values
        valid30 = idx_30m_arr >= 0
        safe_idx30 = np.where(valid30, idx_30m_arr, 0)
        lower_bb_arr = np.where(valid30, m30_bb_lower_full[safe_idx30], np.nan)
        ema100_30_arr = np.where(valid30, m30_ema_fast_full[safe_idx30], np.nan)
        ema200_30_arr = np.where(valid30, m30_ema_slow_full[safe_idx30], np.nan)
        bb_width_pct_arr = np.where(valid30, m30_bb_width_pct_full[safe_idx30], np.nan)
        bb_width_avg_arr = np.where(valid30, m30_bb_width_avg_full[safe_idx30], np.nan)
        bb_squeeze_arr   = np.where(valid30, m30_bb_squeeze_full[safe_idx30], np.nan)
        # Per-day daily ATR % (broadcast scalar; bars share the same daily close).
        _daily_close_today = float(drow["close"]) if drow is not None else float("nan")
        if (not math.isnan(atr_risk_val) and not math.isnan(_daily_close_today)
                and _daily_close_today > 0):
            daily_atr_pct_today = atr_risk_val / _daily_close_today * 100.0
        else:
            daily_atr_pct_today = float("nan")
        # bb_vs_atr_ratio per bar = bb_width_pct_arr / daily_atr_pct_today
        with np.errstate(invalid="ignore", divide="ignore"):
            if math.isnan(daily_atr_pct_today) or daily_atr_pct_today <= 0:
                bb_vs_atr_arr = np.full_like(bb_width_pct_arr, np.nan)
            else:
                bb_vs_atr_arr = bb_width_pct_arr / daily_atr_pct_today
        # Boolean per 5-min bar: did a bullish 30m EMA cross occur within
        # the last cfg.HAD_30M_BULLISH_CROSS_LOOKBACK_BARS 30-min bars
        # (as observed at the 30-min bar most recently closed before this
        # 5-min bar)?  False until a 30m bar exists for the day.
        had_cross_30m_arr = np.where(
            valid30, had_cross_window[safe_idx30], False).astype(bool)

        # ---- Vectorised running session low for today ----
        running_low_today = np.minimum.accumulate(l_arr)

        # ---- Vectorised BB cross detection over the whole day ----
        _mode = getattr(cfg, "ENTRY_CROSS_MODE", "same_bar")
        valid_bb = ~np.isnan(lower_bb_arr)
        # Build prev-bar arrays in BOTH modes so signals.csv can record
        # prev_close / prev_lower_bb regardless of which rule fired the cross.
        prev_c_full = np.empty(n_bars, dtype=float)
        prev_bb_full = np.empty(n_bars, dtype=float)
        prev_c_full[0] = prev_close
        prev_bb_full[0] = prev_lower_bb
        if n_bars > 1:
            prev_c_full[1:] = c_arr[:-1]
            prev_bb_full[1:] = lower_bb_arr[:-1]
        if _mode == "prev_close":
            is_cross_arr = (valid_bb
                            & ~np.isnan(prev_bb_full)
                            & ~np.isnan(prev_c_full)
                            & (prev_c_full < prev_bb_full)
                            & (c_arr > lower_bb_arr))
        else:
            is_cross_arr = (valid_bb
                            & (o_arr < lower_bb_arr)
                            & (c_arr > lower_bb_arr))
        # Refresh prev cache once per day (was per-bar).
        prev_close = float(c_arr[-1])
        prev_lower_bb = float(lower_bb_arr[-1]) \
            if not np.isnan(lower_bb_arr[-1]) else float("nan")

        day_open = float(o_arr[0])
        day_low_so_far = float("inf")
        session_low = float("inf")
        entries_today = 0
        triggered_today = False

        for i in range(n_bars):
            bar_time = t_list[i]
            bar_open = float(o_arr[i])
            bar_high = float(h_arr[i])
            bar_low = float(l_arr[i])
            bar_close = float(c_arr[i])
            rl_i = float(running_low_today[i])
            if rl_i < session_low:
                session_low = rl_i
            day_low_so_far = rl_i

            idx_30m = int(idx_30m_arr[i])
            lower_bb = float(lower_bb_arr[i])
            ema100_30 = float(ema100_30_arr[i])
            ema200_30 = float(ema200_30_arr[i])
            had_30m_bullish_cross = bool(had_cross_30m_arr[i])

            # --- (A0) MAE / MFE on every still-open position ---
            for pos in open_positions:
                ep = pos["entry_price"]
                if ep > 0:
                    pos["mae_pct"] = min(pos.get("mae_pct", 0.0),
                                         (bar_low / ep - 1.0) * 100.0)
                    pos["mfe_pct"] = max(pos.get("mfe_pct", 0.0),
                                         (bar_high / ep - 1.0) * 100.0)

            # --- (A) Exit check for every open sub-position ---
            survivors: list[dict] = []
            for pos in open_positions:
                exited = False
                if pos["stop_price"] > 0 and bar_low <= pos["stop_price"]:
                    pnl_net = _close_position(pos, bar_time, pos["stop_price"],
                                              "stop", trades)
                    equity += pnl_net
                    day_net += pnl_net
                    trades[-1]["equity"] = equity
                    exited = True
                elif cfg.ENABLE_TARGET and pos["target_price"] > 0 and bar_high >= pos["target_price"]:
                    tk = pos.get("target_kind")
                    reason = ("target_swing_high" if tk == "swing_high"
                              else "target_fib_1272" if tk == "fib_1272"
                              else "target")
                    fill_price = (bar_open if bar_open > pos["target_price"]
                                  else pos["target_price"])
                    pnl_net = _close_position(pos, bar_time, fill_price,
                                              reason, trades)
                    equity += pnl_net
                    day_net += pnl_net
                    trades[-1]["equity"] = equity
                    exited = True

                    if (cfg.ENABLE_TRIM_AND_TRAIL
                            and pos.get("leg") == 1
                            and pos.get("parent_id")):
                        pid = pos["parent_id"]
                        for sib in open_positions:
                            if sib is pos or sib.get("parent_id") != pid:
                                continue
                            if cfg.MOVE_STOPS_TO_BE_AFTER_LEG1:
                                sib["stop_price"] = round(sib["entry_price"], 2)
                            sib["ema_exit_period"] = cfg.EMA_EXIT_PERIOD_POST_LEG1
                if not exited:
                    survivors.append(pos)
            open_positions = survivors

            # --- (B) Signal detection (precomputed) ---
            is_cross = bool(is_cross_arr[i])

            # --- (C) Shortlist pre-filter ---
            shortlist_skip = ""
            if cfg.ENABLE_SHORTLIST:
                if cfg.SHORTLIST_REQUIRE_RED and bar_close >= day_open:
                    shortlist_skip = "not_red"
                elif not math.isnan(lower_bb) and lower_bb > 0 and \
                        bar_low > lower_bb * (1 + cfg.SHORTLIST_MAX_DIST_PCT / 100):
                    shortlist_skip = "far_from_bb"

            # --- (D) Log every BB cross as a signal (with skip reason if any) ---
            if is_cross:
                skip = ""
                if bar_time.time() < entry_after_time:
                    skip = "before_entry_time"
                elif bar_time.time() > entry_before_time:
                    skip = "after_entry_time"
                elif shortlist_skip:
                    skip = f"shortlist_{shortlist_skip}"
                elif not uptrend_ok:
                    skip = "daily_ema_filter"
                elif cfg.ENABLE_DAILY_MACD_FILTER and (
                        math.isnan(macd_hist) or macd_hist <= 0):
                    skip = "daily_macd_filter"
                elif cfg.ENABLE_INDEX_MACD_FILTER and (
                        math.isnan(index_macd_hist) or index_macd_hist <= 0):
                    skip = "index_macd_filter"
                elif cfg.ENABLE_ROC_RANK_FILTER and (
                        getattr(cfg, "ROC_RANK_MODE", "Simple_Rank")
                            == "Simple_Rank") and (
                        math.isnan(roc_val) or math.isnan(roc_cut) or roc_val < roc_cut):
                    skip = "roc_rank_filter"
                elif (cfg.ENABLE_ROC_RANK_FILTER
                      and getattr(cfg, "ROC_RANK_MODE", "Simple_Rank")
                          == "TC2000_Complex_Rank"
                      and not day_screen_pass):
                    skip = "complex_rank_filter"
                elif math.isnan(ema100_30) or math.isnan(ema200_30) or ema100_30 <= ema200_30:
                    skip = "30m_ema_filter"
                elif cfg.ENABLE_BELOW_30M_EMA and (
                        math.isnan(ema100_30) or bar_close > ema100_30):
                    skip = "above_30m_ema"
                elif (getattr(cfg, "ENABLE_RECENT_30M_CROSS_FILTER", False)
                      and not had_30m_bullish_cross):
                    skip = "no_recent_30m_cross"
                elif cfg.SIZING_TYPE == "atr_risk" and (
                        math.isnan(atr_risk_val) or atr_risk_val <= 0):
                    skip = "atr_unavailable"
                elif entries_today >= cfg.MAX_ENTRIES_PER_SYMBOL_PER_DAY:
                    skip = "max_entries_per_day"
                # MAX_POSITIONS check intentionally skipped inside workers
                # (cross-symbol cap; user has it at 100k = effectively unlimited).
                elif (last_entry_time is not None
                      and (bar_time - last_entry_time) < cooldown_td):
                    skip = "cooldown"

                _pc = float(prev_c_full[i])
                _pbb = float(prev_bb_full[i])
                _bbw  = float(bb_width_pct_arr[i])
                _bbwa = float(bb_width_avg_arr[i])
                _sq   = float(bb_squeeze_arr[i])
                _atrp = float(daily_atr_pct_today)
                _bba  = float(bb_vs_atr_arr[i])
                signals.append({
                    "date": str(day), "symbol": sym,
                    "signal_time": str(bar_time),
                    "bar_open": round(bar_open, 2),
                    "bar_close": round(bar_close, 2),
                    "bar_high": round(bar_high, 2),
                    "bar_low": round(bar_low, 2),
                    "lower_bb": round(lower_bb, 2),
                    "ema100_30m": round(ema100_30, 2) if not math.isnan(ema100_30) else "",
                    "ema200_30m": round(ema200_30, 2) if not math.isnan(ema200_30) else "",
                    f"daily_atr_{cfg.ATR_RISK_PERIOD}": round(atr_risk_val, 4) if not math.isnan(atr_risk_val) else "",
                    "daily_macd_hist": round(macd_hist, 4) if not math.isnan(macd_hist) else "",
                    "index_macd_hist": round(index_macd_hist, 4) if not math.isnan(index_macd_hist) else "",
                    f"daily_roc_{cfg.ROC_PERIOD}": round(roc_val, 3) if not math.isnan(roc_val) else "",
                    "daily_roc_cutoff": round(roc_cut, 3) if not math.isnan(roc_cut) else "",
                    "prev_close": round(_pc, 2) if not math.isnan(_pc) else "",
                    "prev_lower_bb": round(_pbb, 2) if not math.isnan(_pbb) else "",
                    "entry_cross_mode": _mode,
                    "had_30m_bullish_cross": had_30m_bullish_cross,
                    "roc_rank_mode": getattr(cfg, "ROC_RANK_MODE",
                                             "Simple_Rank"),
                    "complex_rank_pass": (day_screen_pass
                                          if (cfg.ENABLE_ROC_RANK_FILTER
                                              and getattr(cfg, "ROC_RANK_MODE",
                                                          "Simple_Rank")
                                                  == "TC2000_Complex_Rank")
                                          else ""),
                    "bb_width_pct_30m":     round(_bbw, 3)  if not math.isnan(_bbw)  else "",
                    "bb_width_pct_30m_avg": round(_bbwa, 3) if not math.isnan(_bbwa) else "",
                    "bb_squeeze_ratio_30m": round(_sq, 3)   if not math.isnan(_sq)   else "",
                    "daily_atr_pct":        round(_atrp, 3) if not math.isnan(_atrp) else "",
                    "bb_vs_atr_ratio":      round(_bba, 3)  if not math.isnan(_bba)  else "",
                    "traded": skip == "",
                    "skip_reason": skip,
                })

            # --- (E) Entry gate ---
            if not is_cross:
                continue
            if bar_time.time() < entry_after_time:
                continue
            if bar_time.time() > entry_before_time:
                continue
            if shortlist_skip:
                continue
            if not uptrend_ok:
                continue
            if cfg.ENABLE_DAILY_MACD_FILTER and (
                    math.isnan(macd_hist) or macd_hist <= 0):
                continue
            if cfg.ENABLE_INDEX_MACD_FILTER and (
                    math.isnan(index_macd_hist) or index_macd_hist <= 0):
                continue
            if (cfg.ENABLE_ROC_RANK_FILTER
                    and getattr(cfg, "ROC_RANK_MODE", "Simple_Rank")
                        == "Simple_Rank"
                    and (math.isnan(roc_val) or math.isnan(roc_cut)
                         or roc_val < roc_cut)):
                continue
            if (cfg.ENABLE_ROC_RANK_FILTER
                    and getattr(cfg, "ROC_RANK_MODE", "Simple_Rank")
                        == "TC2000_Complex_Rank"
                    and not day_screen_pass):
                continue
            if math.isnan(ema100_30) or math.isnan(ema200_30) or ema100_30 <= ema200_30:
                continue
            if cfg.ENABLE_BELOW_30M_EMA and (math.isnan(ema100_30) or bar_close > ema100_30):
                continue
            if (getattr(cfg, "ENABLE_RECENT_30M_CROSS_FILTER", False)
                    and not had_30m_bullish_cross):
                continue
            if entries_today >= cfg.MAX_ENTRIES_PER_SYMBOL_PER_DAY:
                continue
            # MAX_POSITIONS check intentionally skipped inside workers.
            if (last_entry_time is not None
                    and (bar_time - last_entry_time) < cooldown_td):
                continue

            # --- (F) Open new sub-position(s) ---
            entry_price = bar_close
            sizing_equity = equity if cfg.USE_COMPOUNDING else cfg.INITIAL_CAPITAL
            total_shares = _position_size(sizing_equity, entry_price, atr_risk_val)
            if total_shares <= 0:
                continue
            stop_price = session_low - cfg.STOP_OFFSET if cfg.ENABLE_STOP else 0.0

            swing_high = _swing_high(m30, idx_30m, cfg.TARGET_LOOKBACK) \
                if cfg.ENABLE_TARGET else float("nan")
            swing_low = _swing_low(m30, idx_30m, cfg.TARGET_LOOKBACK) \
                if cfg.ENABLE_TARGET else float("nan")
            fib_target = float("nan")
            if (not math.isnan(swing_high) and not math.isnan(swing_low)
                    and swing_high > swing_low):
                fib_target = swing_low + cfg.LEG2_FIB_EXTENSION * (swing_high - swing_low)

            if (not math.isnan(swing_high) and not math.isnan(swing_low)
                    and swing_high > swing_low):
                entry_pct_in_range = (entry_price - swing_low) / (swing_high - swing_low) * 100.0
            else:
                entry_pct_in_range = float("nan")

            high_6mo = float("nan")
            pct_from_6mo_high = float("nan")
            prior_mask = daily["date"].dt.date < day
            if prior_mask.any():
                prior_highs = daily.loc[prior_mask, "high"].tail(126)
                if not prior_highs.empty:
                    h6 = float(prior_highs.max())
                    if h6 > 0:
                        high_6mo = h6
                        pct_from_6mo_high = (entry_price - h6) / h6 * 100.0

            risk_dollars = sizing_equity * (cfg.RISK_PCT_PER_TRADE / 100.0) \
                if cfg.SIZING_TYPE == "atr_risk" else 0.0
            daily_close_for_atr = float(drow["close"]) if drow is not None else float("nan")
            daily_atr_pct = (atr_risk_val / daily_close_for_atr * 100.0) \
                if (not math.isnan(atr_risk_val) and not math.isnan(daily_close_for_atr)
                    and daily_close_for_atr > 0) else float("nan")

            parent_id = f"{sym}_{bar_time.isoformat()}"
            base_pos = {
                "date": str(day), "symbol": sym,
                "entry_time": str(bar_time), "entry_price": round(entry_price, 4),
                "original_entry_price": round(entry_price, 4),
                "parent_id": parent_id,
                "stop_price": round(stop_price, 2),
                "session_low_at_entry": round(session_low, 2),
                "lower_bb": round(lower_bb, 2),
                "ema100_30m": round(ema100_30, 2),
                "ema200_30m": round(ema200_30, 2),
                "daily_ema_fast": round(ema_fast, 2) if not math.isnan(ema_fast) else float("nan"),
                "daily_ema_slow": round(ema_slow, 2) if not math.isnan(ema_slow) else float("nan"),
                "atr7": round(atr_risk_val, 4) if not math.isnan(atr_risk_val) else float("nan"),
                # parent_risk_dollars = total $ risk budget for the WHOLE parent
                # trade (= equity * RISK_PCT). Same value on every leg row by
                # design. Per-leg risk_dollars (proportional share) is added
                # below in the position_value loop, after leg sizes are known.
                "parent_risk_dollars": round(risk_dollars, 2),
                "mae_pct": 0.0,
                "mfe_pct": 0.0,
                "daily_roc": round(roc_val, 3) if not math.isnan(roc_val) else float("nan"),
                "daily_macd_hist": round(macd_hist, 4) if not math.isnan(macd_hist) else float("nan"),
                "daily_atr_pct": round(daily_atr_pct, 3) if not math.isnan(daily_atr_pct) else float("nan"),
                # ---- Volatility regime at entry (audit-only; bracketed in analysis.html) ----
                "bb_width_pct_30m":     round(float(bb_width_pct_arr[i]), 3) if not math.isnan(float(bb_width_pct_arr[i])) else float("nan"),
                "bb_width_pct_30m_avg": round(float(bb_width_avg_arr[i]), 3) if not math.isnan(float(bb_width_avg_arr[i])) else float("nan"),
                "bb_squeeze_ratio_30m": round(float(bb_squeeze_arr[i]), 3)   if not math.isnan(float(bb_squeeze_arr[i]))   else float("nan"),
                "bb_vs_atr_ratio":      round(float(bb_vs_atr_arr[i]), 3)    if not math.isnan(float(bb_vs_atr_arr[i]))    else float("nan"),
                "day_of_week": bar_time.day_name() if hasattr(bar_time, "day_name") else "",
                "entry_hhmm": bar_time.strftime("%H:%M") if hasattr(bar_time, "strftime") else "",
                "swing_low_at_entry": round(swing_low, 2) if not math.isnan(swing_low) else 0.0,
                "swing_high_at_entry": round(swing_high, 2) if not math.isnan(swing_high) else 0.0,
                "fib_target_at_entry": round(fib_target, 2) if not math.isnan(fib_target) else 0.0,
                "entry_pct_in_range": round(entry_pct_in_range, 2) if not math.isnan(entry_pct_in_range) else float("nan"),
                "high_6mo": round(high_6mo, 4) if not math.isnan(high_6mo) else float("nan"),
                "pct_from_6mo_high": round(pct_from_6mo_high, 2) if not math.isnan(pct_from_6mo_high) else float("nan"),
                # True if a bullish 30m EMA cross (TREND_FAST over TREND_SLOW)
                # occurred within the last cfg.HAD_30M_BULLISH_CROSS_LOOKBACK_BARS
                # 30-min bars at the moment of entry. Tag-only by default;
                # also gates entries when ENABLE_RECENT_30M_CROSS_FILTER=True.
                "had_30m_bullish_cross": bool(had_30m_bullish_cross),
            }

            legs: list[dict] = []
            swing_high_target = (round(swing_high, 2)
                                 if (not math.isnan(swing_high) and swing_high > entry_price)
                                 else 0.0)
            fib_ok = (not math.isnan(fib_target)
                      and fib_target > max(entry_price, swing_high if not math.isnan(swing_high) else entry_price))
            fib_target_rnd = round(fib_target, 2) if fib_ok else 0.0

            if cfg.ENABLE_TRIM_AND_TRAIL:
                n1 = int(total_shares * cfg.LEG1_PCT)
                n2 = int(total_shares * cfg.LEG2_PCT)
                n3 = total_shares - n1 - n2
                if not fib_ok:
                    n1 += n2
                    n2 = 0
                if swing_high_target == 0.0:
                    n3 += n1 + n2
                    n1 = n2 = 0
                if n1 > 0:
                    legs.append({**base_pos, "shares": n1, "leg": 1,
                                 "target_kind": "swing_high",
                                 "target_price": swing_high_target,
                                 "ema_exit_period": cfg.EMA_EXIT_PERIOD_PRE_LEG1})
                if n2 > 0:
                    legs.append({**base_pos, "shares": n2, "leg": 2,
                                 "target_kind": "fib_1272",
                                 "target_price": fib_target_rnd,
                                 "ema_exit_period": cfg.EMA_EXIT_PERIOD_PRE_LEG1})
                if n3 > 0:
                    legs.append({**base_pos, "shares": n3, "leg": 3,
                                 "target_kind": "ema_trail",
                                 "target_price": 0.0,
                                 "ema_exit_period": cfg.EMA_EXIT_PERIOD_PRE_LEG1})
            else:
                legs.append({**base_pos, "shares": total_shares, "leg": 1,
                             "target_kind": "swing_high",
                             "target_price": swing_high_target,
                             "ema_exit_period": cfg.EMA_EXIT_PERIOD})

            if not legs:
                continue

            for L in legs:
                L["position_value"] = round(L["shares"] * entry_price, 2)
                # Per-leg share of the parent risk budget. Uses actual leg
                # shares (not LEG_PCT) so it survives integer truncation, the
                # fib-fold-up (n1 += n2) and the swing-fold-up (n3 += n1+n2).
                # Sums across legs back to parent_risk_dollars.
                L["risk_dollars"] = (round(risk_dollars * L["shares"] / total_shares, 2)
                                     if total_shares > 0 else 0.0)
                open_positions.append(L)

            last_entry_time = bar_time
            entries_today += 1
            triggered_today = True

        # --- end-of-day diagnostics (shortlist.csv) ---
        day_close = float(c_arr[-1])
        last_lower_bb = float(lower_bb_arr[-1])
        dist_pct = float("nan")
        if not math.isnan(last_lower_bb) and last_lower_bb > 0:
            dist_pct = (day_low_so_far - last_lower_bb) / last_lower_bb * 100
        was_red = day_close < day_open
        shortlist_rows.append({
            "date": str(day), "symbol": sym,
            "day_open": round(day_open, 2),
            "day_low": round(day_low_so_far, 2),
            "day_close": round(day_close, 2),
            "last_lower_bb": round(last_lower_bb, 2) if not math.isnan(last_lower_bb) else "",
            "distance_to_lower_bb_pct": round(dist_pct, 3) if not math.isnan(dist_pct) else "",
            "was_red": was_red,
            "uptrend_ok": uptrend_ok,
            "triggered": triggered_today,
        })

        # --- (G) End-of-day: EOD flatten (only if explicitly enabled) ---
        if cfg.ENABLE_EOD_FLATTEN and open_positions:
            exit_price = float(c_arr[-1])
            exit_time = t_list[-1]
            for pos in open_positions:
                pnl_net = _close_position(pos, exit_time, exit_price,
                                          "eod", trades)
                equity += pnl_net
                day_net += pnl_net
                trades[-1]["equity"] = equity
            open_positions = []

        # --- (H) End-of-day: EMA-close exit (per-symbol) ---
        # Original code ran this in a separate after-all-syms loop, but it
        # only ever touches THIS symbol's positions, so per-symbol per-day
        # is functionally identical.
        if cfg.ENABLE_EMA_EXIT and open_positions:
            d_idx2 = int(np.searchsorted(daily_dates, day, side="right")) - 1
            if d_idx2 >= 0 and daily_dates[d_idx2] == day:
                drow2 = daily.iloc[d_idx2]
                daily_close_eod = float(drow2["close"])
                e0 = int(np.searchsorted(m5_dates, day, side="left"))
                e1 = int(np.searchsorted(m5_dates, day, side="right"))
                if e0 == e1:
                    exit_time = pd.Timestamp(day, tz="US/Eastern")
                else:
                    exit_time = m5["date"].iloc[e1 - 1]
                survivors2: list[dict] = []
                for pos in open_positions:
                    ema_p = pos.get("ema_exit_period", cfg.EMA_EXIT_PERIOD)
                    ema_val = float(drow2.get(f"EMA_{ema_p}", float("nan")))
                    if not math.isnan(ema_val) and daily_close_eod < ema_val:
                        pnl_net = _close_position(pos, exit_time, daily_close_eod,
                                                  "ema_exit", trades)
                        equity += pnl_net
                        day_net += pnl_net
                        trades[-1]["equity"] = equity
                    else:
                        survivors2.append(pos)
                open_positions = survivors2

    # --- Mark-to-market flush at backtest end ---
    if open_positions:
        last_row = m5.iloc[-1]
        exit_price = float(last_row["close"])
        for pos in open_positions:
            pnl_net = _close_position(pos, last_row["date"], exit_price,
                                      "open_at_end", trades)
            equity += pnl_net
            trades[-1]["equity"] = equity
        open_positions = []

    return {"sym": sym, "trades": trades, "signals": signals,
            "shortlist_rows": shortlist_rows, "skipped": False}


# ===================================================================
# OUTPUT WRITER
# ===================================================================

def _safe_csv(df: pd.DataFrame, name: str):
    path = os.path.join(cfg.OUTPUT_DIR, name)
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        alt = path.replace(".csv", "_new.csv")
        df.to_csv(alt, index=False)
        print(f"  WARNING: {name} locked, saved as {os.path.basename(alt)}")


# ===================================================================
# MAIN ORCHESTRATOR
# ===================================================================

def run_backtest() -> tuple[list[dict], pd.DataFrame]:
    symbols = load_watchlist()
    print(f"Backtesting {len(symbols)} symbols from {cfg.START_DATE} to {cfg.END_DATE}")

    # --- Prefetch daily data (needed for cross-symbol ROC cutoffs) ---
    daily_only: dict[str, pd.DataFrame] = {}
    for sym in tqdm(symbols, desc="Daily prefetch"):
        d = _compute_daily(_load(sym, "daily"))
        if d.empty:
            continue
        daily_only[sym] = d

    if not daily_only:
        print("No data loaded. Run bt_download.py first.")
        return [], pd.DataFrame()

    roc_cutoff_by_day = _build_roc_cutoffs(daily_only)
    complex_cutoffs_by_period = _build_complex_rank_cutoffs()
    index_macd_by_day = _build_index_macd()
    trading_days = _build_trading_days(list(daily_only.keys()))

    print(f"Simulating {len(trading_days)} trading days across {len(daily_only)} symbols...")

    # --- Decide on parallelism ---
    n_workers = getattr(cfg, "NUM_WORKERS", 1)
    if cfg.USE_COMPOUNDING and n_workers > 1:
        print("WARNING: USE_COMPOUNDING=True is incompatible with parallel "
              "workers (cross-symbol equity coupling). Falling back to NUM_WORKERS=1.")
        n_workers = 1

    entry_after_time = dtime.fromisoformat(cfg.ENTRY_TIME_AFTER)
    entry_before_time = dtime.fromisoformat(cfg.ENTRY_TIME_BEFORE)
    cooldown_td = pd.Timedelta(minutes=cfg.ENTRY_COOLDOWN_MINUTES)

    task_args = [(sym, trading_days, roc_cutoff_by_day, index_macd_by_day,
                  complex_cutoffs_by_period,
                  entry_after_time, entry_before_time, cooldown_td)
                 for sym in daily_only]

    all_trades: list[dict] = []
    all_signals: list[dict] = []
    all_shortlist: list[dict] = []

    if n_workers == 1:
        print("Running sequentially (NUM_WORKERS=1)...")
        for a in tqdm(task_args, desc="Backtesting"):
            r = _simulate_one_symbol(a)
            all_trades.extend(r["trades"])
            all_signals.extend(r["signals"])
            all_shortlist.extend(r["shortlist_rows"])
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Running with {n_workers} worker processes...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_simulate_one_symbol, a) for a in task_args]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Backtesting"):
                r = f.result()
                all_trades.extend(r["trades"])
                all_signals.extend(r["signals"])
                all_shortlist.extend(r["shortlist_rows"])

    # --- Rebuild equity curve sequentially from merged trades ---
    # Sort by exit_time so the curve is deterministic regardless of
    # worker completion order; tie-break by symbol/parent_id/leg.
    all_trades.sort(key=lambda t: (str(t.get("exit_time", "")),
                                   t.get("symbol", ""),
                                   t.get("parent_id", ""),
                                   t.get("leg", 0)))

    equity = cfg.INITIAL_CAPITAL
    daily_net: dict[str, float] = {}
    eod_equity: dict[str, float] = {}
    for t in all_trades:
        equity += t["net_pnl"]
        t["equity"] = round(equity, 2)
        d = str(t.get("exit_time", ""))[:10]
        if d:
            daily_net[d] = daily_net.get(d, 0.0) + t["net_pnl"]
            eod_equity[d] = equity

    daily_pnl_rows = [{"date": d,
                       "daily_pnl": round(daily_net[d], 2),
                       "equity": round(eod_equity[d], 2),
                       # open_positions snapshot is not tracked in parallel
                       # mode (would require a global replay).  Set to 0.
                       "open_positions": 0}
                      for d in sorted(daily_net.keys())]

    # Sort signals/shortlist for deterministic CSV output.
    all_signals.sort(key=lambda s: (s.get("date", ""),
                                    s.get("signal_time", ""),
                                    s.get("symbol", "")))
    all_shortlist.sort(key=lambda s: (s.get("date", ""),
                                      s.get("symbol", "")))

    # --- Build output ---
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    daily_df = pd.DataFrame(daily_pnl_rows) if daily_pnl_rows else pd.DataFrame()
    signals_df = pd.DataFrame(all_signals) if all_signals else pd.DataFrame()
    shortlist_df = pd.DataFrame(all_shortlist) if all_shortlist else pd.DataFrame()

    if not daily_df.empty:
        daily_df["cumulative_pnl"] = daily_df["daily_pnl"].cumsum()
        peak = daily_df["equity"].cummax()
        daily_df["drawdown_pct"] = ((daily_df["equity"] - peak) / peak * 100).round(2)

    # --- Save ---
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not trades_df.empty:
        _safe_csv(trades_df, "trades.csv")
    if not signals_df.empty:
        _safe_csv(signals_df, "signals.csv")
        print(f"  Signals logged: {len(signals_df)}")
    if not shortlist_df.empty:
        _safe_csv(shortlist_df, "shortlist.csv")
        print(f"  Shortlist rows: {len(shortlist_df)}")
    if not daily_df.empty:
        _safe_csv(daily_df, "strategy_daily_pnl.csv")

    _print_summary(trades_df, daily_df)
    return all_trades, daily_df


def _print_summary(trades_df: pd.DataFrame, daily_df: pd.DataFrame):
    if trades_df.empty:
        print("\nNo trades generated.")
        return
    n = len(trades_df)
    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    total_pnl = trades_df["net_pnl"].sum()
    win_rate = len(wins) / n * 100 if n else 0
    avg_win = wins["net_pnl"].mean() if len(wins) else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0
    pf = abs(wins["net_pnl"].sum() / losses["net_pnl"].sum()) if losses["net_pnl"].sum() != 0 else float("inf")

    trades_df["pct_change"] = (trades_df["net_pnl"] / trades_df["position_value"] * 100)
    avg_pct = trades_df["pct_change"].mean()

    max_dd = daily_df["drawdown_pct"].min() if not daily_df.empty else 0

    exit_counts = trades_df["exit_reason"].value_counts().to_dict() if "exit_reason" in trades_df.columns else {}

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades:   {n}")
    print(f"Win Rate:       {win_rate:.1f}%")
    print(f"Net P&L:        ${total_pnl:,.2f}")
    print(f"Avg %/Trade:    {avg_pct:.2f}%")
    print(f"Avg Win:        ${avg_win:,.2f}")
    print(f"Avg Loss:       ${avg_loss:,.2f}")
    print(f"Profit Factor:  {pf:.2f}")
    print(f"Max Drawdown:   {max_dd:.2f}%")
    if not daily_df.empty:
        print(f"Final Equity:   ${daily_df['equity'].iloc[-1]:,.2f}")
    for reason, count in exit_counts.items():
        print(f"  Exit {reason}: {count}")
    print(f"{'='*60}")


def main():
    run_backtest()


if __name__ == "__main__":
    main()
