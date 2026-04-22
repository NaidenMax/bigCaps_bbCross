"""One-off audit script for AAPL on 2026-04-21.

Verifies the look-ahead fix in `_compute_30min`:
  - dumps every 5-min bar with the 30-min bar it maps to,
  - shows which 30-min bar's CLOSE the indicator value was sealed at
    (post-shift this is the bar at idx-1, NOT the in-progress bar),
  - reproduces the prev_close / prev_lower_bb cross check,
  - then runs `_simulate_one_symbol` for AAPL only and prints the
    resulting trade(s) on that day.

Delete after review.
"""
from __future__ import annotations

import os
import sys
import math
import datetime as dt
import numpy as np
import pandas as pd

# Make `backtest/` importable when invoked from project root.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bt_config as cfg
from bt_backtest import (
    _load,
    _compute_daily,
    _compute_30min,
    _simulate_one_symbol,
    _build_roc_cutoffs,
    _build_complex_rank_cutoffs,
    _build_index_macd,
)

SYMBOL = "AAPL"
TARGET_DAY = dt.date(2026, 4, 21)
OUT_DIR = os.path.join(cfg.OUTPUT_DIR, "debug")
OUT_CSV = os.path.join(OUT_DIR, f"{SYMBOL}_{TARGET_DAY.isoformat()}_bars.csv")


def _fmt_ts(ts) -> str:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return ""
    try:
        return pd.Timestamp(ts).tz_convert("US/Eastern").strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Load + compute (USES PATCHED _compute_30min) ----
    daily = _compute_daily(_load(SYMBOL, "daily"))
    m30 = _compute_30min(_load(SYMBOL, "30min"))
    m5 = _load(SYMBOL, "5min")
    if daily.empty or m30.empty or m5.empty:
        print(f"ERROR: missing parquet data for {SYMBOL}")
        return

    # ---- 2. Slice to TARGET_DAY (ET) ----
    m5_day = m5[m5["date"].dt.date == TARGET_DAY].reset_index(drop=True)
    if m5_day.empty:
        print(f"ERROR: no 5-min bars for {SYMBOL} on {TARGET_DAY}")
        return

    # ---- 2b. Daily look-ahead audit (post-fix). Show both yesterday's
    #          and today's daily rows for the columns that actually feed
    #          the entry filter; post-fix the engine should consume the
    #          YESTERDAY row at intraday signal time on TARGET_DAY.
    daily_dates = daily["date"].dt.date.values
    d_idx_postfix = int(np.searchsorted(daily_dates, TARGET_DAY,
                                        side="left")) - 1
    d_idx_today = int(np.searchsorted(daily_dates, TARGET_DAY,
                                      side="right")) - 1
    fast_d = f"EMA_{cfg.UPTREND_FAST_EMA}"
    slow_d = f"EMA_{cfg.UPTREND_SLOW_EMA}"
    cols = ["date", "close", fast_d, slow_d, "MACD_HIST"]
    cols += [c_ for c_ in daily.columns
             if (c_.startswith("ATR_") or c_.startswith("ROC_"))
             and c_ in daily.columns and c_ not in cols]
    print(f"\n--- daily-row audit for {SYMBOL} on {TARGET_DAY} ---")
    print(f"  d_idx_postfix (side='left'-1)  = {d_idx_postfix}  "
          f"-> date={daily_dates[d_idx_postfix] if d_idx_postfix >= 0 else None}")
    print(f"  d_idx_buggy   (side='right'-1) = {d_idx_today}  "
          f"-> date={daily_dates[d_idx_today]   if d_idx_today   >= 0 else None}")
    avail = [c_ for c_ in cols if c_ in daily.columns]
    if d_idx_postfix >= 0:
        print(f"\n  YESTERDAY row used post-fix (intraday signals on {TARGET_DAY}):")
        print(daily.iloc[[d_idx_postfix]][avail].to_string(index=False))
    if d_idx_today >= 0 and d_idx_today != d_idx_postfix:
        print(f"\n  TODAY row that the BUGGY engine would have used:")
        print(daily.iloc[[d_idx_today]][avail].to_string(index=False))
        if d_idx_postfix >= 0:
            yrow = daily.iloc[d_idx_postfix]
            trow = daily.iloc[d_idx_today]
            print("\n  delta(today - yesterday) for filter inputs:")
            for cc in avail:
                if cc == "date":
                    continue
                try:
                    yv = float(yrow[cc]); tv = float(trow[cc])
                    print(f"    {cc:>20s}: yesterday={yv:.4f}  today={tv:.4f}  "
                          f"delta={tv - yv:+.4f}")
                except Exception:
                    pass

    m30_time_ns = m30["date"].values.astype("datetime64[ns]")
    bar_times_ns = m5_day["date"].values.astype("datetime64[ns]")
    idx_30m_arr = np.searchsorted(m30_time_ns, bar_times_ns, side="left") - 1

    # ---- 3. For each 5m bar, look up the 30m bar it maps to + the
    #         indicator values it actually reads (post-shift). Also
    #         compute prev_close / prev_lower_bb / cross flag exactly
    #         as the worker does (mode = ENTRY_CROSS_MODE).
    fast = f"EMA_{cfg.TREND_FAST_EMA}"
    slow = f"EMA_{cfg.TREND_SLOW_EMA}"

    bb_lower = m30["BB_LOWER"].values
    bb_upper = m30["BB_UPPER"].values
    bb_w_pct = m30["BB_WIDTH_PCT"].values
    bb_w_avg = m30["BB_WIDTH_PCT_AVG"].values
    bb_sq    = m30["BB_SQUEEZE_RATIO"].values
    ema_fast = m30[fast].values
    ema_slow = m30[slow].values
    m30_open = m30["open"].values
    m30_close = m30["close"].values

    valid30 = idx_30m_arr >= 0
    safe = np.where(valid30, idx_30m_arr, 0)

    rows = []
    n = len(m5_day)
    o = m5_day["open"].values
    h = m5_day["high"].values
    l = m5_day["low"].values
    c = m5_day["close"].values
    t = m5_day["date"].tolist()

    # prev_close/prev_lower_bb seed -- worker carries from prev day; for
    # the audit we leave the first bar as NaN. Subsequent bars mirror
    # the worker's vectorised prev_c_full / prev_bb_full arrays.
    prev_c_full = np.full(n, np.nan)
    prev_bb_full = np.full(n, np.nan)
    if n > 1:
        prev_c_full[1:] = c[:-1]
        prev_bb_full[1:] = np.where(valid30, bb_lower[safe], np.nan)[:-1]

    mode = getattr(cfg, "ENTRY_CROSS_MODE", "same_bar")
    cur_bb = np.where(valid30, bb_lower[safe], np.nan)
    if mode == "prev_close":
        cross = (~np.isnan(cur_bb) & ~np.isnan(prev_bb_full)
                 & ~np.isnan(prev_c_full)
                 & (prev_c_full < prev_bb_full)
                 & (c > cur_bb))
    else:
        cross = (~np.isnan(cur_bb)
                 & (o < cur_bb)
                 & (c > cur_bb))

    for i in range(n):
        idx = int(idx_30m_arr[i]) if valid30[i] else -1
        # The 30m bar whose START timestamp idx points to.
        t_30m_start = m30["date"].iloc[idx] if idx >= 0 else None
        # Post-shift the indicator at row idx was COMPUTED USING closes
        # through bar (idx-1), i.e. sealed at the END of bar (idx-1) =
        # the START of bar idx. Show that timestamp explicitly.
        t_30m_sealed = m30["date"].iloc[idx - 1] if idx >= 1 else None
        # Bar idx-1 itself, for context (this is the 30m bar that just
        # CLOSED at t_30m_start).
        prev_30m_open = float(m30_open[idx - 1]) if idx >= 1 else float("nan")
        prev_30m_close = float(m30_close[idx - 1]) if idx >= 1 else float("nan")

        rows.append({
            "i": i,
            "t_5m_ET":              _fmt_ts(t[i]),
            "5m_open":              round(float(o[i]), 4),
            "5m_high":              round(float(h[i]), 4),
            "5m_low":               round(float(l[i]), 4),
            "5m_close":             round(float(c[i]), 4),
            "idx_30m":              idx,
            "t_30m_start_ET":       _fmt_ts(t_30m_start),
            "t_30m_indicator_sealed_at_ET": _fmt_ts(t_30m_start),  # post-shift = start of mapped bar
            "t_30m_window_that_produced_indicator_close_ET":
                                    _fmt_ts(t_30m_sealed),  # window [sealed_at, start)
            "30m_prev_window_open":  round(prev_30m_open, 4) if not math.isnan(prev_30m_open) else "",
            "30m_prev_window_close": round(prev_30m_close, 4) if not math.isnan(prev_30m_close) else "",
            "BB_LOWER_30m":         round(float(bb_lower[idx]), 4) if idx >= 0 and not math.isnan(bb_lower[idx]) else "",
            "BB_UPPER_30m":         round(float(bb_upper[idx]), 4) if idx >= 0 and not math.isnan(bb_upper[idx]) else "",
            "BB_WIDTH_PCT_30m":     round(float(bb_w_pct[idx]), 4) if idx >= 0 and not math.isnan(bb_w_pct[idx]) else "",
            "BB_WIDTH_PCT_AVG_30m": round(float(bb_w_avg[idx]), 4) if idx >= 0 and not math.isnan(bb_w_avg[idx]) else "",
            "BB_SQUEEZE_RATIO_30m": round(float(bb_sq[idx]), 4)    if idx >= 0 and not math.isnan(bb_sq[idx])    else "",
            "EMA_fast_30m":         round(float(ema_fast[idx]), 4) if idx >= 0 and not math.isnan(ema_fast[idx]) else "",
            "EMA_slow_30m":         round(float(ema_slow[idx]), 4) if idx >= 0 and not math.isnan(ema_slow[idx]) else "",
            "trend_30m_ok":         (idx >= 0
                                     and not math.isnan(ema_fast[idx])
                                     and not math.isnan(ema_slow[idx])
                                     and ema_fast[idx] > ema_slow[idx]),
            "prev_5m_close":        round(float(prev_c_full[i]), 4) if not math.isnan(prev_c_full[i]) else "",
            "prev_lower_bb":        round(float(prev_bb_full[i]), 4) if not math.isnan(prev_bb_full[i]) else "",
            "is_cross_5m":          bool(cross[i]),
            "cross_mode":           mode,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    n_cross = int(df["is_cross_5m"].sum())
    print(f"[ok] wrote {len(df)} rows to {OUT_CSV}")
    print(f"     bb_crosses on 5m={n_cross}; cross_mode={mode}")
    if n_cross:
        sub = df[df["is_cross_5m"]][[
            "t_5m_ET", "5m_open", "5m_close",
            "BB_LOWER_30m", "EMA_fast_30m", "EMA_slow_30m", "trend_30m_ok",
            "t_30m_start_ET", "t_30m_window_that_produced_indicator_close_ET",
        ]]
        print("\n--- cross bars ---")
        print(sub.to_string(index=False))

        # Swing high/low look-ahead audit (post-fix uses idx_30m - 1).
        from bt_backtest import _swing_high, _swing_low  # noqa: E402
        print("\n--- swing high/low audit at each cross (TARGET_LOOKBACK="
              f"{cfg.TARGET_LOOKBACK} 30m bars) ---")
        for i in range(n):
            if not bool(cross[i]):
                continue
            idx = int(idx_30m_arr[i])
            sh_buggy   = _swing_high(m30, idx,     cfg.TARGET_LOOKBACK)
            sl_buggy   = _swing_low(m30,  idx,     cfg.TARGET_LOOKBACK)
            sh_postfix = _swing_high(m30, idx - 1, cfg.TARGET_LOOKBACK)
            sl_postfix = _swing_low(m30,  idx - 1, cfg.TARGET_LOOKBACK)
            print(f"  {_fmt_ts(t[i])}  idx_30m={idx}")
            print(f"    BUGGY   (end_idx={idx}):   "
                  f"swing_high={sh_buggy:.4f}  swing_low={sl_buggy:.4f}")
            print(f"    POSTFIX (end_idx={idx - 1}): "
                  f"swing_high={sh_postfix:.4f}  swing_low={sl_postfix:.4f}")
            d_sh = sh_postfix - sh_buggy
            d_sl = sl_postfix - sl_buggy
            print(f"    delta(postfix - buggy):    "
                  f"swing_high={d_sh:+.4f}  swing_low={d_sl:+.4f}")

    # ---- 4. Run the engine for AAPL only and print this day's trade(s) ----
    print("\n--- building cross-symbol inputs (Simple/Complex cutoffs, index MACD) ---")
    daily_only = {}
    if cfg.ENABLE_ROC_RANK_FILTER and getattr(cfg, "ROC_RANK_MODE",
                                              "Simple_Rank") == "Simple_Rank":
        # Simple_Rank needs daily ROC across all watchlist symbols.
        from bt_download import load_watchlist
        for sym in load_watchlist():
            d = _compute_daily(_load(sym, "daily"))
            if not d.empty:
                daily_only[sym] = d
    roc_cutoff_by_day = _build_roc_cutoffs(daily_only) if daily_only else {}
    complex_cutoffs = _build_complex_rank_cutoffs()
    index_macd_by_day = _build_index_macd()

    from datetime import time as dtime
    args = (
        SYMBOL,
        [TARGET_DAY],
        roc_cutoff_by_day,
        index_macd_by_day,
        complex_cutoffs,
        dtime.fromisoformat(cfg.ENTRY_TIME_AFTER),
        dtime.fromisoformat(cfg.ENTRY_TIME_BEFORE),
        pd.Timedelta(minutes=cfg.ENTRY_COOLDOWN_MINUTES),
    )

    print(f"\n--- running _simulate_one_symbol for {SYMBOL} on {TARGET_DAY} only ---")
    result = _simulate_one_symbol(args)

    sigs = [s for s in result["signals"] if s.get("date") == str(TARGET_DAY)]
    trades = [t_ for t_ in result["trades"]
              if str(t_.get("entry_time", ""))[:10] == TARGET_DAY.isoformat()
              or str(t_.get("exit_time", ""))[:10] == TARGET_DAY.isoformat()]

    print(f"\nsignals on {TARGET_DAY}: {len(sigs)}")
    for s in sigs:
        print(
            f"  {s['signal_time']}  cross={s.get('entry_cross_mode')}  "
            f"prev_close={s.get('prev_close')}  prev_lower_bb={s.get('prev_lower_bb')}  "
            f"bar_open={s.get('bar_open')}  bar_close={s.get('bar_close')}  "
            f"lower_bb_30m={s.get('lower_bb')}  "
            f"ema100_30m={s.get('ema100_30m')}  ema200_30m={s.get('ema200_30m')}  "
            f"traded={s.get('traded')}  skip={s.get('skip_reason')}")

    print(f"\ntrades touching {TARGET_DAY}: {len(trades)}")
    for tr in trades:
        keys = [
            "symbol", "parent_id", "leg", "shares",
            "entry_time", "entry_price", "lower_bb",
            "ema100_30m", "ema200_30m",
            "swing_high_at_entry", "swing_low_at_entry", "fib_target_at_entry",
            "stop_price", "target_kind", "target_price",
            "exit_time", "exit_price", "exit_reason",
            "gross_pnl", "commission", "net_pnl",
            "bb_width_pct_30m", "bb_squeeze_ratio_30m", "bb_vs_atr_ratio",
        ]
        print("  " + " | ".join(f"{k}={tr.get(k)}" for k in keys if k in tr))


if __name__ == "__main__":
    main()
