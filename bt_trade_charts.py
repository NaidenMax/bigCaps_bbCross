"""Intraday trade setup charts for the BB 5-Min Cross backtest.

Produces a single self-contained HTML file at
``backtest/output/HTML_Reports/trade_charts.html`` with one card per entry
(one ``parent_id``).  Each card has:

    - 30-min candlestick chart for ``[entry_day - PAD, last_exit_day + PAD]``
        - BB(20,2) bands (the screening BB)
        - Daily EMA(21) and EMA(8) step-lines forward-filled across intraday
          bars (the EMA values the engine uses for its end-of-day exit)
        - Translucent fib-range rectangle from swing_low to swing_high (when
          the trade is trim-and-trail with both leg-1 and leg-2 targets);
          dotted lines at swing_low / swing_high / fib_1.272 with right-edge
          annotations
        - Cyan triangle at the entry bar (snapped to the nearest 30-min slot)
        - Coloured square per leg at its exit bar with refined label and
          shares + net P&L on hover
    - HTML table beneath the chart listing every leg of the parent (entry,
      shares, exit, refined reason, gross & net P&L) with a totals footer

Trade selection (see ``bt_config.TRADE_CHARTS_*``):
    - all parents whose entry is within the last ``TRADE_CHARTS_RECENT_DAYS`` days
    - plus the top-N and bottom-N parents by realised % return from older trades
    - deduped on ``parent_id`` and sorted newest first
    - if ``cfg.SHOW_CLOSED_TRADES_ONLY`` (default True), only parents whose
      every leg has truly exited are rendered: parents with NaT exits OR with
      any leg whose ``exit_reason == 'open_at_end'`` (mark-to-market at
      backtest end) are excluded

Data is loaded from local Parquet via the same helpers the backtest uses — no
Polygon redownload.  Plotly is loaded from CDN so the file opens anywhere.

Usage:
    python bt_trade_charts.py              # full selection from trades.csv
    python bt_trade_charts.py AAPL_2025-09-12T10:05:00-04:00   # single parent_id
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

import bt_config as cfg
from bt_backtest import _compute_30min, _compute_daily, _load

# =====================================================================
# PATHS
# =====================================================================
HTML_DIR = os.path.join(cfg.OUTPUT_DIR, "HTML_Reports")
TRADES_CSV = os.path.join(cfg.OUTPUT_DIR, "trades.csv")

# =====================================================================
# STYLING (kept local so this module doesn't depend on bt_report internals)
# =====================================================================
_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0f0f1a; color:#e0e0e0; font-family:'Segoe UI',system-ui,sans-serif; padding:24px; }
h1 { font-size:1.4rem; margin-bottom:4px; color:#ffffff; }
.subtitle { color:#8888aa; font-size:0.85rem; margin-bottom:20px; }
.legend-row { display:flex; gap:20px; flex-wrap:wrap; margin-bottom:20px;
              font-size:0.78rem; color:#8888aa; }
.legend-item { display:flex; align-items:center; gap:6px; }
.nav-hint { font-size:0.75rem; margin: -10px 0 16px; }
.legend-swatch { width:24px; height:3px; border-radius:2px; }
.chart-card { background:#1a1a2e; border-radius:8px; padding:16px 16px 12px;
              margin-bottom:16px; border:1px solid #2a2a4a; }
.chart-header { margin-bottom:2px; display:flex; align-items:baseline;
                gap:12px; flex-wrap:wrap; }
.ticker { font-size:1.1rem; font-weight:700; color:#ffffff; }
.date { font-size:0.9rem; color:#8888aa; }
.summary { font-size:0.85rem; font-weight:600; }
.meta { font-size:0.75rem; color:#888; margin-bottom:6px; }
.legs { font-size:0.75rem; color:#aaa; margin-bottom:6px; }
.setup-levels { font-size:0.78rem; color:#cdd; background:#10101e;
                border:1px solid #2a2a4a; border-radius:4px;
                padding:4px 10px; margin:6px 0 10px;
                font-family:'Consolas','SF Mono',monospace; }
.win  { color:#4caf50; }
.loss { color:#ef5350; }
.muted { color:#8888aa; }
.tbl { width:100%; border-collapse:collapse; font-size:0.78rem;
       background:#16162b; border:1px solid #2a2a4a; margin-top:10px;
       font-family: 'Consolas','SF Mono',monospace; }
.tbl th { background:#10101e; color:#aaaacc; font-weight:600; padding:5px 8px;
          text-align:right; border-bottom:1px solid #2a2a4a; }
.tbl td { padding:4px 8px; text-align:right; border-bottom:1px solid #1e1e35; }
.tbl tr:last-child td { border-bottom:none; }
.tbl tfoot td { background:#10101e; font-weight:600; border-top:2px solid #2a2a4a; }
.tbl th.left, .tbl td.left { text-align:left; }
"""

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# Colours used for the EMA overlays (kept distinct from BB band colours so the
# eye can separate intraday-screening lines from end-of-day-exit lines).
_EMA21_COLOR = "#ffa726"   # orange
_EMA8_COLOR  = "#fdd835"   # yellow
_FIB_BOX_FILL = "rgba(171,71,188,0.10)"
_FIB_BOX_LINE = "#ab47bc"
_SWING_LO_COLOR = "#90a4ae"

# =====================================================================
# TRADES — load & aggregate to parent level
# =====================================================================

def _load_trades() -> pd.DataFrame:
    if not os.path.isfile(TRADES_CSV):
        print(f"Missing {TRADES_CSV}. Run the backtest first.")
        sys.exit(1)
    df = pd.read_csv(TRADES_CSV)
    # Timestamps in trades.csv include mixed tz offsets and NaT entries (open
    # legs), so we parse with utc=True -> tz-convert -> drop tz for simpler
    # comparisons below.
    for col in ("date", "entry_time", "exit_time"):
        if col not in df.columns:
            continue
        s = pd.to_datetime(df[col], utc=True, errors="coerce")
        if getattr(s.dtype, "tz", None) is not None:
            s = s.dt.tz_convert("US/Eastern").dt.tz_localize(None)
        df[col] = s
    return df


def _aggregate_parents(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-leg trades.csv to one row per parent_id."""
    if trades.empty:
        return trades
    g = trades.groupby("parent_id", sort=False)
    parents = g.agg(
        symbol=("symbol", "first"),
        entry_time=("entry_time", "first"),
        entry_price=("entry_price", "first"),
        date=("date", "first"),
        last_exit_time=("exit_time", "max"),
        n_open_legs=("exit_time", lambda s: int(s.isna().sum())),
        n_mtm_legs=("exit_reason", lambda s: int((s == "open_at_end").sum())),
        parent_net_pnl=("net_pnl", "sum"),
        parent_cost=("position_value", "sum"),
        n_legs=("leg", "count"),
    ).reset_index()
    parents["parent_pct"] = np.where(
        parents["parent_cost"] > 0,
        parents["parent_net_pnl"] / parents["parent_cost"] * 100.0,
        0.0,
    )
    # "Closed" for chart purposes = every leg has a real exit AND no leg was
    # mark-to-market'd at the end of the backtest run.  That second condition
    # is what filters out open_at_end "trades" the user wasn't seeing real
    # strategy outcomes for.
    parents["is_closed"] = (parents["n_open_legs"] == 0) & (parents["n_mtm_legs"] == 0)
    return parents


def _pick_parents(parents: pd.DataFrame) -> pd.DataFrame:
    """Last N days of entries + top/bottom N older entries, deduped."""
    if parents.empty:
        return parents
    cutoff = pd.Timestamp.now().normalize() - timedelta(days=cfg.TRADE_CHARTS_RECENT_DAYS)
    recent = parents[parents["entry_time"] >= cutoff]
    older = parents[parents["entry_time"] < cutoff]
    picks = [recent]
    if not older.empty:
        if cfg.TRADE_CHARTS_TOP_N > 0:
            picks.append(older.nlargest(cfg.TRADE_CHARTS_TOP_N, "parent_pct"))
        if cfg.TRADE_CHARTS_BOTTOM_N > 0:
            picks.append(older.nsmallest(cfg.TRADE_CHARTS_BOTTOM_N, "parent_pct"))
    combined = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["parent_id"])
    return combined.sort_values("entry_time", ascending=False).reset_index(drop=True)


# =====================================================================
# REFINED EXIT-REASON LABELS
# =====================================================================
# The engine emits 5 raw exit_reason values; we map them to 6 user-facing
# labels here (chart-only — engine and bt_report.py aggregations are unchanged).

def _refined_reason(leg: pd.Series) -> str:
    raw = str(leg.get("exit_reason") or "open")
    if raw == "target_swing_high":
        return "L1 Target (swing high)"
    if raw == "target_fib_1272":
        return "L2 Target (fib 1.272)"
    if raw == "ema_exit":
        ema_p = int(leg.get("ema_exit_period") or 0)
        if ema_p == cfg.EMA_EXIT_PERIOD_POST_LEG1:
            return f"L3 Trail (close < EMA{ema_p})"
        return f"Full Stop (close < EMA{ema_p})"
    if raw == "stop":
        ep = float(leg.get("entry_price") or 0)
        sp = float(leg.get("stop_price") or 0)
        if ep > 0 and abs(sp - ep) <= 0.01:
            return "Breakeven Stop"
        return "Hard Stop"
    if raw == "open_at_end":
        return "Open at backtest end"
    return raw


# =====================================================================
# PRICE DATA — per-symbol cache so each symbol loads once
# =====================================================================

class _SymbolCache:
    """Lazy cache of {symbol: (m30_with_bb, daily_with_emas)}."""

    def __init__(self) -> None:
        self._m30: dict[str, pd.DataFrame] = {}
        self._daily: dict[str, pd.DataFrame] = {}

    def get(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if symbol not in self._m30:
            m30 = _load(symbol, "30min")
            self._m30[symbol] = _compute_30min(m30) if not m30.empty else m30
        if symbol not in self._daily:
            daily = _load(symbol, "daily")
            self._daily[symbol] = _compute_daily(daily) if not daily.empty else daily
        return self._m30[symbol], self._daily[symbol]


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    d = df["date"]
    if getattr(d.dtype, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    mask = (d >= start) & (d <= end)
    return df.loc[mask].reset_index(drop=True).assign(date=d[mask].reset_index(drop=True))


def _attach_daily_ema(m30_slice: pd.DataFrame,
                      daily: pd.DataFrame,
                      periods: tuple[int, ...]) -> pd.DataFrame:
    """Forward-fill daily EMA columns onto the 30-min slice as step lines.

    The engine evaluates the EMA exit at end-of-day using that day's close.
    To visualise that level on the intraday chart faithfully, each daily EMA
    becomes "effective" from the start of the NEXT trading session: the value
    you see across a given day's intraday bars is the EMA at the close of the
    PRIOR session — i.e. the level the next end-of-day comparison will be made
    against.  ``merge_asof`` on a shifted ``effective`` column achieves this.
    """
    if daily.empty:
        for p in periods:
            m30_slice[f"DAILY_EMA_{p}"] = np.nan
        return m30_slice
    cols = [c for c in (f"EMA_{p}" for p in periods) if c in daily.columns]
    if not cols:
        for p in periods:
            m30_slice[f"DAILY_EMA_{p}"] = np.nan
        return m30_slice
    daily_em = daily[["date", *cols]].copy()
    if getattr(daily_em["date"].dtype, "tz", None) is not None:
        daily_em["date"] = daily_em["date"].dt.tz_localize(None)
    # Daily bars timestamp at session start (00:00 ET); shifting by +1 day so
    # the asof-backward picks up the prior session's close-EMA for every
    # intraday bar of the current session.
    daily_em["effective"] = daily_em["date"] + pd.Timedelta(days=1)
    daily_em = daily_em.drop(columns=["date"]).sort_values("effective")
    merged = pd.merge_asof(
        m30_slice.sort_values("date"),
        daily_em,
        left_on="date", right_on="effective",
        direction="backward",
    ).drop(columns=["effective"])
    for p in periods:
        merged[f"DAILY_EMA_{p}"] = merged.get(f"EMA_{p}")
    return merged


# =====================================================================
# FIB RANGE GEOMETRY (derived from leg targets)
# =====================================================================

def _fib_geometry(legs: pd.DataFrame) -> dict | None:
    """Return {swing_low, swing_high, fib_target} for a parent.

    Preferred path: read the engine-written fields (added 2026-04) so the
    annotations are exact, even if LEG2_FIB_EXTENSION has changed since the
    trade was placed.  Fallback: back-derive swing_low from L1+L2 target
    prices using the *current* config constant (legacy behaviour for older
    trades.csv files).
    """
    if {"swing_low_at_entry", "swing_high_at_entry"}.issubset(legs.columns):
        any_leg = legs.iloc[0]
        sl = float(any_leg.get("swing_low_at_entry") or 0)
        sh = float(any_leg.get("swing_high_at_entry") or 0)
        fib = float(any_leg.get("fib_target_at_entry") or 0)
        if sl > 0 and sh > sl:
            if fib <= 0:
                fib = sl + float(cfg.LEG2_FIB_EXTENSION) * (sh - sl)
            return {"swing_low": sl, "swing_high": sh, "fib_target": fib}

    l1 = legs[legs["leg"] == 1]
    l2 = legs[legs["leg"] == 2]
    if l1.empty or l2.empty:
        return None
    sh = float(l1["target_price"].iloc[0] or 0)
    fib = float(l2["target_price"].iloc[0] or 0)
    if sh <= 0 or fib <= 0:
        return None
    ext = float(cfg.LEG2_FIB_EXTENSION)
    if ext <= 1.0:
        return None
    sl = (ext * sh - fib) / (ext - 1.0)
    if sl <= 0 or sl >= sh:
        return None
    return {"swing_low": sl, "swing_high": sh, "fib_target": fib}


# =====================================================================
# CHART BUILDER
# =====================================================================

def _fmt_money(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _leg_color(pnl: float) -> str:
    return "#4caf50" if pnl >= 0 else "#ef5350"


def _build_chart_html(parent: pd.Series,
                      legs: pd.DataFrame,
                      m30: pd.DataFrame,
                      daily: pd.DataFrame) -> str | None:
    if m30.empty:
        return None

    sym = parent["symbol"]
    entry_time = pd.Timestamp(parent["entry_time"])
    last_exit = parent.get("last_exit_time")
    window_end = (pd.Timestamp(last_exit) if last_exit is not None and not pd.isna(last_exit)
                  else pd.Timestamp.now())
    pad = timedelta(days=cfg.TRADE_CHARTS_CONTEXT_PAD_DAYS)
    win_start = entry_time.normalize() - pad
    win_end   = window_end.normalize() + pad + timedelta(days=1)

    m30_slice = _slice_window(m30, win_start, win_end)
    if m30_slice.empty:
        return None

    pre_p = cfg.EMA_EXIT_PERIOD_PRE_LEG1
    post_p = cfg.EMA_EXIT_PERIOD_POST_LEG1
    m30_slice = _attach_daily_ema(m30_slice, daily, (pre_p, post_p))

    chart_id = f"chart_{parent['parent_id'].replace(':','').replace('-','').replace('T','_').replace('+','p')}"

    # ------- 30-min traces ------------------------------------------------
    t30 = m30_slice["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    candle30 = {
        "x": t30,
        "open": m30_slice["open"].round(4).tolist(),
        "high": m30_slice["high"].round(4).tolist(),
        "low":  m30_slice["low"].round(4).tolist(),
        "close": m30_slice["close"].round(4).tolist(),
        "type": "candlestick",
        "increasing": {"line": {"color": "#4caf50"}, "fillcolor": "#2e7d3280"},
        "decreasing": {"line": {"color": "#ef5350"}, "fillcolor": "#c6282880"},
        "showlegend": False,
        "xaxis": "x", "yaxis": "y",
        "name": "30-min",
    }
    bb_lower = {
        "x": t30, "y": m30_slice["BB_LOWER"].round(4).tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": "#66bb6a", "width": 1, "dash": "dash"},
        "name": "BB lower", "showlegend": False,
        "xaxis": "x", "yaxis": "y",
        "hovertemplate": "BB lower %{y:.2f}<extra></extra>",
    }
    bb_upper = {
        "x": t30, "y": m30_slice["BB_UPPER"].round(4).tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": "#ab47bc", "width": 1, "dash": "dash"},
        "name": "BB upper", "showlegend": False,
        "xaxis": "x", "yaxis": "y",
        "hovertemplate": "BB upper %{y:.2f}<extra></extra>",
    }
    ema21_trace = {
        "x": t30,
        "y": m30_slice[f"DAILY_EMA_{pre_p}"].round(4).tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _EMA21_COLOR, "width": 1.4, "shape": "hv"},
        "name": f"Daily EMA{pre_p}", "showlegend": False,
        "xaxis": "x", "yaxis": "y",
        "hovertemplate": f"Daily EMA{pre_p} %{{y:.2f}}<extra></extra>",
        "connectgaps": False,
    }
    ema8_trace = {
        "x": t30,
        "y": m30_slice[f"DAILY_EMA_{post_p}"].round(4).tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _EMA8_COLOR, "width": 1.4, "shape": "hv"},
        "name": f"Daily EMA{post_p}", "showlegend": False,
        "xaxis": "x", "yaxis": "y",
        "hovertemplate": f"Daily EMA{post_p} %{{y:.2f}}<extra></extra>",
        "connectgaps": False,
    }

    entry_x = _snap_to_axis(entry_time, t30)
    entry_price = float(parent["entry_price"])
    entry_marker = {
        "x": [entry_x], "y": [entry_price],
        "type": "scatter", "mode": "markers+text",
        "marker": {"symbol": "triangle-up", "size": 14, "color": "#4fc3f7",
                   "line": {"color": "white", "width": 1}},
        "text": ["Entry"], "textposition": "top center",
        "textfont": {"color": "#4fc3f7", "size": 10},
        "showlegend": False,
        "hovertemplate": f"Entry @ {entry_price:.2f}<br>{entry_time}<extra></extra>",
        "xaxis": "x", "yaxis": "y",
    }

    exit_traces = []
    for _, leg in legs.iterrows():
        if pd.isna(leg["exit_time"]):
            continue
        ex_time = pd.Timestamp(leg["exit_time"])
        ex_price = float(leg["exit_price"])
        label_full = _refined_reason(leg)
        ex_x = _snap_to_axis(ex_time, t30)
        color = _leg_color(float(leg["net_pnl"]))
        marker_text = f"L{int(leg['leg'])}"
        exit_traces.append({
            "x": [ex_x], "y": [ex_price],
            "type": "scatter", "mode": "markers+text",
            "marker": {"symbol": "square", "size": 11, "color": color,
                       "line": {"color": "white", "width": 1}},
            "text": [marker_text], "textposition": "bottom center",
            "textfont": {"color": color, "size": 10},
            "showlegend": False,
            "hovertemplate": (f"{marker_text}: {label_full}"
                              f"<br>@ {ex_price:.2f}"
                              f"<br>{ex_time}"
                              f"<br>shares {int(leg['shares'])}"
                              f"<br>net {_fmt_money(float(leg['net_pnl']))}<extra></extra>"),
            "xaxis": "x", "yaxis": "y",
        })

    all_traces = ([bb_lower, bb_upper, ema21_trace, ema8_trace,
                   candle30, entry_marker] + exit_traces)

    # ------- shapes (target lines + fib range) ---------------------------
    shapes: list[dict] = []
    annotations: list[dict] = []
    fib = _fib_geometry(legs)
    if fib is not None:
        shapes.append({
            "type": "rect", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1, "y0": fib["swing_low"], "y1": fib["swing_high"],
            "fillcolor": _FIB_BOX_FILL,
            "line": {"color": _FIB_BOX_LINE, "width": 0},
            "layer": "below",
        })
        for level, color, text in (
            (fib["swing_high"], "#66bb6a", f"swing_high {fib['swing_high']:.2f}"),
            (fib["swing_low"],  _SWING_LO_COLOR, f"swing_low {fib['swing_low']:.2f}"),
            (fib["fib_target"], "#ab47bc", f"fib 1.272 {fib['fib_target']:.2f}"),
        ):
            shapes.append({
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": level, "y1": level,
                "line": {"color": color, "width": 1, "dash": "dot"},
            })
            annotations.append({
                "x": 1.0, "xref": "paper", "y": level, "yref": "y",
                "text": text, "showarrow": False, "xanchor": "right",
                "yanchor": "bottom",
                "font": {"color": color, "size": 9, "family": "monospace"},
            })
    else:
        # Fall back to per-leg target lines when fib geometry isn't available
        # (single-leg trades or trim-and-trail with missing leg-2 target).
        for _, leg in legs.iterrows():
            tgt = float(leg.get("target_price") or 0.0)
            if tgt <= 0:
                continue
            kind = str(leg.get("target_kind") or "")
            color = "#66bb6a" if kind == "swing_high" else "#ab47bc"
            shapes.append({
                "type": "line", "xref": "paper", "yref": "y",
                "x0": 0, "x1": 1, "y0": tgt, "y1": tgt,
                "line": {"color": color, "width": 1, "dash": "dot"},
            })

    # y-axis range — pad above/below so markers, target lines AND EMA overlays
    # all stay inside the visible area.
    y_lo = float(min(m30_slice["low"].min(), entry_price,
                     legs["exit_price"].min() if not legs["exit_price"].isna().all() else entry_price))
    y_hi = float(max(m30_slice["high"].max(), entry_price,
                     legs["exit_price"].max() if not legs["exit_price"].isna().all() else entry_price))
    if fib is not None:
        y_lo = min(y_lo, fib["swing_low"])
        y_hi = max(y_hi, fib["swing_high"], fib["fib_target"])
    for col in (f"DAILY_EMA_{pre_p}", f"DAILY_EMA_{post_p}"):
        s = m30_slice[col].dropna()
        if not s.empty:
            y_lo = min(y_lo, float(s.min()))
            y_hi = max(y_hi, float(s.max()))
    span = max(y_hi - y_lo, 0.01)
    y_lo -= span * 0.05
    y_hi += span * 0.08

    layout = {
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor":  "#1a1a2e",
        "font": {"color": "#aaaacc", "size": 11},
        "margin": {"l": 55, "r": 100, "t": 10, "b": 40},
        "xaxis": {
            "gridcolor": "#2a2a4a", "linecolor": "#2a2a4a",
            "rangeslider": {
                "visible": True,
                "thickness": 0.06,
                "bgcolor": "#10101e",
                "bordercolor": "#2a2a4a",
                "borderwidth": 1,
            },
            "type": "category",
            "nticks": 14,
        },
        "yaxis": {
            "gridcolor": "#2a2a4a", "linecolor": "#2a2a4a",
            "tickprefix": "$",
            "range": [y_lo, y_hi],
            "side": "right",
        },
        "hovermode": "x",
        "shapes": shapes,
        "annotations": annotations,
        "height": 700,
    }

    # ------- card header --------------------------------------------------
    net = float(parent["parent_net_pnl"])
    pct = float(parent["parent_pct"])
    days_held = max(1, int((pd.Timestamp(window_end).normalize()
                            - entry_time.normalize()).days))
    net_cls = "win" if net >= 0 else "loss"

    entry_str = entry_time.strftime("%Y-%m-%d %H:%M")
    exit_str  = (pd.Timestamp(window_end).strftime("%Y-%m-%d")
                 if last_exit is not None and not pd.isna(last_exit) else "open")

    header = (
        f'<div class="chart-header">'
        f'<span class="ticker">{sym}</span>'
        f'<span class="date">{entry_str} &rarr; {exit_str} ({days_held}d held)</span>'
        f'<span class="summary {net_cls}">{pct:+.2f}% &nbsp; {_fmt_money(net)}</span>'
        f'</div>'
        f'<div class="meta muted">parent_id: {parent["parent_id"]}</div>'
    )

    # ------- setup-levels caption (one-line readout of geometry) ---------
    first_leg = legs.iloc[0]
    entry_px = float(first_leg.get("entry_price") or 0)
    stop_px  = float(first_leg.get("stop_price") or 0)
    parts: list[str] = [f"Entry {entry_px:.2f}"]

    pct_in_range = first_leg.get("entry_pct_in_range")
    pct_vs_6mo   = first_leg.get("pct_from_6mo_high")
    extras: list[str] = []
    if pct_in_range is not None and not pd.isna(pct_in_range):
        extras.append(f"{float(pct_in_range):.0f}% in range")
    if pct_vs_6mo is not None and not pd.isna(pct_vs_6mo):
        extras.append(f"{float(pct_vs_6mo):+.1f}% vs 6mo high")
    if extras:
        parts[0] = f"Entry {entry_px:.2f} ({', '.join(extras)})"

    if stop_px > 0:
        parts.append(f"Stop {stop_px:.2f}")
    if fib is not None:
        parts.append(f"Swing low {fib['swing_low']:.2f}")
        parts.append(f"Swing high {fib['swing_high']:.2f}")
        parts.append(f"Fib 1.272 {fib['fib_target']:.2f}")
    setup_caption = (
        f'<div class="setup-levels">{" &nbsp;&middot;&nbsp; ".join(parts)}</div>'
    )

    # ------- per-leg trade table -----------------------------------------
    table = _build_legs_table(parent, legs)

    return f"""
<div class="chart-card">
{header}
{setup_caption}
<div id="{chart_id}"></div>
<script>
Plotly.newPlot('{chart_id}', {json.dumps(all_traces, default=_json_safe)},
               {json.dumps(layout)},
               {{responsive:true, displayModeBar:false}});
</script>
{table}
</div>
"""


def _format_held(total_minutes: int) -> str:
    """Mirror of bt_backtest._format_held so older trades.csv files (without
    the period_held column) still render a sensible value in the chart table.
    """
    if total_minutes <= 0:
        return "0m"
    days, rem = divmod(total_minutes, 24 * 60)
    hours, mins = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _leg_period_held(leg: pd.Series) -> str:
    """Prefer the engine-written period_held; fall back to live compute so
    charts stay useful against an older trades.csv."""
    raw = leg.get("period_held")
    if isinstance(raw, str) and raw and raw != "nan":
        return raw
    if pd.isna(leg.get("exit_time")) or pd.isna(leg.get("entry_time")):
        return "-"
    try:
        td = pd.Timestamp(leg["exit_time"]) - pd.Timestamp(leg["entry_time"])
        mins = max(0, int(td.total_seconds() // 60))
        return _format_held(mins)
    except Exception:
        return "-"


def _build_legs_table(parent: pd.Series, legs: pd.DataFrame) -> str:
    """Render the per-leg trade summary table."""
    rows: list[str] = []
    total_shares = 0
    total_cost = 0.0
    total_gross = 0.0
    total_net = 0.0
    for _, leg in legs.sort_values("leg").iterrows():
        leg_no = int(leg["leg"])
        et = pd.Timestamp(leg["entry_time"]).strftime("%Y-%m-%d %H:%M") \
            if not pd.isna(leg["entry_time"]) else "-"
        ep = float(leg.get("entry_price") or 0)
        shares = int(leg.get("shares") or 0)
        cost = float(leg.get("position_value") or 0)
        if pd.isna(leg["exit_time"]):
            xt = "-"
            xp = "-"
        else:
            xt = pd.Timestamp(leg["exit_time"]).strftime("%Y-%m-%d %H:%M")
            xp = f"{float(leg['exit_price']):.2f}"
        held = _leg_period_held(leg)
        reason = _refined_reason(leg)
        gross = float(leg.get("gross_pnl") or 0)
        net   = float(leg.get("net_pnl") or 0)
        pct = (net / cost * 100.0) if cost > 0 else 0.0
        cls = "win" if net >= 0 else "loss"
        rows.append(
            f'<tr>'
            f'<td>L{leg_no}</td>'
            f'<td class="left">{et}</td>'
            f'<td>{ep:.2f}</td>'
            f'<td>{shares:,}</td>'
            f'<td>{_fmt_money(cost)}</td>'
            f'<td class="left">{xt}</td>'
            f'<td>{xp}</td>'
            f'<td>{held}</td>'
            f'<td class="left">{reason}</td>'
            f'<td class="{cls}">{_fmt_money(gross)}</td>'
            f'<td class="{cls}">{_fmt_money(net)}</td>'
            f'<td class="{cls}">{pct:+.2f}%</td>'
            f'</tr>'
        )
        total_shares += shares
        total_cost += cost
        total_gross += gross
        total_net += net

    total_pct = (total_net / total_cost * 100.0) if total_cost > 0 else 0.0
    total_cls = "win" if total_net >= 0 else "loss"
    # Parent lifespan = first entry to last exit across all legs.  Open legs
    # render as "-" (the parent isn't really closed yet).
    try:
        entries = pd.to_datetime(legs["entry_time"], errors="coerce").dropna()
        exits   = pd.to_datetime(legs["exit_time"], errors="coerce").dropna()
        if not entries.empty and not exits.empty and len(exits) == len(legs):
            span_min = max(0, int((exits.max() - entries.min()).total_seconds() // 60))
            parent_held = _format_held(span_min)
        else:
            parent_held = "-"
    except Exception:
        parent_held = "-"
    foot = (
        f'<tr>'
        f'<td colspan="3" class="left">Totals ({len(legs)} legs)</td>'
        f'<td>{total_shares:,}</td>'
        f'<td>{_fmt_money(total_cost)}</td>'
        f'<td colspan="2"></td>'
        f'<td>{parent_held}</td>'
        f'<td></td>'
        f'<td class="{total_cls}">{_fmt_money(total_gross)}</td>'
        f'<td class="{total_cls}">{_fmt_money(total_net)}</td>'
        f'<td class="{total_cls}">{total_pct:+.2f}%</td>'
        f'</tr>'
    )

    return f"""
<table class="tbl">
<thead>
<tr>
<th>Leg</th><th class="left">Entry time</th><th>Entry $</th>
<th>Shares</th><th>Cost</th>
<th class="left">Exit time</th><th>Exit $</th>
<th>Period held</th>
<th class="left">Reason</th>
<th>Gross P&amp;L</th><th>Net P&amp;L</th><th>Pct</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
<tfoot>{foot}</tfoot>
</table>
"""


def _json_safe(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def _snap_to_axis(ts: pd.Timestamp, axis_strs: list[str]) -> str:
    """Return the axis category string nearest to ``ts``.

    The 30-min chart uses a categorical x-axis of ``YYYY-MM-DD HH:MM`` strings,
    so a marker placed at an out-of-grid timestamp would silently land at the
    wrong slot.  We snap to the nearest bar instead.
    """
    if not axis_strs:
        return ts.strftime("%Y-%m-%d %H:%M")
    axis_ts = pd.to_datetime(axis_strs).values.astype("datetime64[ns]")
    target = np.datetime64(pd.Timestamp(ts).to_datetime64(), "ns")
    deltas = np.abs((axis_ts - target).astype("timedelta64[s]").astype(np.int64))
    idx = int(np.argmin(deltas))
    return axis_strs[idx]


# =====================================================================
# HTML ASSEMBLY
# =====================================================================

def _generate_html(cards: list[str], n_total_parents: int, n_eligible: int) -> str:
    os.makedirs(HTML_DIR, exist_ok=True)
    out = os.path.join(HTML_DIR, cfg.TRADE_CHARTS_FILENAME)
    closed_note = ("closed only (excl. open & open_at_end)"
                   if cfg.SHOW_CLOSED_TRADES_ONLY else "open + closed")
    pre_p = cfg.EMA_EXIT_PERIOD_PRE_LEG1
    post_p = cfg.EMA_EXIT_PERIOD_POST_LEG1
    subtitle = (
        f"{len(cards)} of {n_eligible} eligible parents ({n_total_parents} total) "
        f"&nbsp;|&nbsp; selection: last {cfg.TRADE_CHARTS_RECENT_DAYS}d "
        f"+ top {cfg.TRADE_CHARTS_TOP_N} / bottom {cfg.TRADE_CHARTS_BOTTOM_N} "
        f"&nbsp;|&nbsp; {closed_note} "
        f"&nbsp;|&nbsp; pad &plusmn;{cfg.TRADE_CHARTS_CONTEXT_PAD_DAYS}d"
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BB 5-Min Cross — Trade Charts</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
<h1>BB 5-Min Cross &mdash; Trade Charts</h1>
<div class="subtitle">{subtitle}</div>
<div class="legend-row">
  <div class="legend-item"><div class="legend-swatch" style="background:#66bb6a"></div> BB lower (30m)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:#ab47bc"></div> BB upper (30m)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:{_EMA21_COLOR}"></div> Daily EMA{pre_p} (full-stop ref)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:{_EMA8_COLOR}"></div> Daily EMA{post_p} (post-L1 trail)</div>
  <div class="legend-item"><div class="legend-swatch" style="background:{_FIB_BOX_FILL};border:1px solid {_FIB_BOX_LINE}"></div> Fib range (swing_low &rarr; swing_high)</div>
  <div class="legend-item"><span style="color:#4fc3f7;font-size:1.1rem">&#9650;</span> Entry</div>
  <div class="legend-item"><span style="color:#4caf50;font-size:1.1rem">&#9632;</span> Exit (win)</div>
  <div class="legend-item"><span style="color:#ef5350;font-size:1.1rem">&#9632;</span> Exit (loss)</div>
</div>
<div class="nav-hint muted">Tip: drag the strip below each chart (or its handles) to scrub through time / zoom the view.</div>
{''.join(cards)}
</body>
</html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


# =====================================================================
# ENTRY POINT
# =====================================================================

def main(parent_id: str | None = None) -> None:
    print("=" * 60)
    print("GENERATING TRADE CHARTS")
    print("=" * 60)

    trades = _load_trades()
    if trades.empty:
        print("No trades in trades.csv.")
        return

    parents = _aggregate_parents(trades)
    n_total = len(parents)

    if getattr(cfg, "SHOW_CLOSED_TRADES_ONLY", True):
        eligible = parents[parents["is_closed"]].copy()
        n_open = int((parents["n_open_legs"] > 0).sum())
        n_mtm  = int(((parents["n_mtm_legs"] > 0) & (parents["n_open_legs"] == 0)).sum())
        if n_open or n_mtm:
            print(f"SHOW_CLOSED_TRADES_ONLY: skipping {n_open} open + {n_mtm} mark-to-market parents")
    else:
        eligible = parents
    n_eligible = len(eligible)

    if parent_id:
        subset = eligible[eligible["parent_id"] == parent_id]
        if subset.empty:
            print(f"No (eligible) parent_id matching {parent_id!r}.")
            return
        picks = subset
    else:
        picks = _pick_parents(eligible)

    if picks.empty:
        print("Nothing to chart (selection empty).")
        return

    print(f"Charting {len(picks)} of {n_eligible} eligible entries...")

    cache = _SymbolCache()
    cards: list[str] = []
    for _, parent in picks.iterrows():
        legs = trades[trades["parent_id"] == parent["parent_id"]].copy()
        m30, daily = cache.get(parent["symbol"])
        if m30.empty:
            print(f"  skip {parent['symbol']} {parent['entry_time']} (no parquet)")
            continue
        card = _build_chart_html(parent, legs, m30, daily)
        if card:
            cards.append(card)
            print(f"  {parent['symbol']} {parent['entry_time']}  "
                  f"{parent['parent_pct']:+.2f}%")

    if not cards:
        print("No cards rendered.")
        return

    out = _generate_html(cards, n_total, n_eligible)
    print(f"Wrote {out}")
    print("=" * 60)


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) >= 2 else None
    main(pid)
