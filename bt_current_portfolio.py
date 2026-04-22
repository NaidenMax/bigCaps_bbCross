"""Current portfolio dashboard for the BB 5-Min Cross backtest.

Produces ``backtest/output/HTML_Reports/current_portfolio.html`` showing all
positions still open at the end of the backtest run, mark-to-market against
the last available bar.

Source of truth: rows in ``trades.csv`` whose ``exit_reason == 'open_at_end'``.
Each such row already carries the leg's full state at backtest end (entry,
shares, current stop, target, unrealised P&L, swing/fib levels, last close
mark) so this module is pure read-side --- no engine or CSV-schema changes.

Layout:
    1. KPI card grid: open trades, open legs, capital deployed, unrealised
       P&L $/%, best/worst leg %, as-of timestamp.
    2. Portfolio table: one row per parent_id, sorted by unrealised $ desc.
    3. Per-position cards (collapsible): a 30-min price chart with daily EMA
       overlays + swing/fib reference levels + current stop, plus a per-leg
       table (shares, entry, current $, stop, target, unrealised).

Usage:
    python bt_current_portfolio.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

import bt_config as cfg
from bt_trade_charts import (
    _SymbolCache, _slice_window, _attach_daily_ema, _snap_to_axis,
    _fib_geometry, _json_safe,
)

# =====================================================================
# PATHS
# =====================================================================
HTML_DIR   = os.path.join(cfg.OUTPUT_DIR, "HTML_Reports")
TRADES_CSV = os.path.join(cfg.OUTPUT_DIR, "trades.csv")
OUT_FILE   = os.path.join(HTML_DIR, "current_portfolio.html")

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# Match the colour palette of the rest of the report family.
_EMA21_COLOR    = "#ffa726"
_EMA8_COLOR     = "#fdd835"
_FIB_BOX_FILL   = "rgba(171,71,188,0.10)"
_FIB_BOX_LINE   = "#ab47bc"
_SWING_LO_COLOR = "#90a4ae"
_STOP_COLOR     = "#ef5350"
_LAST_COLOR     = "#4fc3f7"
_EXIT_COLOR     = "#ef5350"   # red down-triangles for closed sibling legs

# How many days of context to pad before entry / after the last bar.
_PAD_BEFORE_DAYS = 3
_PAD_AFTER_DAYS  = 1

# =====================================================================
# CSS
# =====================================================================
_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0f0f1a; color:#e0e0e0;
       font-family:'Segoe UI',system-ui,sans-serif; padding:24px; }
h1 { font-size:1.5rem; margin-bottom:4px; color:#ffffff; }
h2 { font-size:1.05rem; font-weight:600; margin:24px 0 10px; color:#ccccdd;
     border-bottom:1px solid #2a2a4a; padding-bottom:6px; }
.subtitle { color:#8888aa; font-size:0.88rem; margin-bottom:20px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:12px; margin-bottom:24px; }
.card { background:#1a1a2e; border-radius:8px; padding:14px; text-align:center;
        border:1px solid #2a2a4a; }
.card-label { font-size:0.7rem; color:#8888aa; text-transform:uppercase;
              letter-spacing:0.5px; margin-bottom:6px; }
.card-value { font-size:1.15rem; font-weight:600; color:#e0e0e0; }
.card-sub   { font-size:0.7rem; color:#8888aa; margin-top:2px; }
.pos { color:#4caf50; font-weight:600; }
.neg { color:#ef5350; font-weight:600; }
.muted { color:#8888aa; }
.tbl { width:100%; border-collapse:collapse; font-size:0.82rem;
       background:#1a1a2e; border:1px solid #2a2a4a; }
.tbl th { background:#16162b; padding:7px 9px; text-align:right;
          color:#aaaacc; font-weight:600; border-bottom:2px solid #2a2a4a; }
.tbl td { border-bottom:1px solid #1e1e35; padding:5px 9px; text-align:right; }
.tbl tr:hover td { background:#1e1e35; }
.tbl th.left, .tbl td.left { text-align:left; }
.win  { color:#4caf50; font-weight:600; }
.loss { color:#ef5350; font-weight:600; }
.pos-card { background:#1a1a2e; border-radius:8px; padding:12px 14px;
            margin-bottom:12px; border:1px solid #2a2a4a; }
.pos-card summary { cursor:pointer; list-style:none; outline:none;
                    display:flex; align-items:baseline; gap:14px;
                    flex-wrap:wrap; }
.pos-card summary::-webkit-details-marker { display:none; }
.pos-card summary::before { content:"\\25B8"; color:#8888aa; font-size:0.85rem;
                            display:inline-block; width:14px;
                            transition:transform 0.15s; }
.pos-card[open] summary::before { transform:rotate(90deg); }
.ticker  { font-size:1.05rem; font-weight:700; color:#ffffff; }
.entry-date { font-size:0.85rem; color:#8888aa; }
.summary-pnl { font-size:0.9rem; font-weight:600; margin-left:auto; }
.setup-levels { font-size:0.78rem; color:#cdd; background:#10101e;
                border:1px solid #2a2a4a; border-radius:4px;
                padding:4px 10px; margin:8px 0;
                font-family:'Consolas','SF Mono',monospace; }
.legs-table { font-family:'Consolas','SF Mono',monospace; font-size:0.78rem; }
.section { margin-top:24px; }
"""


# =====================================================================
# FORMATTING
# =====================================================================
def _fmt_money(v: float, dp: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.{dp}f}"


def _fmt_pct(v: float, dp: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:+.{dp}f}%" if v != 0 else f"{v:.{dp}f}%"


def _fmt_int(v) -> str:
    try:
        return f"{int(v):,}"
    except Exception:
        return str(v)


def _signed_class(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "muted"
    return "pos" if v > 0 else ("neg" if v < 0 else "muted")


def _card(label: str, value: str, sub: str = "", value_class: str = "") -> str:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    cls = f' class="card-value {value_class}"' if value_class else ' class="card-value"'
    return (f'<div class="card"><div class="card-label">{label}</div>'
            f'<div{cls}>{value}</div>{sub_html}</div>')


# =====================================================================
# DATA LOAD
# =====================================================================
def _load_trades() -> pd.DataFrame:
    """Load full trades.csv with timezone-normalised datetime columns."""
    if not os.path.isfile(TRADES_CSV):
        print(f"Missing {TRADES_CSV}. Run the backtest first.")
        return pd.DataFrame()
    df = pd.read_csv(TRADES_CSV)
    if df.empty or "exit_reason" not in df.columns:
        return pd.DataFrame()
    for col in ("date", "entry_time", "exit_time"):
        if col in df.columns:
            s = pd.to_datetime(df[col], utc=True, errors="coerce")
            if getattr(s.dtype, "tz", None) is not None:
                s = s.dt.tz_convert("US/Eastern").dt.tz_localize(None)
            df[col] = s
    df["unreal_pct"] = np.where(
        df["position_value"] > 0,
        df["net_pnl"] / df["position_value"] * 100.0,
        0.0,
    )
    return df


def _filter_open_legs(trades: pd.DataFrame) -> pd.DataFrame:
    """Return only legs still open at backtest end (exit_reason == 'open_at_end')."""
    if trades.empty:
        return trades
    return trades[trades["exit_reason"] == "open_at_end"].copy()


def _filter_parent_legs(trades: pd.DataFrame, parent_ids: set) -> pd.DataFrame:
    """Return ALL legs (open + closed siblings) for the given parent_ids."""
    if trades.empty or not parent_ids:
        return trades.iloc[0:0].copy()
    return trades[trades["parent_id"].isin(parent_ids)].copy()


def _load_open_legs() -> pd.DataFrame:
    """Backwards-compat shim — used by the empty-state branch of main()."""
    return _filter_open_legs(_load_trades())


def _aggregate_open_parents(legs: pd.DataFrame) -> pd.DataFrame:
    """One row per parent_id with portfolio-table fields."""
    if legs.empty:
        return legs
    g = legs.groupby("parent_id", sort=False)
    parents = g.agg(
        symbol=("symbol", "first"),
        entry_time=("entry_time", "first"),
        as_of_time=("exit_time", "max"),
        last_price=("exit_price", "first"),
        n_legs=("leg", "count"),
        shares=("shares", "sum"),
        cost=("position_value", "sum"),
        unreal_gross=("gross_pnl", "sum"),
        unreal_net=("net_pnl", "sum"),
    ).reset_index()
    parents["mtm"] = parents["last_price"] * parents["shares"]
    parents["unreal_pct"] = np.where(
        parents["cost"] > 0,
        parents["unreal_net"] / parents["cost"] * 100.0,
        0.0,
    )
    parents["avg_entry"] = np.where(
        parents["shares"] > 0,
        (legs.groupby("parent_id", sort=False)
              .apply(lambda d: (d["entry_price"] * d["shares"]).sum()
                              / d["shares"].sum())
              .reindex(parents["parent_id"]).values),
        0.0,
    )
    parents["days_held"] = (
        pd.to_datetime(parents["as_of_time"]).dt.normalize()
        - pd.to_datetime(parents["entry_time"]).dt.normalize()
    ).dt.days.clip(lower=0)
    parents = parents.sort_values("unreal_net", ascending=False).reset_index(drop=True)
    return parents


def _portfolio_kpis(legs: pd.DataFrame, parents: pd.DataFrame) -> dict:
    """KPI numbers for the top card grid."""
    if legs.empty:
        return {"n_trades": 0, "n_legs": 0, "cost": 0.0, "mtm": 0.0,
                "unreal_net": 0.0, "unreal_pct": 0.0,
                "best_leg_pct": float("nan"), "best_leg_sym": "",
                "worst_leg_pct": float("nan"), "worst_leg_sym": "",
                "as_of": ""}
    cost = float(legs["position_value"].sum())
    unreal_net = float(legs["net_pnl"].sum())
    mtm = float((legs["exit_price"] * legs["shares"]).sum())
    best_idx  = int(legs["unreal_pct"].idxmax())
    worst_idx = int(legs["unreal_pct"].idxmin())
    as_of = pd.to_datetime(legs["exit_time"]).max()
    return {
        "n_trades":      int(parents["parent_id"].nunique()) if not parents.empty else 0,
        "n_legs":        int(len(legs)),
        "cost":          cost,
        "mtm":           mtm,
        "unreal_net":    unreal_net,
        "unreal_pct":    (unreal_net / cost * 100.0) if cost > 0 else 0.0,
        "best_leg_pct":  float(legs.loc[best_idx, "unreal_pct"]),
        "best_leg_sym":  str(legs.loc[best_idx, "symbol"]),
        "worst_leg_pct": float(legs.loc[worst_idx, "unreal_pct"]),
        "worst_leg_sym": str(legs.loc[worst_idx, "symbol"]),
        "as_of":         as_of.strftime("%Y-%m-%d %H:%M") if pd.notna(as_of) else "",
    }


# =====================================================================
# KPI GRID + PORTFOLIO TABLE
# =====================================================================
def _build_kpi_grid(k: dict) -> str:
    return '<div class="cards">' + "".join([
        _card("Open Trades", _fmt_int(k["n_trades"]),
              sub=f'{_fmt_int(k["n_legs"])} legs'),
        _card("Capital Deployed", _fmt_money(k["cost"], 0)),
        _card("Mark-to-Market", _fmt_money(k["mtm"], 0)),
        _card("Unrealised P&L", _fmt_money(k["unreal_net"], 0),
              value_class=_signed_class(k["unreal_net"])),
        _card("Unrealised %", _fmt_pct(k["unreal_pct"]),
              value_class=_signed_class(k["unreal_pct"])),
        _card("Best Leg %", _fmt_pct(k["best_leg_pct"]),
              sub=k["best_leg_sym"], value_class="pos"),
        _card("Worst Leg %", _fmt_pct(k["worst_leg_pct"]),
              sub=k["worst_leg_sym"], value_class="neg"),
        _card("As of", k["as_of"] or "—"),
    ]) + "</div>"


def _build_portfolio_table(parents: pd.DataFrame) -> str:
    if parents.empty:
        return '<p class="muted">No open positions.</p>'
    rows: list[str] = []
    for _, p in parents.iterrows():
        cls = "win" if p["unreal_net"] >= 0 else "loss"
        et = pd.Timestamp(p["entry_time"]).strftime("%Y-%m-%d") \
            if pd.notna(p["entry_time"]) else "—"
        rows.append(
            f'<tr>'
            f'<td class="left"><a href="#pos_{p["parent_id"]}" '
            f'style="color:#4fc3f7;text-decoration:none;">{p["symbol"]}</a></td>'
            f'<td class="left">{et}</td>'
            f'<td>{float(p["avg_entry"]):.2f}</td>'
            f'<td>{float(p["last_price"]):.2f}</td>'
            f'<td>{_fmt_int(p["shares"])}</td>'
            f'<td>{_fmt_money(p["cost"], 0)}</td>'
            f'<td>{_fmt_money(p["mtm"], 0)}</td>'
            f'<td class="{cls}">{_fmt_money(p["unreal_net"], 0)}</td>'
            f'<td class="{cls}">{_fmt_pct(float(p["unreal_pct"]))}</td>'
            f'<td>{int(p["days_held"])}</td>'
            f'<td>{int(p["n_legs"])}</td>'
            f'</tr>'
        )
    return f"""
<table class="tbl">
<thead><tr>
<th class="left">Symbol</th><th class="left">Entry</th>
<th>Avg Entry $</th><th>Last $</th><th>Shares</th>
<th>Cost</th><th>MTM</th>
<th>Unrealised $</th><th>Unrealised %</th>
<th>Days held</th><th>Legs</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


# =====================================================================
# PER-POSITION CHART
# =====================================================================
def _build_position_chart(parent: pd.Series, legs: pd.DataFrame,
                          all_legs: pd.DataFrame,
                          m30: pd.DataFrame, daily: pd.DataFrame) -> str | None:
    """Return Plotly chart HTML (script + div) for one open position.

    ``legs`` holds only the still-open legs (exit_reason == 'open_at_end')
    and drives the entry / mark-to-market markers.  ``all_legs`` holds every
    leg of this parent (including closed siblings whose targets / stops have
    already filled) and drives the red down-triangle exit markers so the
    chart shows the full lifecycle of a multi-leg trade.
    """
    if m30.empty:
        return None

    entry_time = pd.Timestamp(parent["entry_time"])
    as_of      = pd.Timestamp(parent["as_of_time"])
    win_start  = entry_time.normalize() - timedelta(days=_PAD_BEFORE_DAYS)
    win_end    = as_of.normalize() + timedelta(days=_PAD_AFTER_DAYS + 1)

    m30_slice = _slice_window(m30, win_start, win_end)
    if m30_slice.empty:
        return None

    pre_p  = cfg.EMA_EXIT_PERIOD_PRE_LEG1
    post_p = cfg.EMA_EXIT_PERIOD_POST_LEG1
    m30_slice = _attach_daily_ema(m30_slice, daily, (pre_p, post_p))

    chart_id = ("chart_" + str(parent["parent_id"])
                .replace(":", "").replace("-", "")
                .replace("T", "_").replace("+", "p"))

    t30 = m30_slice["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    candle = {
        "x": t30,
        "open":  m30_slice["open"].round(4).tolist(),
        "high":  m30_slice["high"].round(4).tolist(),
        "low":   m30_slice["low"].round(4).tolist(),
        "close": m30_slice["close"].round(4).tolist(),
        "type": "candlestick",
        "increasing": {"line": {"color": "#4caf50"}, "fillcolor": "#2e7d3280"},
        "decreasing": {"line": {"color": "#ef5350"}, "fillcolor": "#c6282880"},
        "showlegend": False, "name": "30-min",
    }
    ema_pre = {
        "x": t30, "y": m30_slice[f"DAILY_EMA_{pre_p}"].round(4).tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _EMA21_COLOR, "width": 1.4, "shape": "hv"},
        "name": f"Daily EMA{pre_p}", "showlegend": False,
        "hovertemplate": f"Daily EMA{pre_p} %{{y:.2f}}<extra></extra>",
        "connectgaps": False,
    }
    ema_post = {
        "x": t30, "y": m30_slice[f"DAILY_EMA_{post_p}"].round(4).tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": _EMA8_COLOR, "width": 1.4, "shape": "hv"},
        "name": f"Daily EMA{post_p}", "showlegend": False,
        "hovertemplate": f"Daily EMA{post_p} %{{y:.2f}}<extra></extra>",
        "connectgaps": False,
    }
    bb_lower = {
        "x": t30,
        "y": m30_slice["BB_LOWER"].round(4).tolist()
              if "BB_LOWER" in m30_slice.columns else [],
        "type": "scatter", "mode": "lines",
        "line": {"color": "#66bb6a", "width": 1, "dash": "dash"},
        "name": "BB lower", "showlegend": False,
        "hovertemplate": "BB lower %{y:.2f}<extra></extra>",
    }
    bb_upper = {
        "x": t30,
        "y": m30_slice["BB_UPPER"].round(4).tolist()
              if "BB_UPPER" in m30_slice.columns else [],
        "type": "scatter", "mode": "lines",
        "line": {"color": "#ab47bc", "width": 1, "dash": "dash"},
        "name": "BB upper", "showlegend": False,
        "hovertemplate": "BB upper %{y:.2f}<extra></extra>",
    }

    entry_price = float(parent["avg_entry"])
    entry_x     = _snap_to_axis(entry_time, t30)
    entry_marker = {
        "x": [entry_x], "y": [entry_price],
        "type": "scatter", "mode": "markers+text",
        "marker": {"symbol": "triangle-up", "size": 13, "color": "#4fc3f7",
                   "line": {"color": "white", "width": 1}},
        "text": ["Entry"], "textposition": "top center",
        "textfont": {"color": "#4fc3f7", "size": 10},
        "showlegend": False,
        "hovertemplate": f"Entry @ {entry_price:.2f}<br>{entry_time}<extra></extra>",
    }

    last_price = float(parent["last_price"])
    last_x     = t30[-1] if t30 else entry_x
    last_marker = {
        "x": [last_x], "y": [last_price],
        "type": "scatter", "mode": "markers+text",
        "marker": {"symbol": "circle", "size": 11, "color": _LAST_COLOR,
                   "line": {"color": "white", "width": 1}},
        "text": [f"Last {last_price:.2f}"], "textposition": "top center",
        "textfont": {"color": _LAST_COLOR, "size": 10},
        "showlegend": False,
        "hovertemplate": (f"Last @ {last_price:.2f}"
                          f"<br>{as_of}<extra></extra>"),
    }

    # Closed sibling exits — red down-triangles, one per leg whose target
    # or stop already filled (mirrors the entry triangle but pointing down).
    exit_traces: list[dict] = []
    if all_legs is not None and not all_legs.empty:
        closed = all_legs[all_legs["exit_reason"] != "open_at_end"]
        for _, leg in closed.iterrows():
            ex_time = pd.Timestamp(leg.get("exit_time"))
            if pd.isna(ex_time):
                continue
            ex_price = float(leg.get("exit_price") or 0.0)
            if ex_price <= 0:
                continue
            ex_x = _snap_to_axis(ex_time, t30)
            leg_no = int(leg["leg"]) if pd.notna(leg.get("leg")) else 0
            shares = int(leg.get("shares") or 0)
            net = float(leg.get("net_pnl") or 0.0)
            reason = str(leg.get("exit_reason") or "exit")
            label = f"L{leg_no} exit" if leg_no else "Exit"
            exit_traces.append({
                "x": [ex_x], "y": [ex_price],
                "type": "scatter", "mode": "markers+text",
                "marker": {"symbol": "triangle-down", "size": 13,
                           "color": _EXIT_COLOR,
                           "line": {"color": "white", "width": 1}},
                "text": [label], "textposition": "bottom center",
                "textfont": {"color": _EXIT_COLOR, "size": 10},
                "showlegend": False,
                "hovertemplate": (f"{label} ({reason})<br>"
                                  f"@ {ex_price:.2f}<br>{ex_time}<br>"
                                  f"shares {shares}<br>"
                                  f"net {_fmt_money(net)}<extra></extra>"),
            })

    traces = ([bb_lower, bb_upper, ema_pre, ema_post,
               candle, entry_marker, last_marker] + exit_traces)

    # ---- shapes & annotations ------------------------------------------
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
            (fib["swing_high"], "#66bb6a",         f"swing_high {fib['swing_high']:.2f}"),
            (fib["swing_low"],  _SWING_LO_COLOR,   f"swing_low {fib['swing_low']:.2f}"),
            (fib["fib_target"], "#ab47bc",         f"fib 1.272 {fib['fib_target']:.2f}"),
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

    # Per-leg current stop level (only legs with a non-zero stop).
    seen_stops: set[float] = set()
    for _, leg in legs.iterrows():
        sp = float(leg.get("stop_price") or 0.0)
        if sp <= 0 or round(sp, 2) in seen_stops:
            continue
        seen_stops.add(round(sp, 2))
        shapes.append({
            "type": "line", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1, "y0": sp, "y1": sp,
            "line": {"color": _STOP_COLOR, "width": 1.2, "dash": "dash"},
        })
        annotations.append({
            "x": 0.0, "xref": "paper", "y": sp, "yref": "y",
            "text": f"L{int(leg['leg'])} stop {sp:.2f}",
            "showarrow": False, "xanchor": "left", "yanchor": "bottom",
            "font": {"color": _STOP_COLOR, "size": 9, "family": "monospace"},
        })

    # Per-leg target lines if no fib geometry available
    if fib is None:
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
            annotations.append({
                "x": 1.0, "xref": "paper", "y": tgt, "yref": "y",
                "text": f"L{int(leg['leg'])} target {tgt:.2f} ({kind})",
                "showarrow": False, "xanchor": "right", "yanchor": "bottom",
                "font": {"color": color, "size": 9, "family": "monospace"},
            })

    # ---- y-range padding -----------------------------------------------
    y_lo = float(min(m30_slice["low"].min(), entry_price, last_price))
    y_hi = float(max(m30_slice["high"].max(), entry_price, last_price))
    if fib is not None:
        y_lo = min(y_lo, fib["swing_low"])
        y_hi = max(y_hi, fib["swing_high"], fib["fib_target"])
    for et in exit_traces:
        ep = float(et["y"][0])
        y_lo = min(y_lo, ep)
        y_hi = max(y_hi, ep)
    for col in (f"DAILY_EMA_{pre_p}", f"DAILY_EMA_{post_p}",
                "BB_LOWER", "BB_UPPER"):
        if col not in m30_slice.columns:
            continue
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
        "margin": {"l": 55, "r": 110, "t": 10, "b": 40},
        "xaxis": {
            "gridcolor": "#2a2a4a", "linecolor": "#2a2a4a",
            "rangeslider": {
                "visible": True, "thickness": 0.06,
                "bgcolor": "#10101e", "bordercolor": "#2a2a4a",
                "borderwidth": 1,
            },
            "type": "category", "nticks": 14,
        },
        "yaxis": {
            "gridcolor": "#2a2a4a", "linecolor": "#2a2a4a",
            "tickprefix": "$", "range": [y_lo, y_hi], "side": "right",
        },
        "hovermode": "x", "shapes": shapes, "annotations": annotations,
        "height": 560,
    }

    return (f'<div id="{chart_id}"></div>'
            f'<script>Plotly.newPlot("{chart_id}", '
            f'{json.dumps(traces, default=_json_safe)}, '
            f'{json.dumps(layout)}, '
            '{responsive:true, displayModeBar:false});</script>')


def _build_legs_table(legs: pd.DataFrame) -> str:
    """Per-leg breakdown table for one open position."""
    rows: list[str] = []
    total_shares = 0
    total_cost   = 0.0
    total_mtm    = 0.0
    total_unreal = 0.0
    for _, leg in legs.sort_values("leg").iterrows():
        leg_no = int(leg["leg"])
        shares = int(leg.get("shares") or 0)
        ep     = float(leg.get("entry_price") or 0)
        cp     = float(leg.get("exit_price") or 0)
        cost   = float(leg.get("position_value") or 0)
        mtm    = cp * shares
        unreal = float(leg.get("net_pnl") or 0)
        pct    = float(leg.get("unreal_pct") or 0)
        sp     = float(leg.get("stop_price") or 0)
        tgt    = float(leg.get("target_price") or 0)
        kind   = str(leg.get("target_kind") or "")
        ema_p  = leg.get("ema_exit_period")
        try:
            ema_p_str = f"EMA{int(ema_p)}"
        except Exception:
            ema_p_str = "—"
        cls = "win" if unreal >= 0 else "loss"
        target_str = (f"{tgt:.2f} ({kind})" if tgt > 0
                      else (kind if kind else "—"))
        stop_str = f"{sp:.2f}" if sp > 0 else "—"
        rows.append(
            f'<tr>'
            f'<td class="left">L{leg_no}</td>'
            f'<td>{shares:,}</td>'
            f'<td>{ep:.2f}</td>'
            f'<td>{cp:.2f}</td>'
            f'<td>{stop_str}</td>'
            f'<td class="left">{target_str}</td>'
            f'<td class="left">{ema_p_str}</td>'
            f'<td>{_fmt_money(cost, 0)}</td>'
            f'<td>{_fmt_money(mtm, 0)}</td>'
            f'<td class="{cls}">{_fmt_money(unreal, 0)}</td>'
            f'<td class="{cls}">{_fmt_pct(pct)}</td>'
            f'</tr>'
        )
        total_shares += shares
        total_cost   += cost
        total_mtm    += mtm
        total_unreal += unreal
    total_pct = (total_unreal / total_cost * 100.0) if total_cost > 0 else 0.0
    total_cls = "win" if total_unreal >= 0 else "loss"
    foot = (
        f'<tr>'
        f'<td class="left">Totals</td>'
        f'<td>{total_shares:,}</td>'
        f'<td colspan="2"></td>'
        f'<td colspan="3"></td>'
        f'<td>{_fmt_money(total_cost, 0)}</td>'
        f'<td>{_fmt_money(total_mtm, 0)}</td>'
        f'<td class="{total_cls}">{_fmt_money(total_unreal, 0)}</td>'
        f'<td class="{total_cls}">{_fmt_pct(total_pct)}</td>'
        f'</tr>'
    )
    return f"""
<table class="tbl legs-table">
<thead><tr>
<th class="left">Leg</th><th>Shares</th>
<th>Entry $</th><th>Last $</th><th>Stop $</th>
<th class="left">Target</th><th class="left">EMA exit</th>
<th>Cost</th><th>MTM</th>
<th>Unrealised $</th><th>Unrealised %</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
<tfoot>{foot}</tfoot>
</table>
"""


def _build_position_card(parent: pd.Series, legs: pd.DataFrame,
                         all_legs: pd.DataFrame,
                         cache: _SymbolCache) -> str:
    sym = parent["symbol"]
    m30, daily = cache.get(sym)

    # Header
    et = pd.Timestamp(parent["entry_time"]).strftime("%Y-%m-%d %H:%M") \
        if pd.notna(parent["entry_time"]) else "—"
    cls = "win" if parent["unreal_net"] >= 0 else "loss"

    header = (
        f'<summary>'
        f'<span class="ticker">{sym}</span>'
        f'<span class="entry-date">entry {et} '
        f'&middot; {int(parent["days_held"])}d held</span>'
        f'<span class="summary-pnl {cls}">'
        f'{_fmt_pct(float(parent["unreal_pct"]))} &nbsp; '
        f'{_fmt_money(float(parent["unreal_net"]))}'
        f'</span>'
        f'</summary>'
    )

    # Setup-levels caption (one-line geometry)
    parts: list[str] = [f'Entry {float(parent["avg_entry"]):.2f}',
                        f'Last {float(parent["last_price"]):.2f}']
    fib = _fib_geometry(legs)
    if fib is not None:
        parts.extend([
            f'Swing low {fib["swing_low"]:.2f}',
            f'Swing high {fib["swing_high"]:.2f}',
            f'Fib 1.272 {fib["fib_target"]:.2f}',
        ])
    setup_caption = (
        f'<div class="setup-levels">'
        f'{" &nbsp;&middot;&nbsp; ".join(parts)}'
        f'</div>'
    )

    chart_html = _build_position_chart(parent, legs, all_legs, m30, daily) or \
        '<p class="muted">No price cache for this symbol — chart unavailable.</p>'
    legs_table = _build_legs_table(legs)

    return (f'<details class="pos-card" id="pos_{parent["parent_id"]}" open>'
            f'{header}{setup_caption}{chart_html}{legs_table}</details>')


# =====================================================================
# HTML ASSEMBLY
# =====================================================================
def _assemble_html(legs: pd.DataFrame, parents: pd.DataFrame,
                   all_parent_legs: pd.DataFrame) -> str:
    if legs.empty:
        body = ('<h1>BB 5-Min Cross — Current Portfolio</h1>'
                '<div class="subtitle">No open positions at end of backtest.</div>')
        return _wrap(body)

    kpis = _portfolio_kpis(legs, parents)
    cards = _build_kpi_grid(kpis)
    table = _build_portfolio_table(parents)

    cache = _SymbolCache()
    pos_cards: list[str] = []
    for _, p in parents.iterrows():
        pid = p["parent_id"]
        plegs     = legs[legs["parent_id"] == pid].copy()
        all_plegs = all_parent_legs[all_parent_legs["parent_id"] == pid].copy()
        pos_cards.append(_build_position_card(p, plegs, all_plegs, cache))

    sub = (f'As of {kpis["as_of"]} &nbsp;|&nbsp; '
           f'{kpis["n_trades"]} open trades, {kpis["n_legs"]} legs '
           f'&nbsp;|&nbsp; deployed {_fmt_money(kpis["cost"], 0)}')

    body = (
        '<h1>BB 5-Min Cross &mdash; Current Portfolio</h1>'
        f'<div class="subtitle">{sub}</div>'
        f'{cards}'
        '<section class="section"><h2>Portfolio</h2>'
        f'{table}</section>'
        '<section class="section"><h2>Positions</h2>'
        f'{"".join(pos_cards)}</section>'
    )
    return _wrap(body)


def _wrap(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BB 5-Min Cross — Current Portfolio</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


# =====================================================================
# ENTRY POINT
# =====================================================================
def generate(out_path: str = OUT_FILE) -> str:
    trades = _load_trades()
    legs = _filter_open_legs(trades)
    parents = _aggregate_open_parents(legs)
    parent_ids = set(parents["parent_id"]) if not parents.empty else set()
    all_parent_legs = _filter_parent_legs(trades, parent_ids)
    html = _assemble_html(legs, parents, all_parent_legs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def main() -> None:
    print("=" * 60)
    print("GENERATING CURRENT PORTFOLIO DASHBOARD")
    print("=" * 60)
    trades = _load_trades()
    legs = _filter_open_legs(trades)
    if legs.empty:
        print("No open_at_end positions in trades.csv.")
        out = generate()
        print(f"Wrote {out} (empty page)")
        return
    parents = _aggregate_open_parents(legs)
    parent_ids = set(parents["parent_id"]) if not parents.empty else set()
    n_closed_siblings = int(
        ((trades["parent_id"].isin(parent_ids))
         & (trades["exit_reason"] != "open_at_end")).sum()
    ) if not trades.empty else 0
    print(f"Open trades: {len(parents)}  |  open legs: {len(legs)}"
          f"  |  closed sibling legs (charted as exits): {n_closed_siblings}")
    out = generate()
    print(f"Wrote {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    sys.exit(0)
