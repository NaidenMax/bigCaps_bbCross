"""HTML reports for the BB 5-Min Cross backtest.

Produces two self-contained HTML files in ``backtest/output/HTML_Reports/``:

* ``backtested_trading_analysis.html`` — top-level equity / risk / monthly summary
* ``analysis.html``                    — deep slice/dice (legs, MAE/MFE, by-symbol,
                                          ROC decile, MACD bucket, what-if sweep, ...)

Both files are styled to match the look of the meanrev report family.  Plotly is
loaded from a CDN so the files render cleanly when opened directly in a browser.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

import bt_config as cfg

# =====================================================================
# PATHS
# =====================================================================
HTML_DIR = os.path.join(cfg.OUTPUT_DIR, "HTML_Reports")

# =====================================================================
# SHARED STYLE
# =====================================================================
_SHARED_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0f0f1a; color:#e0e0e0; font-family:'Segoe UI',system-ui,sans-serif; padding:24px; }
h1 { font-size:1.6rem; margin-bottom:4px; color:#ffffff; }
h2 { font-size:1.1rem; font-weight:600; margin:24px 0 10px; color:#ccccdd;
     border-bottom:1px solid #2a2a4a; padding-bottom:6px; }
.subtitle { color:#8888aa; font-size:0.9rem; margin-bottom:20px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:12px; margin-bottom:24px; }
.card { background:#1a1a2e; border-radius:8px; padding:14px; text-align:center;
        border:1px solid #2a2a4a; }
.card-label { font-size:0.7rem; color:#8888aa; text-transform:uppercase;
              letter-spacing:0.5px; margin-bottom:6px; }
.card-value { font-size:1.15rem; font-weight:600; color:#e0e0e0; }
.card-sub   { font-size:0.7rem; color:#8888aa; margin-top:2px; }
.chart-container { background:#1a1a2e; border-radius:8px; padding:14px;
                   margin-bottom:18px; border:1px solid #2a2a4a; }
.chart-title { font-size:0.95rem; font-weight:600; margin-bottom:8px; color:#ccccdd; }
section { margin-top:24px; }
.section-title { font-size:1.0rem; font-weight:600; margin:18px 0 8px; color:#ccccdd; }
.caption { color:#8888aa; font-size:0.8rem; margin-bottom:10px; }
.tbl { width:100%; border-collapse:collapse; font-size:0.85rem;
       background:#1a1a2e; border:1px solid #2a2a4a; }
.tbl th { background:#16162b; padding:8px 10px; text-align:center;
          color:#aaaacc; font-weight:600; border-bottom:2px solid #2a2a4a; }
.tbl td { border-bottom:1px solid #1e1e35; padding:6px 10px; text-align:center; }
.tbl tr:hover td { background:#1e1e35; }
.tbl td.left, .tbl th.left { text-align:left; }
.pos { color:#4caf50; font-weight:600; }
.neg { color:#ef5350; font-weight:600; }
.warn { color:#ffa726; font-weight:600; }
.muted { color:#8888aa; }
.nav { margin-bottom:20px; font-size:0.85rem; }
.nav a { color:#4fc3f7; text-decoration:none; margin-right:14px; }
.nav a:hover { text-decoration:underline; }
.monthly { font-size:0.8rem; }
.monthly th, .monthly td { padding:5px 8px; }
"""

_PLOTLY_LAYOUT_BASE = {
    "paper_bgcolor": "#1a1a2e",
    "plot_bgcolor":  "#1a1a2e",
    "font":   {"color": "#aaaacc", "size": 11},
    "margin": {"l": 50, "r": 20, "t": 30, "b": 40},
    "xaxis":  {"gridcolor": "#2a2a4a", "linecolor": "#2a2a4a"},
    "yaxis":  {"gridcolor": "#2a2a4a", "linecolor": "#2a2a4a"},
    "showlegend": False,
}


def _html_shell(title: str, body: str) -> str:
    """Wrap body content in a complete dark-theme HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{_SHARED_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


# =====================================================================
# LOADERS
# =====================================================================
def _pick_newest(*names: str) -> str:
    """Return the most recently modified existing path, or '' if none exist."""
    paths = [os.path.join(cfg.OUTPUT_DIR, n) for n in names]
    existing = [p for p in paths if os.path.exists(p)]
    return max(existing, key=os.path.getmtime) if existing else ""


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_path = _pick_newest("trades.csv", "trades_new.csv")
    daily_path = _pick_newest("strategy_daily_pnl.csv", "strategy_daily_pnl_new.csv")

    trades = pd.read_csv(trades_path) if trades_path else pd.DataFrame()
    daily = pd.read_csv(daily_path) if daily_path else pd.DataFrame()

    if not trades.empty:
        if "pct_change" not in trades.columns and "net_pnl" in trades.columns:
            trades["pct_change"] = (trades["net_pnl"]
                                    / trades["position_value"].replace(0, np.nan) * 100)
        # Derive day_of_week / entry_hhmm if missing (older CSVs)
        if "entry_time" in trades.columns:
            ts = pd.to_datetime(trades["entry_time"], errors="coerce", utc=True)
            try:
                ts = ts.dt.tz_convert("US/Eastern")
            except Exception:
                pass
            if "day_of_week" not in trades.columns:
                trades["day_of_week"] = ts.dt.day_name()
            if "entry_hhmm" not in trades.columns:
                trades["entry_hhmm"] = ts.dt.strftime("%H:%M")

    return trades, daily


# =====================================================================
# FORMATTING
# =====================================================================
def _fmt_money(v: float, dp: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"${v:,.{dp}f}"


def _fmt_pct(v: float, dp: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{dp}f}%"


def _fmt_num(v: float, dp: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:,.{dp}f}"


def _signed_class(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "muted"
    return "pos" if v > 0 else ("neg" if v < 0 else "muted")


def _card(label: str, value: str, sub: str = "", value_class: str = "") -> str:
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    cls = f' class="card-value {value_class}"' if value_class else ' class="card-value"'
    return f'<div class="card"><div class="card-label">{label}</div><div{cls}>{value}</div>{sub_html}</div>'


def _format_held_minutes(mins: float) -> str:
    """Humanize a duration in minutes to '{d}d {h}h' or '{h}h {m}m' or '{m}m'."""
    if mins is None or (isinstance(mins, float) and (math.isnan(mins) or math.isinf(mins))):
        return "—"
    total = int(round(mins))
    if total <= 0:
        return "0m"
    days, rem = divmod(total, 24 * 60)
    hours, m = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {m}m"
    return f"{m}m"


# =====================================================================
# HOLD TIME / EXPOSURE
# =====================================================================
def _hold_time_stats(trades: pd.DataFrame) -> dict:
    """Per-trade (parent_id) lifespan stats in minutes.

    Lifespan of a trade = max(exit_time of any leg) - min(entry_time of any leg),
    so a trim-and-trail trade with a long-trailing leg counts once at its full
    duration, not once per leg.
    """
    empty = {"avg_mins": float("nan"), "median_mins": float("nan"),
             "max_mins": float("nan")}
    if trades.empty:
        return empty
    if "parent_id" in trades.columns and "entry_time" in trades.columns \
            and "exit_time" in trades.columns:
        df = trades[["parent_id", "entry_time", "exit_time"]].copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["entry_time", "exit_time"])
        if df.empty:
            return empty
        g = df.groupby("parent_id")
        spans = (g["exit_time"].max() - g["entry_time"].min()).dt.total_seconds() / 60.0
        spans = spans[spans >= 0]
        if spans.empty:
            return empty
        return {"avg_mins": float(spans.mean()),
                "median_mins": float(spans.median()),
                "max_mins": float(spans.max())}
    if "period_held_minutes" in trades.columns:
        s = pd.to_numeric(trades["period_held_minutes"], errors="coerce").dropna()
        if s.empty:
            return empty
        return {"avg_mins": float(s.mean()), "median_mins": float(s.median()),
                "max_mins": float(s.max())}
    return empty


def _daily_overnight_exposure(trades: pd.DataFrame,
                              daily: pd.DataFrame) -> pd.DataFrame:
    """Daily overnight exposure = sum(position_value of legs open at 16:00) /
    equity * 100, evaluated for each row in ``daily``.

    A leg is "open at the close" on day D if entry_time <= 16:00 of D and
    (exit_time is missing OR exit_time > 16:00 of D).
    """
    cols = ["date", "exposure_pct"]
    if daily.empty or trades.empty:
        return pd.DataFrame(columns=cols)
    if not {"entry_time", "exit_time", "position_value"}.issubset(trades.columns):
        return pd.DataFrame(columns=cols)
    if "equity" not in daily.columns or "date" not in daily.columns:
        return pd.DataFrame(columns=cols)

    t = trades[["entry_time", "exit_time", "position_value"]].copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], errors="coerce", utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], errors="coerce", utc=True)
    t["position_value"] = pd.to_numeric(t["position_value"], errors="coerce").fillna(0.0)
    t = t.dropna(subset=["entry_time"])
    entries_naive = t["entry_time"].dt.tz_convert(None) \
        if t["entry_time"].dt.tz is not None else t["entry_time"]
    exits_naive = t["exit_time"].dt.tz_convert(None) \
        if t["exit_time"].dt.tz is not None else t["exit_time"]

    out_dates: list[str] = []
    out_pct: list[float] = []
    for _, r in daily.iterrows():
        try:
            eod = pd.to_datetime(r["date"]) + pd.Timedelta(hours=16)
        except Exception:
            continue
        equity = float(r.get("equity", float("nan")))
        if not math.isfinite(equity) or equity <= 0:
            continue
        open_mask = (entries_naive <= eod) & (exits_naive.isna() | (exits_naive > eod))
        invested = float(t.loc[open_mask, "position_value"].sum())
        out_dates.append(str(r["date"])[:10])
        out_pct.append(round(invested / equity * 100.0, 2))
    return pd.DataFrame({"date": out_dates, "exposure_pct": out_pct})


# =====================================================================
# CHART / TABLE BUILDERS
# =====================================================================
_PLOT_COUNTER = [0]


def _next_id() -> str:
    _PLOT_COUNTER[0] += 1
    return f"plt{_PLOT_COUNTER[0]}"


def _plotly(div_id: str, data: list[dict], layout: dict | None = None,
            height: int = 320) -> str:
    """Return chart-container HTML with a Plotly script."""
    full_layout = {**_PLOTLY_LAYOUT_BASE, **(layout or {})}
    return (f'<div id="{div_id}" style="height:{height}px;"></div>'
            f'<script>Plotly.newPlot('
            f'"{div_id}", {json.dumps(data)}, {json.dumps(full_layout)}, '
            '{responsive:true, displayModeBar:false});</script>')


def _line_chart(x: Iterable, y: Iterable, title: str, *,
                fill: str | None = None, color: str = "#4fc3f7",
                y_suffix: str = "", height: int = 280,
                y_min: float | None = None,
                y_max: float | None = None) -> str:
    """Render a responsive single-line Plotly chart.

    ``y_min`` / ``y_max`` set an explicit y-axis range so the curve isn't
    squashed by ``fill='tozeroy'`` (which otherwise forces 0 into view).
    When ``y_min`` is supplied the chart switches to a manual baseline-trace
    fill so the shaded area starts at ``y_min`` instead of zero.
    """
    div_id = _next_id()
    line_trace = {
        "x": list(x), "y": list(y), "type": "scatter", "mode": "lines",
        "line": {"color": color, "width": 2},
    }
    traces: list[dict] = []
    if fill and y_min is not None:
        # Invisible baseline trace + tonexty fill so the shading anchors at
        # y_min, not at zero.  Drawn first so the line paints on top.
        traces.append({
            "x": list(x), "y": [y_min] * len(list(x)),
            "type": "scatter", "mode": "lines",
            "line": {"width": 0, "color": "rgba(0,0,0,0)"},
            "showlegend": False, "hoverinfo": "skip",
        })
        line_trace["fill"] = "tonexty"
        line_trace["fillcolor"] = "rgba(79,195,247,0.10)"
    elif fill:
        line_trace["fill"] = fill
        line_trace["fillcolor"] = "rgba(79,195,247,0.10)"
    traces.append(line_trace)

    yaxis = {**_PLOTLY_LAYOUT_BASE["yaxis"], "ticksuffix": y_suffix}
    if y_min is not None or y_max is not None:
        ys = [v for v in y if v is not None and not (isinstance(v, float) and math.isnan(v))]
        lo = y_min if y_min is not None else (min(ys) if ys else 0)
        hi = y_max if y_max is not None else (max(ys) if ys else 1)
        yaxis["range"] = [lo, hi]
        yaxis["autorange"] = False

    layout = {"yaxis": yaxis}
    body = (f'<div class="chart-container"><div class="chart-title">{title}</div>'
            f'{_plotly(div_id, traces, layout, height)}</div>')
    return body


def _bar_chart(x: Iterable, y: Iterable, title: str, *,
               y_suffix: str = "", height: int = 280,
               colors: list[str] | None = None) -> str:
    div_id = _next_id()
    if colors is None:
        colors = ["#4caf50" if v >= 0 else "#ef5350" for v in y]
    trace = {"x": list(x), "y": list(y), "type": "bar",
             "marker": {"color": colors}}
    layout = {"yaxis": {**_PLOTLY_LAYOUT_BASE["yaxis"], "ticksuffix": y_suffix}}
    return (f'<div class="chart-container"><div class="chart-title">{title}</div>'
            f'{_plotly(div_id, [trace], layout, height)}</div>')


def _hist_chart(values: Iterable, title: str, *, bins: int = 40,
                color: str = "#4fc3f7", x_suffix: str = "%",
                height: int = 260) -> str:
    div_id = _next_id()
    vals = [v for v in values if v is not None and not math.isnan(v)]
    trace = {"x": vals, "type": "histogram", "nbinsx": bins,
             "marker": {"color": color}}
    layout = {"xaxis": {**_PLOTLY_LAYOUT_BASE["xaxis"], "ticksuffix": x_suffix}}
    return (f'<div class="chart-container"><div class="chart-title">{title}</div>'
            f'{_plotly(div_id, [trace], layout, height)}</div>')


def _scatter_chart(x: Iterable, y: Iterable, title: str, *,
                   x_label: str = "", y_label: str = "",
                   color: str = "#4fc3f7", height: int = 320) -> str:
    div_id = _next_id()
    trace = {"x": list(x), "y": list(y), "mode": "markers", "type": "scatter",
             "marker": {"color": color, "size": 5, "opacity": 0.55}}
    layout = {
        "xaxis": {**_PLOTLY_LAYOUT_BASE["xaxis"], "title": x_label, "ticksuffix": "%"},
        "yaxis": {**_PLOTLY_LAYOUT_BASE["yaxis"], "title": y_label, "ticksuffix": "%"},
    }
    return (f'<div class="chart-container"><div class="chart-title">{title}</div>'
            f'{_plotly(div_id, [trace], layout, height)}</div>')


def _table_html(headers: list[str], rows: list[list[str]],
                first_col_left: bool = True) -> str:
    th_cells = "".join(
        f'<th class="left">{h}</th>' if first_col_left and i == 0 else f'<th>{h}</th>'
        for i, h in enumerate(headers))
    body_rows = []
    for row in rows:
        cells = "".join(
            f'<td class="left">{c}</td>' if first_col_left and i == 0 else f'<td>{c}</td>'
            for i, c in enumerate(row))
        body_rows.append(f'<tr>{cells}</tr>')
    return ('<table class="tbl"><thead><tr>'
            f'{th_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>')


# =====================================================================
# SLICE / DICE HELPERS
# =====================================================================
def _slice_stats(df: pd.DataFrame, group_col: str,
                 order: list | None = None) -> pd.DataFrame:
    """Group trades by a column and compute count/mean_pct/win_rate/MAE/MFE."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_col, dropna=False)
    out = pd.DataFrame({
        "count": g.size(),
        "mean_pct": g["pct_change"].mean(),
        "median_pct": g["pct_change"].median(),
        "win_rate": g["pct_change"].apply(lambda s: (s > 0).mean() * 100),
        "net_pnl": g["net_pnl"].sum() if "net_pnl" in df.columns else g.size() * np.nan,
    })
    if "mae_pct" in df.columns:
        out["mean_mae"] = g["mae_pct"].mean()
    if "mfe_pct" in df.columns:
        out["mean_mfe"] = g["mfe_pct"].mean()
    out = out.reset_index()
    if order is not None:
        out["__ord"] = out[group_col].map({v: i for i, v in enumerate(order)})
        out = out.sort_values("__ord", na_position="last").drop(columns="__ord")
    return out


def _slice_table(df: pd.DataFrame, group_col: str, label: str,
                 order: list | None = None) -> str:
    stats = _slice_stats(df, group_col, order=order)
    if stats.empty:
        return f'<p class="muted">No data for {label}.</p>'
    headers = [label, "Count", "Mean %", "Median %", "Win %", "Net P&L"]
    if "mean_mae" in stats.columns:
        headers.append("Mean MAE %")
    if "mean_mfe" in stats.columns:
        headers.append("Mean MFE %")
    rows = []
    for _, r in stats.iterrows():
        row = [
            str(r[group_col]),
            f'{int(r["count"]):,}',
            _fmt_pct(r["mean_pct"]),
            _fmt_pct(r["median_pct"]),
            _fmt_pct(r["win_rate"]),
            _fmt_money(r["net_pnl"]),
        ]
        if "mean_mae" in stats.columns:
            row.append(_fmt_pct(r["mean_mae"]))
        if "mean_mfe" in stats.columns:
            row.append(_fmt_pct(r["mean_mfe"]))
        rows.append(row)
    return _table_html(headers, rows)


def _decile_bins(series: pd.Series, n: int = 10) -> pd.Series:
    """Return a category Series with decile labels D1..Dn (D1=lowest).

    NaN inputs (e.g. trades booked before a daily indicator was warmed up)
    stay as real NaN in the output -- previously they leaked through as the
    literal string "nan" because of `astype(str)`, which then crashed any
    downstream `sorted(... key=lambda s: int(s.lstrip("D")))`.
    """
    s = series.dropna()
    if s.empty or s.nunique() < 2:
        return pd.Series([f"D1"] * len(series), index=series.index)
    try:
        labels = [f"D{i+1}" for i in range(n)]
        binned = pd.qcut(series, q=n, labels=labels, duplicates="drop")
    except ValueError:
        # Too few unique values for n bins; fall back to fewer bins.
        nbins = max(2, series.nunique())
        labels = [f"D{i+1}" for i in range(nbins)]
        binned = pd.qcut(series, q=nbins, labels=labels, duplicates="drop")
    # Convert to object so NaN survives as NaN (not the string "nan").
    out = binned.astype(object)
    return out.where(series.notna(), other=np.nan)


def _aggregate_to_parents(trades: pd.DataFrame) -> pd.DataFrame:
    """Collapse leg-level trades to one row per parent_id for setup-level
    analysis (entry context buckets, etc.).

    Multi-leg trades share the same entry, so analysing them per-leg would
    triple-count and dilute medians.  Setup-quality questions like "where in
    the swing range did we enter?" should always be answered at parent level.
    """
    if trades.empty or "parent_id" not in trades.columns:
        return pd.DataFrame()
    g = trades.groupby("parent_id", sort=False)
    # "first" is fine for entry-time fields because the leg list is built from
    # a single base_pos at entry, so all legs of one parent carry identical
    # values for these columns.
    first_cols = {
        "symbol":              ("symbol", "first"),
        "entry_time":          ("entry_time", "first"),
        "entry_price":         ("entry_price", "first"),
        "entry_pct_in_range":  ("entry_pct_in_range", "first"),
        "pct_from_6mo_high":   ("pct_from_6mo_high", "first"),
        "high_6mo":            ("high_6mo", "first"),
        "swing_low_at_entry":  ("swing_low_at_entry", "first"),
        "swing_high_at_entry": ("swing_high_at_entry", "first"),
        # Volatility regime at entry (audit-only; bracketed below)
        "bb_width_pct_30m":     ("bb_width_pct_30m", "first"),
        "bb_width_pct_30m_avg": ("bb_width_pct_30m_avg", "first"),
        "bb_squeeze_ratio_30m": ("bb_squeeze_ratio_30m", "first"),
        "daily_atr_pct":        ("daily_atr_pct", "first"),
        "bb_vs_atr_ratio":      ("bb_vs_atr_ratio", "first"),
    }
    available = {k: v for k, v in first_cols.items() if v[0] in trades.columns}
    parents = g.agg(
        n_legs=("leg", "count") if "leg" in trades.columns else ("net_pnl", "size"),
        parent_net_pnl=("net_pnl", "sum"),
        parent_cost=("position_value", "sum") if "position_value" in trades.columns
                    else ("net_pnl", "size"),
        **available,
    ).reset_index()
    if "position_value" in trades.columns:
        parents["parent_pct"] = np.where(
            parents["parent_cost"] > 0,
            parents["parent_net_pnl"] / parents["parent_cost"] * 100.0,
            0.0,
        )
    else:
        parents["parent_pct"] = 0.0
    return parents


def _bucket_with_bins(s: pd.Series, edges: list[float],
                      labels: list[str]) -> pd.Series:
    """pd.cut wrapper that returns a string Series including a category for
    NaN inputs, so they don't silently disappear from the bracket table.
    """
    out = pd.cut(s, bins=edges, labels=labels, include_lowest=True,
                 right=False).astype(object)
    out = out.where(s.notna(), other="(no data)")
    return out


def _slice_table_parent(parents: pd.DataFrame, group_col: str, label: str,
                        order: list | None = None) -> str:
    """Parent-level analogue of _slice_table.  Aggregates parent_pct and
    parent_net_pnl per bucket so multi-leg setups are counted once."""
    if parents.empty or group_col not in parents.columns:
        return f'<p class="muted">No data for {label}.</p>'
    g = parents.groupby(group_col, dropna=False)
    stats = pd.DataFrame({
        "count":      g.size(),
        "mean_pct":   g["parent_pct"].mean(),
        "median_pct": g["parent_pct"].median(),
        "win_rate":   g["parent_pct"].apply(lambda s: (s > 0).mean() * 100),
        "net_pnl":    g["parent_net_pnl"].sum(),
    }).reset_index()
    if order is not None:
        stats["__ord"] = stats[group_col].map({v: i for i, v in enumerate(order)})
        stats = stats.sort_values("__ord", na_position="last").drop(columns="__ord")
    headers = [label, "Setups", "Mean %", "Median %", "Win %", "Net P&L"]
    rows = []
    for _, r in stats.iterrows():
        rows.append([
            str(r[group_col]),
            f'{int(r["count"]):,}',
            _fmt_pct(r["mean_pct"]),
            _fmt_pct(r["median_pct"]),
            _fmt_pct(r["win_rate"]),
            _fmt_money(r["net_pnl"]),
        ])
    return _table_html(headers, rows)


def _whatif_sweep(trades: pd.DataFrame, multipliers: list[float]) -> pd.DataFrame:
    """Simulate "exit at target = entry * m if MFE crossed, else actual exit".

    For each multiplier m, target_pct = (m - 1) * 100.  A trade "would have hit"
    that target if its mfe_pct >= target_pct.
    """
    if trades.empty or "mfe_pct" not in trades.columns or "pct_change" not in trades.columns:
        return pd.DataFrame()
    actual = trades["pct_change"].astype(float)
    mfe = trades["mfe_pct"].astype(float)
    baseline = float(actual.mean())
    rows = []
    for m in multipliers:
        target_pct = (m - 1.0) * 100.0
        hit = mfe >= target_pct
        sim = np.where(hit, target_pct, actual)
        rows.append({
            "multiplier": m,
            "target_pct": target_pct,
            "hit_rate_pct": float(hit.mean() * 100),
            "mean_return_sim": float(np.mean(sim)),
            "mean_return_baseline": baseline,
            "improvement": float(np.mean(sim) - baseline),
        })
    return pd.DataFrame(rows)


# =====================================================================
# EQUITY / RISK CORE METRICS
# =====================================================================
def _core_metrics(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {}
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    total_pnl = float(trades["net_pnl"].sum())
    win_rate = len(wins) / n * 100
    avg_pct = float(trades["pct_change"].mean()) if "pct_change" in trades.columns else float("nan")
    avg_win_pct = float(wins["pct_change"].mean()) if len(wins) and "pct_change" in trades.columns else 0.0
    avg_loss_pct = float(losses["pct_change"].mean()) if len(losses) and "pct_change" in trades.columns else 0.0
    pf = (abs(wins["net_pnl"].sum() / losses["net_pnl"].sum())
          if losses["net_pnl"].sum() != 0 else float("inf"))

    # Equity / DD / CAGR / Sharpe
    final_eq = float(daily["equity"].iloc[-1]) if not daily.empty else cfg.INITIAL_CAPITAL
    init = cfg.INITIAL_CAPITAL
    ret_pct = (final_eq - init) / init * 100 if init else 0
    cagr = 0.0
    sharpe = 0.0
    max_dd = 0.0
    if not daily.empty:
        years = max(1, len(daily)) / 252
        cagr = (((final_eq / init) ** (1 / years) - 1) * 100) if years > 0 and init > 0 else 0
        if "drawdown_pct" in daily.columns:
            max_dd = float(daily["drawdown_pct"].min())
        if len(daily) > 1 and "daily_pnl" in daily.columns:
            rets = daily["daily_pnl"] / daily["equity"].shift(1).fillna(init)
            std = rets.std()
            if std and std > 0:
                sharpe = float(rets.mean() / std * math.sqrt(252))

    # Best / worst by net_pnl
    best_idx = trades["net_pnl"].idxmax()
    worst_idx = trades["net_pnl"].idxmin()
    best_row = trades.loc[best_idx]
    worst_row = trades.loc[worst_idx]

    # Expectancy (per-trade $)
    expectancy = float(trades["net_pnl"].mean())
    median_pnl = float(trades["net_pnl"].median())

    return {
        "n": n, "win_rate": win_rate, "total_pnl": total_pnl,
        "avg_pct": avg_pct, "avg_win_pct": avg_win_pct, "avg_loss_pct": avg_loss_pct,
        "pf": pf, "final_eq": final_eq, "ret_pct": ret_pct,
        "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
        "best_pnl": float(best_row["net_pnl"]), "best_sym": best_row.get("symbol", ""),
        "worst_pnl": float(worst_row["net_pnl"]), "worst_sym": worst_row.get("symbol", ""),
        "expectancy": expectancy, "median_pnl": median_pnl,
    }


def _monthly_returns_table(daily: pd.DataFrame) -> str:
    """Render a year x month grid of monthly returns as percentages.

    Denominator depends on cfg.USE_COMPOUNDING:
      * False -> cfg.INITIAL_CAPITAL for every cell (constant base, all years
        comparable; matches the engine's actual sizing behavior).
      * True  -> equity at the LAST bar of the prior month (so each cell reads
        "% growth this month"); for the first month with no prior, falls back
        to cfg.INITIAL_CAPITAL.

    Year totals: simple sum when non-compounding (consistent with constant
    base), geometric chain (prod(1+r)-1) when compounding.
    """
    if daily.empty or "daily_pnl" not in daily.columns:
        return '<p class="muted">No daily P&L data.</p>'
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["mon"] = df["date"].dt.month

    pnl_pivot = df.pivot_table(index="year", columns="mon", values="daily_pnl",
                               aggfunc="sum", fill_value=0)

    use_comp = bool(getattr(cfg, "USE_COMPOUNDING", False))
    init_cap = float(cfg.INITIAL_CAPITAL)

    # Start-of-month equity = equity on the last bar BEFORE the month starts.
    # Built only when compounding; otherwise unused.
    base_pivot = None
    if use_comp and "equity" in df.columns:
        df_sorted = df.sort_values("date")
        eq_eom = (df_sorted.groupby(["year", "mon"])["equity"]
                           .last().unstack("mon"))
        # Shift "end of (year, mon)" -> base for the FOLLOWING month.
        # Flatten to chronological list, shift by 1, then restore the grid.
        ordered = eq_eom.stack().sort_index()        # MultiIndex (year, mon)
        prior_eq = ordered.shift(1).reindex(ordered.index)
        base_pivot = prior_eq.unstack("mon")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    headers = ["Year"] + months + ["Total"]

    rows = []
    for yr in pnl_pivot.index.tolist():
        row = [str(yr)]
        monthly_rets: list[float] = []
        for m in range(1, 13):
            if m not in pnl_pivot.columns:
                row.append("—")
                continue
            pnl = float(pnl_pivot.at[yr, m])
            if pnl == 0.0:
                row.append("—")
                continue
            if use_comp and base_pivot is not None:
                base = base_pivot.at[yr, m] if m in base_pivot.columns else float("nan")
                if not (isinstance(base, float) and math.isnan(base)) and base > 0:
                    pct = pnl / float(base) * 100.0
                else:
                    pct = pnl / init_cap * 100.0
            else:
                pct = pnl / init_cap * 100.0
            monthly_rets.append(pct)
            cls = _signed_class(pct)
            row.append(f'<span class="{cls}">{_fmt_pct(pct, 2)}</span>')

        if not monthly_rets:
            row.append("—")
        elif use_comp:
            chained = 1.0
            for r in monthly_rets:
                chained *= (1.0 + r / 100.0)
            ytotal = (chained - 1.0) * 100.0
            row.append(f'<span class="{_signed_class(ytotal)}">{_fmt_pct(ytotal, 2)}</span>')
        else:
            ytotal = sum(monthly_rets)
            row.append(f'<span class="{_signed_class(ytotal)}">{_fmt_pct(ytotal, 2)}</span>')
        rows.append(row)

    return ('<table class="tbl monthly"><thead><tr>'
            + "".join(f'<th>{h}</th>' for h in headers)
            + '</tr></thead><tbody>'
            + "".join('<tr>' + "".join(f'<td>{c}</td>' for c in r) + '</tr>'
                      for r in rows)
            + '</tbody></table>')


# =====================================================================
# REPORT 1: backtested_trading_analysis.html (top-level summary)
# =====================================================================
def _build_summary_report(trades: pd.DataFrame, daily: pd.DataFrame) -> str:
    m = _core_metrics(trades, daily)
    hold = _hold_time_stats(trades)
    expo_df = _daily_overnight_exposure(trades, daily)
    if not expo_df.empty:
        avg_expo = float(expo_df["exposure_pct"].mean())
        peak_expo = float(expo_df["exposure_pct"].max())
        peak_date = str(expo_df.loc[expo_df["exposure_pct"].idxmax(), "date"])
    else:
        avg_expo = float("nan")
        peak_expo = float("nan")
        peak_date = ""
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cards_html = '<div class="cards">' + "".join([
        _card("Total Return", _fmt_pct(m["ret_pct"]),
              value_class=_signed_class(m["ret_pct"])),
        _card("CAGR", _fmt_pct(m["cagr"]), value_class=_signed_class(m["cagr"])),
        _card("Sharpe", _fmt_num(m["sharpe"], 2)),
        _card("Max Drawdown", _fmt_pct(m["max_dd"]),
              value_class="neg" if m["max_dd"] < 0 else ""),
        _card("Win Rate", _fmt_pct(m["win_rate"], 1)),
        _card("Profit Factor", _fmt_num(m["pf"], 2)),
        _card("Avg %/Trade", _fmt_pct(m["avg_pct"]),
              value_class=_signed_class(m["avg_pct"])),
        _card("Avg Win %", _fmt_pct(m["avg_win_pct"]), value_class="pos"),
        _card("Avg Loss %", _fmt_pct(m["avg_loss_pct"]), value_class="neg"),
        _card("Total Trades", f'{m["n"]:,}'),
        _card("Net Profit", _fmt_money(m["total_pnl"]),
              value_class=_signed_class(m["total_pnl"])),
        _card("Avg P&L / Trade", _fmt_money(m["expectancy"]),
              value_class=_signed_class(m["expectancy"])),
        _card("Median P&L", _fmt_money(m["median_pnl"]),
              value_class=_signed_class(m["median_pnl"])),
        _card("Best Trade", _fmt_money(m["best_pnl"]),
              sub=str(m["best_sym"]), value_class="pos"),
        _card("Worst Trade", _fmt_money(m["worst_pnl"]),
              sub=str(m["worst_sym"]), value_class="neg"),
        _card("Expectancy", _fmt_money(m["expectancy"]),
              value_class=_signed_class(m["expectancy"])),
        _card("Avg Hold (Trade)", _format_held_minutes(hold["avg_mins"])),
        _card("Median Hold (Trade)", _format_held_minutes(hold["median_mins"])),
        _card("Avg Overnight Exposure", _fmt_pct(avg_expo, 1)),
        _card("Peak Overnight Exposure", _fmt_pct(peak_expo, 1),
              sub=peak_date,
              value_class="neg" if (math.isfinite(peak_expo) and peak_expo > 400)
                          else ""),
    ]) + "</div>"

    charts_html = ""
    if not daily.empty:
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        eq_vals = pd.to_numeric(d["equity"], errors="coerce").dropna()
        if not eq_vals.empty:
            eq_lo = float(eq_vals.min())
            eq_hi = float(eq_vals.max())
            span = max(eq_hi - eq_lo, 1.0)
            pad = span * 0.05
            # Anchor floor at the lower of (initial capital, min equity) so
            # the first point isn't clipped, with a small breathing pad.
            floor_ref = min(eq_lo, float(cfg.INITIAL_CAPITAL))
            y_min = max(0.0, floor_ref - pad)
            y_max = eq_hi + pad
        else:
            y_min, y_max = 0.0, float(cfg.INITIAL_CAPITAL)
        charts_html += _line_chart(d["date"], d["equity"],
                                   "Equity Curve", fill="tozeroy",
                                   y_min=y_min, y_max=y_max)
        if "drawdown_pct" in d.columns:
            charts_html += _line_chart(d["date"], d["drawdown_pct"],
                                       "Drawdown (%)", fill="tozeroy",
                                       color="#ef5350", y_suffix="%")
        if "daily_pnl" in d.columns:
            prior_eq = d["equity"].shift(1).fillna(cfg.INITIAL_CAPITAL)
            d["daily_ret"] = (d["daily_pnl"] / prior_eq * 100).round(3)
            charts_html += _bar_chart(d["date"], d["daily_ret"],
                                      "Daily Return % (vs prior-day equity)",
                                      y_suffix="%")
        if not expo_df.empty:
            charts_html += _line_chart(expo_df["date"], expo_df["exposure_pct"],
                                       "Overnight Capital Exposure (% of Equity)",
                                       y_suffix="%", color="#ffa726")

    monthly_html = ('<section><div class="section-title">Monthly Return (%)</div>'
                    + _monthly_returns_table(daily) + '</section>')

    body = (
        f'<h1>BB 5-Min Cross — Backtest Summary</h1>'
        f'<div class="subtitle">Generated {gen} &nbsp;|&nbsp; '
        f'{cfg.START_DATE} → {cfg.END_DATE} &nbsp;|&nbsp; '
        f'<a href="analysis.html" style="color:#4fc3f7;">deep analysis →</a></div>'
        f'<div class="nav"><a href="analysis.html">Open deep slice/dice analysis</a></div>'
        + cards_html
        + '<section><h2>Equity, Drawdown &amp; Daily Returns</h2>'
        + charts_html + '</section>'
        + monthly_html
    )

    out_path = os.path.join(HTML_DIR, "backtested_trading_analysis.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_html_shell("BB 5-Min Cross — Backtest Summary", body))
    return out_path


# =====================================================================
# REPORT 2: analysis.html (deep slice / dice)
# =====================================================================
_TIME_OF_DAY_ORDER = [f"{h:02d}:{mm:02d}" for h in range(9, 16)
                      for mm in (0, 30) if not (h == 9 and mm == 0)]
_DAY_OF_WEEK_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _bucket_time_of_day(hhmm: str) -> str:
    """Map 'HH:MM' to a time-of-day bucket label.

    Most of the day uses 30-min buckets (10:00, 10:30, 11:00, ...).  The
    opening half-hour is split into two 15-min sub-buckets — 09:30 and
    09:45 — for finer resolution in `analysis.html`, since open-bar
    behavior typically differs sharply from the rest of the morning.
    """
    try:
        h, mm = hhmm.split(":")
        h, mm = int(h), int(mm)
    except Exception:
        return ""
    if h == 9 and 30 <= mm < 45:
        return "09:30"
    if h == 9 and 45 <= mm < 60:
        return "09:45"
    bucket_min = 0 if mm < 30 else 30
    return f"{h:02d}:{bucket_min:02d}"


def _week_of_month(date_str: str) -> str:
    try:
        d = pd.to_datetime(date_str)
        return f"Week {((d.day - 1) // 7) + 1}"
    except Exception:
        return ""


# ---- Refined exit-reason labels for analysis.html --------------------
# The engine writes a single `"ema_exit"` for every EMA-trail exit, but
# the *active* EMA period (8 vs 21) is preserved on each leg row in the
# `ema_exit_period` column, because `_close_position` does `{**position}`
# at exit time and the value is mutated to EMA_EXIT_PERIOD_POST_LEG1
# when L1 fills.  We split that lump here so the user can see EMA8
# trail wins vs EMA21 stop losses separately.
_EXIT_REASON_LABELS = {
    "stop":              "Hard Stop (session-low - offset)",
    "target_swing_high": "L1 Target (swing high)",
    "target_fib_1272":   "L2 Target (fib 1.272)",
    "target":            "Target",
    "open_at_end":       "Open at backtest end",
}


def _refine_exit_reason(row) -> str:
    raw = str(row.get("exit_reason", ""))
    if raw == "ema_exit":
        try:
            p = int(float(row.get("ema_exit_period", 0)))
        except Exception:
            p = 0
        if p == cfg.EMA_EXIT_PERIOD_POST_LEG1:
            return f"EMA{p} Trail (close < EMA{p})"
        if p == cfg.EMA_EXIT_PERIOD_PRE_LEG1:
            return f"EMA{p} Stop (close < EMA{p})"
        if p > 0:
            return f"EMA{p} Exit (close < EMA{p})"
        return "EMA Exit (period n/a)"
    return _EXIT_REASON_LABELS.get(raw, raw)


def _build_analysis_report(trades: pd.DataFrame, daily: pd.DataFrame) -> str:
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = trades.copy()
    df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
    if "mae_pct" in df.columns:
        df["mae_pct"] = pd.to_numeric(df["mae_pct"], errors="coerce")
    if "mfe_pct" in df.columns:
        df["mfe_pct"] = pd.to_numeric(df["mfe_pct"], errors="coerce")

    sections: list[str] = []

    # --- 1. PnL by Leg -------------------------------------------------
    if "leg" in df.columns and df["leg"].notna().any():
        leg_tbl = _slice_table(df, "leg", "Leg", order=[1, 2, 3])
        sections.append(
            '<section><div class="section-title">PnL by Leg (trim-and-trail)</div>'
            f'{leg_tbl}</section>'
        )
    else:
        sections.append(
            '<section><div class="section-title">PnL by Leg (trim-and-trail)</div>'
            '<p class="muted">No <code>leg</code> column in trades.csv — '
            'the trim-and-trail multi-leg exit is not enabled.</p></section>'
        )

    # --- 1b. Entry context (parent-level) -----------------------------
    # Two bracket tables that group SETUPS (not legs) by where the entry
    # sat geometrically.  Parent-level avoids triple-counting trim-and-trail
    # legs that share an entry price.
    parents = _aggregate_to_parents(df)

    if (not parents.empty and "entry_pct_in_range" in parents.columns
            and parents["entry_pct_in_range"].notna().any()):
        # Edges chosen to match how the user described the question:
        # "0-25%, etc... vs near highs".  -inf/+inf catch entries outside
        # the swing range (rare, but possible if entry below swing_low or
        # above swing_high).
        range_edges  = [-np.inf, 0.0, 25.0, 50.0, 75.0, 100.0, np.inf]
        range_labels = [
            "< 0% (below swing low)",
            "0-25% (lower quarter)",
            "25-50%",
            "50-75%",
            "75-100% (upper quarter)",
            "> 100% (above swing high)",
        ]
        parents["_range_bkt"] = _bucket_with_bins(
            parents["entry_pct_in_range"], range_edges, range_labels)
        order = range_labels + ["(no data)"]
        tbl = _slice_table_parent(parents, "_range_bkt",
                                  "Entry within swing range", order=order)
        sections.append(
            '<section><div class="section-title">'
            'Entry Position Within Swing Range (parent-level)</div>'
            '<p class="caption">0% = entry at swing low, 100% = entry at '
            'swing high. Lower buckets = pulled-back entries; higher '
            'buckets = chasing into highs.</p>'
            f'{tbl}</section>'
        )
    else:
        sections.append(
            '<section><div class="section-title">'
            'Entry Position Within Swing Range (parent-level)</div>'
            '<p class="muted">No <code>entry_pct_in_range</code> in '
            'trades.csv yet &mdash; re-run the backtest to populate it.'
            '</p></section>'
        )

    if (not parents.empty and "pct_from_6mo_high" in parents.columns
            and parents["pct_from_6mo_high"].notna().any()):
        sixmo_edges  = [-np.inf, -30.0, -20.0, -15.0, -10.0, -5.0, 0.0, np.inf]
        sixmo_labels = [
            "<= -30%",
            "-30 to -20%",
            "-20 to -15%",
            "-15 to -10%",
            "-10 to -5%",
            "-5 to 0% (near highs)",
            "> 0% (new high entry)",
        ]
        parents["_sixmo_bkt"] = _bucket_with_bins(
            parents["pct_from_6mo_high"], sixmo_edges, sixmo_labels)
        order = sixmo_labels + ["(no data)"]
        tbl = _slice_table_parent(parents, "_sixmo_bkt",
                                  "Entry vs 6-month high", order=order)
        sections.append(
            '<section><div class="section-title">'
            'Entry vs 6-Month High (parent-level)</div>'
            '<p class="caption">Distance of entry price from the trailing '
            '126 trading-day daily high (excluding the entry day). '
            'Negative = pulled back from highs; near 0% = entering at/near '
            'the 6-month high.</p>'
            f'{tbl}</section>'
        )
    else:
        sections.append(
            '<section><div class="section-title">'
            'Entry vs 6-Month High (parent-level)</div>'
            '<p class="muted">No <code>pct_from_6mo_high</code> in '
            'trades.csv yet &mdash; re-run the backtest to populate it.'
            '</p></section>'
        )

    # --- 1c. PnL by 30m BB Squeeze Ratio at Entry ---------------------
    # Volatility regime test #1: current 30m BB width % divided by its
    # trailing N-day mean.  < 1 => contracted (squeeze) at entry,
    # > 1 => expanded.  Lookback set via BB_WIDTH_AVG_LOOKBACK_DAYS in
    # bt_config.py.
    if (not parents.empty
            and "bb_squeeze_ratio_30m" in parents.columns
            and parents["bb_squeeze_ratio_30m"].notna().any()):
        sq_edges  = [-np.inf, 0.7, 0.9, 1.1, 1.5, np.inf]
        sq_labels = [
            "Strong Squeeze (<0.7)",
            "Squeeze (0.7-0.9)",
            "Neutral (0.9-1.1)",
            "Expansion (1.1-1.5)",
            "Strong Expansion (>1.5)",
        ]
        parents["_squeeze_bkt"] = _bucket_with_bins(
            parents["bb_squeeze_ratio_30m"], sq_edges, sq_labels)
        order = sq_labels + ["(no data)"]
        tbl = _slice_table_parent(parents, "_squeeze_bkt",
                                  "30m BB squeeze ratio", order=order)
        lookback_days = getattr(cfg, "BB_WIDTH_AVG_LOOKBACK_DAYS", 10)
        sections.append(
            '<section><div class="section-title">'
            'PnL by 30m BB Squeeze Ratio at Entry (parent-level)</div>'
            '<p class="caption">'
            f'Current 30-min Bollinger Band width % divided by its trailing '
            f'{lookback_days}-day mean (configured via '
            '<code>BB_WIDTH_AVG_LOOKBACK_DAYS</code>). '
            '&lt; 1 = bands tighter than usual at entry (compression / squeeze). '
            '&gt; 1 = bands wider than usual at entry (already in expansion). '
            'Tests whether trades fired during a squeeze run differently than '
            'trades fired during expansion.</p>'
            f'{tbl}</section>'
        )
    else:
        sections.append(
            '<section><div class="section-title">'
            'PnL by 30m BB Squeeze Ratio at Entry (parent-level)</div>'
            '<p class="muted">No <code>bb_squeeze_ratio_30m</code> in '
            'trades.csv yet &mdash; re-run the backtest to populate it.'
            '</p></section>'
        )

    # --- 1d. PnL by 30m BB Width vs Daily ATR at Entry ----------------
    # Volatility regime test #2: 30m BB width % divided by daily ATR %.
    # < 1 => intraday vol < daily vol (BB tight relative to the day's
    # typical range); > 1 => intraday vol > daily vol.
    if (not parents.empty
            and "bb_vs_atr_ratio" in parents.columns
            and parents["bb_vs_atr_ratio"].notna().any()):
        ba_edges  = [-np.inf, 0.5, 1.0, 1.5, 2.5, np.inf]
        ba_labels = [
            "BB << ATR (<0.5)",
            "BB < ATR (0.5-1.0)",
            "BB ~ ATR (1.0-1.5)",
            "BB > ATR (1.5-2.5)",
            "BB >> ATR (>2.5)",
        ]
        parents["_bbatr_bkt"] = _bucket_with_bins(
            parents["bb_vs_atr_ratio"], ba_edges, ba_labels)
        order = ba_labels + ["(no data)"]
        tbl = _slice_table_parent(parents, "_bbatr_bkt",
                                  "30m BB width % / daily ATR %", order=order)
        sections.append(
            '<section><div class="section-title">'
            'PnL by 30m BB Width vs Daily ATR at Entry (parent-level)</div>'
            '<p class="caption">'
            'Ratio of current 30-min Bollinger Band width % to daily ATR %. '
            '&lt; 1 = the 30-min bands are narrower than the typical daily range '
            '(intraday compressed against a wider daily envelope). '
            '&gt; 1 = the 30-min bands are wider than the daily ATR '
            '(intraday already expanded relative to daily volatility).</p>'
            f'{tbl}</section>'
        )
    else:
        sections.append(
            '<section><div class="section-title">'
            'PnL by 30m BB Width vs Daily ATR at Entry (parent-level)</div>'
            '<p class="muted">No <code>bb_vs_atr_ratio</code> in '
            'trades.csv yet &mdash; re-run the backtest to populate it.'
            '</p></section>'
        )

    # --- 2. PnL by Exit Reason ----------------------------------------
    # Refine the lumped "ema_exit" row by splitting on the per-leg
    # `ema_exit_period` so users see EMA8 trail (post-L1-fill / L3
    # default) separately from EMA21 stop (pre-L1-fill default).
    if "exit_reason" in df.columns:
        df["_exit_reason_refined"] = df.apply(_refine_exit_reason, axis=1)
        ema8_label  = (f"EMA{cfg.EMA_EXIT_PERIOD_POST_LEG1} Trail "
                       f"(close < EMA{cfg.EMA_EXIT_PERIOD_POST_LEG1})")
        ema21_label = (f"EMA{cfg.EMA_EXIT_PERIOD_PRE_LEG1} Stop "
                       f"(close < EMA{cfg.EMA_EXIT_PERIOD_PRE_LEG1})")
        desired_order = [
            "L1 Target (swing high)",
            "L2 Target (fib 1.272)",
            "Target",
            ema8_label,
            ema21_label,
            "Hard Stop (session-low - offset)",
            "Open at backtest end",
        ]
        present_vals = set(df["_exit_reason_refined"].dropna().unique().tolist())
        present = [c for c in desired_order if c in present_vals]
        extras  = sorted(present_vals - set(desired_order))
        order = present + extras
        sections.append(
            '<section><div class="section-title">PnL by Exit Reason</div>'
            '<p class="caption">The engine writes a single <code>ema_exit</code> '
            'reason for every EMA-trail exit; this table splits it into '
            f'<b>{ema8_label}</b> (active on L3 by default and on surviving '
            'siblings after L1 fills) and '
            f'<b>{ema21_label}</b> (active on L1, L2 and L3 before any L1 '
            'fills) using the per-leg <code>ema_exit_period</code> column.</p>'
            + _slice_table(df, "_exit_reason_refined", "Exit Reason", order=order)
            + '</section>'
        )

    # --- 3. PnL by Time of Day ----------------------------------------
    if "entry_hhmm" in df.columns and df["entry_hhmm"].notna().any():
        df["_tod"] = df["entry_hhmm"].astype(str).map(_bucket_time_of_day)
        order = sorted(df["_tod"].dropna().unique().tolist())
        tod_tbl = _slice_table(df, "_tod", "Entry Slot", order=order)
        stats = _slice_stats(df, "_tod", order=order)
        chart = _bar_chart(stats["_tod"], stats["mean_pct"],
                           "Mean % return by entry time-of-day", y_suffix="%")
        sections.append(
            '<section><div class="section-title">PnL by Time of Day</div>'
            '<p class="caption">Buckets are 30 minutes wide except for the '
            'opening half-hour, which is split into 09:30-09:45 and '
            '09:45-10:00 for finer resolution.</p>'
            f'{chart}{tod_tbl}</section>'
        )

    # --- 3b. PnL by Recent 30m EMA Cross (tag) ------------------------
    # Tag attached at entry time: True iff a bullish TREND_FAST_EMA >
    # TREND_SLOW_EMA cross occurred within the last
    # cfg.HAD_30M_BULLISH_CROSS_LOOKBACK_BARS 30-min bars.  Read from CSV
    # as strings ("True"/"False"), so coerce explicitly.
    if "had_30m_bullish_cross" in df.columns:
        _ser = df["had_30m_bullish_cross"]
        if _ser.dtype == bool:
            _bool = _ser
        else:
            _bool = _ser.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False})
        df["_cross_tag"] = _bool.map(
            {True: "Yes (cross within lookback)",
             False: "No (stale trend)"})
        if df["_cross_tag"].notna().any():
            order = ["Yes (cross within lookback)", "No (stale trend)"]
            n_bars = getattr(cfg, "HAD_30M_BULLISH_CROSS_LOOKBACK_BARS", 130)
            tbl = _slice_table(df, "_cross_tag",
                               f"30m EMA cross in last {n_bars} bars",
                               order=order)
            stats = _slice_stats(df, "_cross_tag", order=order)
            chart = _bar_chart(stats["_cross_tag"], stats["mean_pct"],
                               "Mean % return by recent-30m-cross tag",
                               y_suffix="%")
            sections.append(
                '<section><div class="section-title">'
                'PnL by Recent 30m EMA Cross</div>'
                '<p class="caption">"Yes" = a bullish 30-min EMA cross '
                f'(fast over slow) happened within the last {n_bars} '
                '30-min bars at the moment of entry. Toggle '
                '<code>ENABLE_RECENT_30M_CROSS_FILTER</code> in '
                '<code>bt_config.py</code> to make this a hard entry filter.</p>'
                + chart + tbl + '</section>'
            )

    # --- 4. PnL by Day of Week ----------------------------------------
    if "day_of_week" in df.columns and df["day_of_week"].notna().any():
        dow_tbl = _slice_table(df, "day_of_week", "Day", order=_DAY_OF_WEEK_ORDER)
        stats = _slice_stats(df, "day_of_week", order=_DAY_OF_WEEK_ORDER)
        chart = _bar_chart(stats["day_of_week"], stats["mean_pct"],
                           "Mean % return by day of week", y_suffix="%")
        sections.append(
            '<section><div class="section-title">PnL by Day of Week</div>'
            f'{chart}{dow_tbl}</section>'
        )

    # --- 5. PnL by Week of Month --------------------------------------
    if "date" in df.columns and df["date"].notna().any():
        df["_wom"] = df["date"].astype(str).map(_week_of_month)
        order = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]
        wom_tbl = _slice_table(df, "_wom", "Week of Month", order=order)
        stats = _slice_stats(df, "_wom", order=order)
        chart = _bar_chart(stats["_wom"], stats["mean_pct"],
                           "Mean % return by week of month", y_suffix="%")
        sections.append(
            '<section><div class="section-title">PnL by Week of Month</div>'
            f'{chart}{wom_tbl}</section>'
        )

    # --- 6. PnL by ROC Decile -----------------------------------------
    if "daily_roc" in df.columns and df["daily_roc"].notna().any():
        df["_roc_dec"] = _decile_bins(df["daily_roc"])
        order = sorted(df["_roc_dec"].dropna().unique().tolist(),
                       key=lambda s: int(str(s).lstrip("D")))
        tbl = _slice_table(df, "_roc_dec",
                           f"ROC({cfg.ROC_PERIOD}) Decile (D1=lowest)",
                           order=order)
        stats = _slice_stats(df, "_roc_dec", order=order)
        chart = _bar_chart(stats["_roc_dec"], stats["mean_pct"],
                           f"Mean % return by ROC({cfg.ROC_PERIOD}) decile",
                           y_suffix="%")
        sections.append(
            '<section><div class="section-title">'
            f'PnL by Daily ROC({cfg.ROC_PERIOD}) Decile</div>'
            '<p class="caption">Higher decile = stronger relative momentum at entry.'
            '</p>' + chart + tbl + '</section>'
        )

    # --- 7. PnL by MACD Histogram Quartile ----------------------------
    if "daily_macd_hist" in df.columns and df["daily_macd_hist"].notna().any():
        df["_macd_q"] = _decile_bins(df["daily_macd_hist"], n=4).map(
            lambda s: s.replace("D", "Q") if isinstance(s, str) else s)
        order = sorted(df["_macd_q"].dropna().unique().tolist(),
                       key=lambda s: int(str(s).lstrip("Q")))
        tbl = _slice_table(df, "_macd_q", "MACD Hist Quartile (Q1=lowest)",
                           order=order)
        stats = _slice_stats(df, "_macd_q", order=order)
        chart = _bar_chart(stats["_macd_q"], stats["mean_pct"],
                           "Mean % return by daily MACD histogram quartile",
                           y_suffix="%")
        sections.append(
            '<section><div class="section-title">PnL by MACD Histogram Quartile</div>'
            + chart + tbl + '</section>'
        )

    # --- 8. Per-Symbol Summary ----------------------------------------
    if "symbol" in df.columns:
        per_sym = df.groupby("symbol").agg(
            count=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            mean_pct=("pct_change", "mean"),
            win_rate=("pct_change", lambda s: (s > 0).mean() * 100),
            best=("net_pnl", "max"),
            worst=("net_pnl", "min"),
        ).reset_index().sort_values("net_pnl", ascending=False)
        cap = ""
        if len(per_sym) > 50:
            cap = (f'<p class="caption">Showing top 50 of {len(per_sym)} symbols '
                   'by net P&L (positive top, negative bottom).</p>')
            per_sym = pd.concat([per_sym.head(25), per_sym.tail(25)])
        rows = []
        for _, r in per_sym.iterrows():
            cls_pnl = _signed_class(r["net_pnl"])
            rows.append([
                str(r["symbol"]),
                f'{int(r["count"]):,}',
                f'<span class="{cls_pnl}">{_fmt_money(r["net_pnl"])}</span>',
                _fmt_pct(r["mean_pct"]),
                _fmt_pct(r["win_rate"], 1),
                _fmt_money(r["best"]),
                _fmt_money(r["worst"]),
            ])
        tbl = _table_html(["Symbol", "Trades", "Net P&L", "Mean %",
                           "Win %", "Best $", "Worst $"], rows)
        sections.append(
            '<section><div class="section-title">Per-Symbol Summary</div>'
            f'{cap}{tbl}</section>'
        )

    # --- 9. MAE Distribution ------------------------------------------
    if "mae_pct" in df.columns and df["mae_pct"].notna().any():
        vals = df["mae_pct"].dropna()
        chart = _hist_chart(vals,
                            "MAE % distribution (max adverse excursion from entry)")
        stats = _table_html(
            ["Stat", "MAE (%)"],
            [["Mean", _fmt_pct(vals.mean())],
             ["Median", _fmt_pct(vals.median())],
             ["25th pct", _fmt_pct(vals.quantile(0.25))],
             ["75th pct", _fmt_pct(vals.quantile(0.75))],
             ["Min", _fmt_pct(vals.min())]])
        sections.append(
            '<section><div class="section-title">MAE Distribution</div>'
            f'{chart}{stats}</section>'
        )

    # --- 10. MFE Distribution -----------------------------------------
    if "mfe_pct" in df.columns and df["mfe_pct"].notna().any():
        vals = df["mfe_pct"].dropna()
        chart = _hist_chart(vals, "MFE % distribution (max favourable excursion from entry)",
                            color="#4caf50")
        stats = _table_html(
            ["Stat", "MFE (%)"],
            [["Mean", _fmt_pct(vals.mean())],
             ["Median", _fmt_pct(vals.median())],
             ["25th pct", _fmt_pct(vals.quantile(0.25))],
             ["75th pct", _fmt_pct(vals.quantile(0.75))],
             ["Max", _fmt_pct(vals.max())]])
        sections.append(
            '<section><div class="section-title">MFE Distribution</div>'
            f'{chart}{stats}</section>'
        )

    # --- 11. MFE vs Actual Return (Capture Efficiency) ----------------
    if {"mfe_pct", "pct_change"}.issubset(df.columns) and df["mfe_pct"].notna().any():
        sub = df.dropna(subset=["mfe_pct", "pct_change"])
        wins = sub[sub["pct_change"] > 0]
        eff = (wins["pct_change"] / wins["mfe_pct"].replace(0, np.nan)).dropna()
        scatter = _scatter_chart(sub["mfe_pct"], sub["pct_change"],
                                 "Realised return vs MFE",
                                 x_label="MFE %", y_label="Realised %")
        eff_tbl = _table_html(
            ["Stat", "Value"],
            [["Trades w/ MFE > 0", f'{len(sub[sub["mfe_pct"] > 0]):,}'],
             ["Mean efficiency (winners only)", _fmt_pct(eff.mean() * 100)
              if not eff.empty else "—"],
             ["Median efficiency (winners only)", _fmt_pct(eff.median() * 100)
              if not eff.empty else "—"],
             ["Mean MFE %", _fmt_pct(sub["mfe_pct"].mean())],
             ["Mean Realised %", _fmt_pct(sub["pct_change"].mean())]])
        sections.append(
            '<section><div class="section-title">MFE vs Realised Return '
            '(Capture Efficiency)</div>'
            '<p class="caption">Efficiency = realised% / MFE% for winning trades. '
            '100% = caught the entire move. Lower = leaving money on the table.</p>'
            f'{scatter}{eff_tbl}</section>'
        )

    # --- 12. What-If Target Sweep -------------------------------------
    sweep = _whatif_sweep(df, list(cfg.WHATIF_TARGET_MULTIPLIERS))
    if not sweep.empty:
        rows = []
        for _, r in sweep.iterrows():
            rows.append([
                _fmt_num(r["multiplier"], 3),
                _fmt_pct(r["target_pct"]),
                _fmt_pct(r["hit_rate_pct"], 1),
                _fmt_pct(r["mean_return_sim"]),
                _fmt_pct(r["mean_return_baseline"]),
                f'<span class="{_signed_class(r["improvement"])}">'
                f'{_fmt_pct(r["improvement"])}</span>',
            ])
        tbl = _table_html(
            ["Multiplier", "Target %", "Hit Rate %", "Sim Mean %",
             "Baseline Mean %", "Improvement"],
            rows)
        chart = _bar_chart(sweep["multiplier"].astype(str),
                           sweep["mean_return_sim"],
                           "Simulated mean return by target multiplier",
                           y_suffix="%")
        sections.append(
            '<section><div class="section-title">What-If Target Sweep</div>'
            '<p class="caption">For each multiplier <em>m</em>, target_pct = '
            '(<em>m</em> &minus; 1) &times; 100%. A trade is assumed to fill at the '
            'target if its MFE &ge; target_pct, otherwise the actual realised return '
            'is used. This is a useful first-cut sanity check; real partial-exit '
            'logic with bar sequencing may differ slightly.</p>'
            f'{chart}{tbl}</section>'
        )

    body = (
        '<h1>BB 5-Min Cross — Deep Trade Analysis</h1>'
        f'<div class="subtitle">Generated {gen} &nbsp;|&nbsp; '
        f'{len(df):,} trades &nbsp;|&nbsp; '
        f'<a href="backtested_trading_analysis.html" style="color:#4fc3f7;">'
        '← back to summary</a></div>'
        '<div class="nav">'
        '<a href="backtested_trading_analysis.html">← Summary report</a>'
        '</div>'
        + "".join(sections)
    )

    out_path = os.path.join(HTML_DIR, "analysis.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_html_shell("BB 5-Min Cross — Deep Trade Analysis", body))
    return out_path


# =====================================================================
# ENTRY POINT
# =====================================================================
def generate_report():
    """Backwards-compatible alias for the older API."""
    main()


def main():
    trades, daily = _load_inputs()
    if trades.empty:
        print("No trades.csv found. Run backtest first.")
        return
    os.makedirs(HTML_DIR, exist_ok=True)
    summary = _build_summary_report(trades, daily)
    analysis = _build_analysis_report(trades, daily)
    print(f"Reports written:\n  {summary}\n  {analysis}")


if __name__ == "__main__":
    main()
