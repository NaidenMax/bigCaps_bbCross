"""Show exactly what's in AGX_5min.parquet for 04/17 and 04/20, including
any pre-market / after-hours bars, plus the daily Polygon row for context.
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bt_config as cfg  # noqa

DAILY = os.path.join(cfg.DATA_DIR, "AGX_daily.parquet")
M5    = os.path.join(cfg.DATA_DIR, "AGX_5min.parquet")
M30   = os.path.join(cfg.DATA_DIR, "AGX_30min.parquet")

d  = pd.read_parquet(DAILY).copy()
m5 = pd.read_parquet(M5).copy()
m30 = pd.read_parquet(M30).copy()
for df in (d, m5, m30):
    df["date"] = pd.to_datetime(df["date"])

def _show_day(label, day):
    print(f"\n=== {label}  (Polygon raw, ALL hours) ===")
    drow = d.loc[d["date"].dt.date == day]
    print("DAILY row:")
    print(drow.to_string(index=False))

    m5d = m5.loc[m5["date"].dt.date == day].copy()
    m5d["t"] = m5d["date"].dt.time
    print(f"\n5-min bar count: {len(m5d)}")
    pre  = m5d[m5d["t"] <  pd.Timestamp("09:30").time()]
    rth  = m5d[(m5d["t"] >= pd.Timestamp("09:30").time()) &
               (m5d["t"] <= pd.Timestamp("15:55").time())]
    post = m5d[m5d["t"] >  pd.Timestamp("15:55").time()]
    print(f"  pre-market (<09:30):   {len(pre)} bars")
    print(f"  RTH (09:30..15:55):    {len(rth)} bars")
    print(f"  after-hours (>15:55):  {len(post)} bars")

    if not pre.empty:
        print("\n  pre-market bars:")
        print(pre[["date","open","high","low","close","volume"]].to_string(index=False))
    if not rth.empty:
        print("\n  RTH bars (first 3 + last 3):")
        rth_show = pd.concat([rth.head(3), rth.tail(3)]).drop_duplicates()
        print(rth_show[["date","open","high","low","close","volume"]].to_string(index=False))
    if not post.empty:
        print("\n  after-hours bars:")
        print(post[["date","open","high","low","close","volume"]].to_string(index=False))

    if not rth.empty and not drow.empty:
        rth_close = float(rth["close"].iloc[-1])
        poly_close = float(drow["close"].iloc[0])
        print(f"\n  RTH last 5-min close = {rth_close:.4f}")
        print(f"  Polygon daily close  = {poly_close:.4f}")
        print(f"  diff (Polygon - RTH) = {poly_close - rth_close:+.4f}")

_show_day("2026-04-17 (Friday)", pd.Timestamp("2026-04-17").date())
_show_day("2026-04-20 (Monday)", pd.Timestamp("2026-04-20").date())
