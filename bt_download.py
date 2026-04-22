"""Download daily + 30-min + 5-min bars from Polygon.io and save as Parquet."""

from __future__ import annotations
import os, time, requests
from datetime import date, time as dtime, timedelta
import pandas as pd
from tqdm import tqdm
import bt_config as cfg

_BASE = "https://api.polygon.io/v2/aggs/ticker"

TIMEFRAMES = {
    "daily":  ("1", "day"),
    "30min":  ("30", "minute"),
    "5min":   ("5", "minute"),
}

_BAR_DURATION = {
    "30min": pd.Timedelta(minutes=30),
    "5min":  pd.Timedelta(minutes=5),
}
_DAILY_SESSION_CLOSE = dtime(16, 0)  # US equities regular session close (ET)


def _trim_partial_trailing(df: pd.DataFrame, tf_key: str) -> pd.DataFrame:
    """Drop trailing rows whose bar window is not yet finished.

    Mid-session runs would otherwise persist incomplete bars to the cache,
    and because forward_from is day-quantized those bars would never be
    re-fetched within the same day.  Dropping them lets the next forward
    fetch re-pull a current snapshot.
    """
    if df.empty:
        return df
    s = df["date"]
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("US/Eastern")
    now_et = pd.Timestamp.now(tz="US/Eastern")
    if tf_key == "daily":
        bar_end = s.dt.normalize() + pd.Timedelta(hours=16)
    else:
        bar_end = s + _BAR_DURATION.get(tf_key, pd.Timedelta(0))
    keep = bar_end <= now_et
    return df[keep].reset_index(drop=True)


def _fetch_bars(symbol: str, multiplier: str, timespan: str,
                from_date: date, to_date: date) -> tuple[pd.DataFrame, str]:
    """Fetch bars from Polygon.

    Returns ``(df, reason)`` where ``reason`` is one of:
      - ``"ok"``         : at least one bar returned
      - ``"forbidden"``  : 403 (free-tier restriction or unauthorized symbol)
      - ``"throttled"``  : exhausted retries on 429 rate-limit responses
      - ``"network"``    : exhausted retries on transient network errors
      - ``"empty"``      : 200 OK but zero bars in the requested window
    """
    url = f"{_BASE}/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000,
              "apiKey": cfg.POLYGON_API_KEY}
    all_results: list = []
    last_429 = False
    for attempt in range(cfg.MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                last_429 = True
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 403:
                return pd.DataFrame(), "forbidden"
            resp.raise_for_status()
            last_429 = False
            data = resp.json()
            all_results.extend(data.get("results", []))
            next_url = data.get("next_url")
            while next_url:
                time.sleep(cfg.RATE_LIMIT_SLEEP)
                r2 = requests.get(next_url, params={"apiKey": cfg.POLYGON_API_KEY}, timeout=30)
                r2.raise_for_status()
                d2 = r2.json()
                all_results.extend(d2.get("results", []))
                next_url = d2.get("next_url")
            break
        except Exception:
            if attempt == cfg.MAX_RETRIES - 1:
                return pd.DataFrame(), "network"
            time.sleep(2 ** attempt)
    else:
        # Loop exhausted without break: every attempt was a 429.
        return pd.DataFrame(), ("throttled" if last_429 else "network")

    if not all_results:
        return pd.DataFrame(), "empty"

    df = pd.DataFrame(all_results)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                            "l": "low", "c": "close", "v": "volume"})
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, "ok"


def _parquet_path(symbol: str, tf_key: str) -> str:
    return os.path.join(cfg.DATA_DIR, f"{symbol}_{tf_key}.parquet")


def _load_existing(symbol: str, tf_key: str) -> pd.DataFrame:
    path = _parquet_path(symbol, tf_key)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def download_symbol(symbol: str, daily_only: bool = False) -> list[dict]:
    """Download timeframes for one symbol (incremental).

    If daily_only is True, only the daily timeframe is fetched (used for
    benchmark / index symbols that aren't traded directly).

    Returns a list of failure dicts ``{symbol, timeframe, reason}`` for each
    timeframe that ended with no parquet written AND no pre-existing cache.
    Timeframes whose forward fetch failed but already have a usable cache
    are not reported (the cache is still good for backtesting).
    """
    os.makedirs(cfg.DATA_DIR, exist_ok=True)

    warmup_start_daily = cfg.START_DATE - timedelta(days=int(cfg.WARMUP_DAILY_BARS * 1.6))
    warmup_start_intra = cfg.START_DATE - timedelta(days=int(cfg.WARMUP_INTRADAY_DAYS * 1.6))

    tf_dates = {
        "daily": warmup_start_daily,
        "30min": warmup_start_intra,
        "5min":  cfg.START_DATE - timedelta(days=5),
    }

    timeframes = {"daily": TIMEFRAMES["daily"]} if daily_only else TIMEFRAMES

    today_et = pd.Timestamp.now(tz="US/Eastern").date()

    failures: list[dict] = []

    for tf_key, (mult, span) in timeframes.items():
        raw_existing = _load_existing(symbol, tf_key)
        # Drop any trailing bars whose window hasn't closed yet, so a later
        # forward fetch can re-pull a current snapshot in their place.
        existing = _trim_partial_trailing(raw_existing, tf_key)
        trimmed_n = len(raw_existing) - len(existing)
        if trimmed_n > 0:
            tqdm.write(f"  trim: {symbol} {tf_key}: dropped {trimmed_n} partial trailing bar(s)")
        requested_from = tf_dates[tf_key]

        pieces: list[pd.DataFrame] = []
        last_reason = "ok"

        # --- backfill OLDER bars if the cache doesn't reach requested_from ---
        if not existing.empty:
            first = pd.Timestamp(existing["date"].min())
            if first.tzinfo is None:
                first = first.tz_localize("US/Eastern")
            if first.date() > requested_from:
                back_to = first.date() - timedelta(days=1)
                df_back, reason = _fetch_bars(symbol, mult, span,
                                              requested_from, back_to)
                last_reason = reason
                time.sleep(cfg.RATE_LIMIT_SLEEP)
                if not df_back.empty:
                    pieces.append(df_back)

        # --- forward incremental (extend toward END_DATE) ---
        # When the cache already touches today, always re-pull from today so
        # any partial bar written during an earlier mid-session run gets
        # overwritten with a current snapshot.  Otherwise stay day-quantized
        # so nightly runs cost zero API calls when there is nothing new.
        if existing.empty:
            forward_from = requested_from
        else:
            last = pd.Timestamp(existing["date"].max())
            if last.tzinfo is None:
                last = last.tz_localize("US/Eastern")
            last_date = last.date()
            if last_date >= today_et:
                forward_from = max(requested_from, today_et)
            else:
                forward_from = max(requested_from, last_date + timedelta(days=1))

        if forward_from <= cfg.END_DATE:
            df_fwd, reason = _fetch_bars(symbol, mult, span,
                                         forward_from, cfg.END_DATE)
            last_reason = reason
            time.sleep(cfg.RATE_LIMIT_SLEEP)
            if not df_fwd.empty:
                pieces.append(df_fwd)

        if not existing.empty:
            pieces.append(existing)

        if not pieces:
            # Nothing on disk and nothing fetched -- record why.  If we never
            # made an HTTP call (e.g. forward_from > END_DATE and no backfill
            # was needed), default to "empty".
            failures.append({"symbol": symbol, "timeframe": tf_key,
                             "reason": last_reason if last_reason != "ok"
                                       else "empty"})
            continue

        # Concat order is [backfill, forward, existing] so when a date appears
        # in both a fresh forward fetch and the cache, drop_duplicates(keep
        # first) keeps the freshly-fetched copy -- this is what overwrites
        # any partial bar that survived the trim's freshness gate.
        df = (pd.concat(pieces)
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True))
        df.to_parquet(_parquet_path(symbol, tf_key), index=False)

    return failures


def load_watchlist() -> list[str]:
    symbols = []
    with open(cfg.WATCHLIST, encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s and not s.startswith("#"):
                symbols.append(s)
    return symbols


def load_universe() -> list[str]:
    """Symbols listed in cfg.ROC_COMPLEX_UNIVERSE_FILE.

    Used as the ranking universe for ROC_RANK_MODE == "TC2000_Complex_Rank".
    Returns [] when the file is absent (so Simple_Rank users never see
    an error).
    """
    path = getattr(cfg, "ROC_COMPLEX_UNIVERSE_FILE", "")
    if not path or not os.path.exists(path):
        return []
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main():
    symbols = load_watchlist()
    print(f"Downloading {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Date range: {cfg.START_DATE} to {cfg.END_DATE}")

    failures: list[dict] = []
    for sym in tqdm(symbols, desc="Downloading"):
        sym_failures = download_symbol(sym)
        if sym_failures:
            failures.extend(sym_failures)
            # Show the first failure inline so the user sees the reason in
            # context without sliding the progress bar around.
            reasons = ",".join(sorted({f["reason"] for f in sym_failures}))
            tqdm.write(f"  miss: {sym} -> {reasons} "
                       f"({len(sym_failures)} timeframe(s))")

    # Benchmark / index symbol used by ENABLE_INDEX_MACD_FILTER (daily only).
    if getattr(cfg, "ENABLE_INDEX_MACD_FILTER", False):
        idx = cfg.INDEX_SYMBOL.upper()
        if idx not in {s.upper() for s in symbols}:
            print(f"Downloading benchmark index {idx} (daily only)...")
            idx_failures = download_symbol(idx, daily_only=True)
            failures.extend(idx_failures)

    # Russell 1000 (or whichever) ranking universe -- daily only.
    # Used by ROC_RANK_MODE == "TC2000_Complex_Rank" to compute per-day
    # percentile cutoffs across the full universe.
    if (getattr(cfg, "ENABLE_ROC_RANK_FILTER", False)
            and getattr(cfg, "ROC_RANK_MODE", "Simple_Rank")
                == "TC2000_Complex_Rank"):
        trading_set = {s.upper() for s in symbols}
        universe = [u for u in load_universe() if u not in trading_set]
        if universe:
            print(f"Downloading {len(universe)} ROC-rank universe symbols "
                  f"(daily only)...")
            for usym in tqdm(universe, desc="Universe download"):
                u_failures = download_symbol(usym, daily_only=True)
                if u_failures:
                    failures.extend(u_failures)
                    reasons = ",".join(sorted({f["reason"] for f in u_failures}))
                    tqdm.write(f"  miss: {usym} -> {reasons} "
                               f"({len(u_failures)} timeframe(s))")

    # ---- summary ---------------------------------------------------------
    if failures:
        from collections import Counter
        n_syms = len({f["symbol"] for f in failures})
        n_tf   = len(failures)
        by_reason = Counter(f["reason"] for f in failures)
        breakdown = ", ".join(f"{n} {r}" for r, n in by_reason.most_common())
        print(f"Missing data: {n_syms} symbols ({n_tf} timeframe failures): {breakdown}")

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        out_csv = os.path.join(cfg.OUTPUT_DIR, "missing_symbols.csv")
        (pd.DataFrame(failures)
           .sort_values(["reason", "symbol", "timeframe"])
           .to_csv(out_csv, index=False))
        print(f"Wrote {out_csv}")
    else:
        print("All symbols downloaded with no failures.")

    print("Download complete.")


if __name__ == "__main__":
    main()
