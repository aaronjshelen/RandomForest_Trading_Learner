"""
Data fetching and caching layer using yfinance.

All data is stored as parquet files under data/cache/ to avoid redundant
network requests. Cache is considered fresh if the file exists and ends
within 1 day of the requested end date.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def _is_fresh(path: Path, end: str) -> bool:
    if not path.exists():
        return False
    df = pd.read_parquet(path)
    if df.empty:
        return False
    last_cached = df.index[-1]
    end_dt = pd.Timestamp(end)
    # Allow up to 5 trading days of slack (weekends, holidays)
    return last_cached >= end_dt - pd.tseries.offsets.BDay(5)


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for a single ticker. Uses parquet cache if fresh.

    Columns: Open, High, Low, Close, Volume
    Index: DatetimeIndex (UTC-naive)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)

    if _is_fresh(path, end):
        df = pd.read_parquet(path)
    else:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")
        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_parquet(path)
        print(f"  Fetched {ticker}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} rows)")

    # Trim to requested range
    df = df.loc[start:end]
    return df


def load_all(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for all tickers. Returns dict mapping ticker → DataFrame.
    """
    print(f"Loading data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch(ticker, start, end)
        except Exception as e:
            print(f"  WARNING: Failed to fetch {ticker}: {e}")
    return data
