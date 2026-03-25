"""
Technical indicator feature engineering.

All features are computed using only data available at close of day t.
No future data leaks into the feature matrix.
"""

import numpy as np
import pandas as pd


# ── RSI ───────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ──────────────────────────────────────────────────────────────────────

def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    mid = close.rolling(period).mean()
    band = close.rolling(period).std()
    upper = mid + std * band
    lower = mid - std * band
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pct_b, width


# ── ATR ───────────────────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr / close.replace(0, np.nan)  # normalize


# ── Main feature builder ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single ticker.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume

    Returns
    -------
    DataFrame of features, NaN rows dropped.
    All features are lagged so they reflect information known at close of day t.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # RSI
    feat["rsi_5"] = _rsi(close, 5)
    feat["rsi_14"] = _rsi(close, 14)

    # MACD
    feat["macd"], feat["macd_signal"], feat["macd_hist"] = _macd(close)
    # Normalize MACD by price level
    feat["macd"] /= close
    feat["macd_signal"] /= close
    feat["macd_hist"] /= close

    # Bollinger Bands
    feat["bb_pct"], feat["bb_width"] = _bollinger(close)

    # ATR
    feat["atr_14"] = _atr(high, low, close)

    # Price momentum (already lagged: ret over [t-N, t])
    for n in [5, 10, 20, 60]:
        feat[f"mom_{n}"] = close.pct_change(n)

    # Recent lagged daily returns (t-1, t-2, t-3, t-5)
    for lag in [1, 2, 3, 5]:
        feat[f"ret_{lag}"] = close.pct_change(1).shift(lag - 1)

    # Volume ratio vs 20-day average
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / vol_ma.replace(0, np.nan)

    # Price relative to moving averages (normalized)
    for n in [10, 20, 50]:
        ma = close.rolling(n).mean()
        feat[f"price_ma_{n}"] = (close - ma) / ma.replace(0, np.nan)

    feat.dropna(inplace=True)
    return feat


FEATURE_NAMES = [
    "rsi_5", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "atr_14",
    "mom_5", "mom_10", "mom_20", "mom_60",
    "ret_1", "ret_2", "ret_3", "ret_5",
    "vol_ratio",
    "price_ma_10", "price_ma_20", "price_ma_50",
]
