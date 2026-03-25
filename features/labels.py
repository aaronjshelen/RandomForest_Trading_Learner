"""
Label generation for Buy / Hold / Sell classification.

Labels are based on forward N-day returns computed from the close price.
The label assigned to day t reflects the return from close(t) to close(t+N).

IMPORTANT: Labels are aligned to the OBSERVATION date t, not the future date
t+N. This means on day t, after the market closes, we know:
  - All features (computed from data up to and including t)
  - The label we want to predict: what will the return be over the next N days?

The last `horizon` rows have NaN labels and must be dropped before training.
"""

import pandas as pd


def compute_labels(
    close: pd.Series,
    horizon: int,
    buy_thresh: float,
    sell_thresh: float,
) -> pd.Series:
    """
    Compute Buy/Hold/Sell labels aligned to observation date t.

    Parameters
    ----------
    close       : Close price series
    horizon     : Forward look-ahead in trading days
    buy_thresh  : Forward return above this → Buy (1)
    sell_thresh : Forward return below this → Sell (-1)

    Returns
    -------
    pd.Series with values in {-1, 0, 1}, NaN rows for last `horizon` days.
    """
    forward_return = close.shift(-horizon) / close - 1

    labels = pd.Series(0, index=close.index, dtype=int)
    labels[forward_return > buy_thresh] = 1
    labels[forward_return < sell_thresh] = -1

    # Last `horizon` rows have no valid forward price — set to NaN
    labels.iloc[-horizon:] = pd.NA
    return labels


def label_distribution(labels: pd.Series) -> pd.Series:
    counts = labels.dropna().value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)
    return pd.DataFrame({"count": counts, "pct": pct})
