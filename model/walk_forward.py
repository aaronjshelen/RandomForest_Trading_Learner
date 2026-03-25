"""
Expanding-window walk-forward cross-validation splitter for time series.

Unlike scikit-learn's TimeSeriesSplit, this splitter:
- Uses an expanding training window (not rolling)
- Enforces a configurable gap between train end and test start
  (equal to FORWARD_HORIZON to prevent label leakage)
- Operates on a DatetimeIndex directly
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Fold:
    train_idx: np.ndarray  # integer positions into the full dataset index
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    """
    Expanding-window walk-forward splitter.

    Parameters
    ----------
    min_train : int
        Minimum number of rows required in the training set.
    test_size : int
        Number of rows in each test fold.
    step : int
        Number of rows to advance the window between folds.
    gap : int
        Rows to skip between train end and test start (set to FORWARD_HORIZON
        to prevent any overlap between training labels and test features).
    """

    def __init__(self, min_train: int, test_size: int, step: int, gap: int = 0):
        self.min_train = min_train
        self.test_size = test_size
        self.step = step
        self.gap = gap

    def split(self, index: pd.DatetimeIndex) -> list[Fold]:
        n = len(index)
        folds = []
        train_end = self.min_train - 1  # inclusive index position

        while True:
            test_start = train_end + 1 + self.gap
            test_end = test_start + self.test_size - 1

            if test_end >= n:
                break

            folds.append(Fold(
                train_idx=np.arange(0, train_end + 1),
                test_idx=np.arange(test_start, test_end + 1),
                train_start=index[0],
                train_end=index[train_end],
                test_start=index[test_start],
                test_end=index[test_end],
            ))

            train_end += self.step

        return folds

    def summary(self, folds: list[Fold]) -> None:
        print(f"Walk-forward CV: {len(folds)} folds")
        for i, f in enumerate(folds):
            print(
                f"  Fold {i+1:2d}: "
                f"train {f.train_start.date()}→{f.train_end.date()} "
                f"({len(f.train_idx)} rows) | "
                f"test {f.test_start.date()}→{f.test_end.date()} "
                f"({len(f.test_idx)} rows)"
            )
