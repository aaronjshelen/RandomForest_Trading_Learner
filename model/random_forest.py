"""
Random Forest model layer.

Builds a single shared model across all tickers by concatenating their
feature matrices (with a ticker indicator feature). Uses walk-forward CV
to produce out-of-sample predictions for the full history.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from data.fetcher import load_all
from features.indicators import compute_features
from features.labels import compute_labels
from model.walk_forward import WalkForwardSplitter, Fold
import config


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch data, compute features and labels for all tickers, stack into one
    flat DataFrame. A `ticker_id` column (integer-encoded) is added so the
    model can learn ticker-specific patterns.

    Returns
    -------
    X : DataFrame of features, indexed by (date, ticker) MultiIndex
    y : Series of integer labels {-1, 0, 1}
    """
    raw_data = load_all(tickers, start, end)
    encoder = LabelEncoder()
    encoder.fit(tickers)

    frames = []
    for ticker, df in raw_data.items():
        feat = compute_features(df)
        labels = compute_labels(
            df["Close"],
            horizon=config.FORWARD_HORIZON,
            buy_thresh=config.BUY_THRESHOLD,
            sell_thresh=config.SELL_THRESHOLD,
        )
        # Align features and labels on the same index
        aligned = feat.join(labels.rename("label"), how="inner")
        aligned.dropna(subset=["label"], inplace=True)
        aligned["label"] = aligned["label"].astype(int)
        aligned["ticker_id"] = int(encoder.transform([ticker])[0])
        aligned.index = pd.MultiIndex.from_arrays(
            [aligned.index, [ticker] * len(aligned)],
            names=["date", "ticker"]
        )
        frames.append(aligned)

    combined = pd.concat(frames).sort_index(level="date")
    y = combined.pop("label")
    X = combined
    return X, y


# ── Training & prediction ─────────────────────────────────────────────────────

def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a RandomForestClassifier and return predictions + probabilities.

    Returns
    -------
    y_pred  : (n_test,) integer array of predicted labels
    y_proba : (n_test, 3) probability array, columns ordered by model.classes_
    """
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return y_pred, y_proba, clf


# ── Walk-forward runner ───────────────────────────────────────────────────────

def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplitter,
    params: dict,
    verbose: bool = True,
) -> tuple[pd.DataFrame, object]:
    """
    Run walk-forward cross-validation. Returns a signals DataFrame and the
    model fitted on the final fold's training data (used for feature importance).

    The dataset index is a (date, ticker) MultiIndex. The splitter operates
    on the sorted unique dates so that all tickers on a given date are always
    in the same fold.

    Returns
    -------
    signals : DataFrame with columns [date, ticker, signal, prob_sell,
                                       prob_hold, prob_buy]
    final_model : the RandomForestClassifier from the last fold
    """
    # Work on date-level positions (all tickers per date move together)
    dates = X.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    folds = splitter.split(unique_dates)

    if verbose:
        splitter.summary(folds)

    all_results = []
    final_model = None

    for i, fold in enumerate(folds):
        train_dates = unique_dates[fold.train_idx]
        test_dates = unique_dates[fold.test_idx]

        mask_train = dates.isin(train_dates)
        mask_test = dates.isin(test_dates)

        X_train, y_train = X[mask_train], y[mask_train]
        X_test = X[mask_test]
        test_index = X_test.index  # (date, ticker) MultiIndex

        if verbose:
            acc_str = ""
        y_pred, y_proba, model = train_and_predict(X_train, y_train, X_test, params)

        # Map proba columns to fixed order: Sell=-1, Hold=0, Buy=1
        class_order = {c: idx for idx, c in enumerate(model.classes_)}
        sell_col = class_order.get(-1, None)
        hold_col = class_order.get(0, None)
        buy_col = class_order.get(1, None)

        result = pd.DataFrame({
            "date": test_index.get_level_values("date"),
            "ticker": test_index.get_level_values("ticker"),
            "signal": y_pred,
            "prob_sell": y_proba[:, sell_col] if sell_col is not None else 0.0,
            "prob_hold": y_proba[:, hold_col] if hold_col is not None else 0.0,
            "prob_buy":  y_proba[:, buy_col]  if buy_col  is not None else 0.0,
        })

        # Train accuracy for overfitting check
        train_acc = (model.predict(X_train) == y_train).mean()
        test_acc = (y_pred == y[mask_test].values).mean()
        if verbose:
            print(
                f"  Fold {i+1:2d} | "
                f"train acc: {train_acc:.3f} | test acc: {test_acc:.3f} | "
                f"test rows: {len(result)}"
            )

        all_results.append(result)
        final_model = model

    signals = pd.concat(all_results, ignore_index=True)
    signals["date"] = pd.to_datetime(signals["date"])
    return signals, final_model


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
