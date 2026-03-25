"""
Portfolio allocation layer.

Takes the RF signal probabilities for each ticker on each day and converts
them into portfolio weights. Two modes are supported:

1. mean_variance  — Maximize Sharpe ratio subject to long-only and max-weight
                    constraints. Expected returns come from the RF probability
                    score (prob_buy - prob_sell). Covariance is estimated from
                    a rolling window of historical returns.

2. risk_parity    — Equal risk contribution (ERC). Ignores expected return
                    estimates; each asset contributes equally to portfolio
                    variance. More robust when return estimates are noisy.

A "signal filter" is applied before optimization: tickers with a Sell signal
on a given day are excluded from the long portfolio. This keeps the optimizer
focused on the tradeable opportunity set.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Mean-variance (maximize Sharpe) ──────────────────────────────────────────

def _max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    """
    Maximize Sharpe = (w @ mu) / sqrt(w @ cov @ w).

    Equivalent to minimizing -Sharpe subject to:
      sum(w) == 1, 0 <= w_i <= max_weight
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w + 1e-10)
        return -port_ret / port_vol

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    if result.success:
        w = np.clip(result.x, 0, max_weight)
        w /= w.sum()
        return w
    # Fallback: equal weight
    return np.ones(n) / n


# ── Risk parity ───────────────────────────────────────────────────────────────

def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """
    Equal risk contribution weights via iterative (naive) approach.
    Each asset contributes equally to total portfolio variance.
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    def risk_budget_objective(w):
        w = np.array(w)
        port_var = w @ cov @ w
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib
        target = port_var / n
        return np.sum((risk_contrib - target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(1e-6, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(
        risk_budget_objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if result.success:
        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        return w
    return np.ones(n) / n


# ── Daily weight computation ──────────────────────────────────────────────────

def compute_weights(
    signals_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    mode: str = "mean_variance",
    lookback: int = 60,
    max_weight: float = 0.40,
    signal_scaling: bool = False,
) -> pd.DataFrame:
    """
    Compute portfolio weights for every date in signals_df.

    Parameters
    ----------
    signals_df : Output of run_walk_forward(); columns [date, ticker, signal,
                 prob_sell, prob_hold, prob_buy]
    price_data : Dict of ticker → OHLCV DataFrame (for covariance estimation)
    mode       : "mean_variance" or "risk_parity"
    lookback        : Rolling window (days) for covariance estimation
    max_weight      : Max allocation to any single ticker
    signal_scaling  : If True, scale risk-parity weights by RF confidence
                      score (prob_buy - prob_sell) then renormalize.
                      High-conviction signals get more weight, weak ones less.

    Returns
    -------
    weights_df : DataFrame with dates as index, tickers as columns, values are
                 portfolio weights [0, 1] summing to 1 on each row.
    """
    tickers = sorted(signals_df["ticker"].unique())
    dates = sorted(signals_df["date"].unique())

    # Build returns matrix (aligned on common dates)
    returns = pd.DataFrame({
        t: price_data[t]["Close"].pct_change()
        for t in tickers
        if t in price_data
    }).sort_index()

    weights_rows = []

    for date in dates:
        day_signals = signals_df[signals_df["date"] == date].set_index("ticker")

        # Exclude Sell signals from long portfolio
        tradeable = day_signals[day_signals["signal"] != -1].index.tolist()

        if not tradeable:
            # All signals are Sell: go to cash (zero weights)
            weights_rows.append({"date": date, **{t: 0.0 for t in tickers}})
            continue

        # Expected return proxy: P(Buy) - P(Sell)
        mu = np.array([
            day_signals.loc[t, "prob_buy"] - day_signals.loc[t, "prob_sell"]
            for t in tradeable
        ])

        # Rolling covariance
        hist = returns.loc[:date].iloc[-lookback:]
        hist_tradeable = hist[[t for t in tradeable if t in hist.columns]].dropna()
        valid_tickers = hist_tradeable.columns.tolist()

        if len(valid_tickers) < 2:
            # Can't optimize with fewer than 2 assets — equal weight
            row = {"date": date, **{t: 0.0 for t in tickers}}
            for t in valid_tickers:
                row[t] = 1.0 / len(valid_tickers)
            weights_rows.append(row)
            continue

        cov = hist_tradeable.cov().values

        # Align mu to valid_tickers
        mu_valid = np.array([
            day_signals.loc[t, "prob_buy"] - day_signals.loc[t, "prob_sell"]
            for t in valid_tickers
        ])

        if mode == "mean_variance":
            w = _max_sharpe_weights(mu_valid, cov, max_weight)
        else:
            w = _risk_parity_weights(cov)
            if signal_scaling:
                # Scale each weight by RF confidence: prob_buy - prob_sell.
                # This keeps the diversification from risk parity but tilts
                # toward stocks the model is most certain about.
                conf = np.array([
                    max(0.0, day_signals.loc[t, "prob_buy"] - day_signals.loc[t, "prob_sell"])
                    for t in valid_tickers
                ])
                w = w * conf
                total = w.sum()
                if total > 1e-8:
                    w = w / total
                else:
                    w = np.ones(len(valid_tickers)) / len(valid_tickers)
                # Apply max_weight cap and renormalize
                w = np.clip(w, 0.0, max_weight)
                w = w / w.sum()

        row = {"date": date, **{t: 0.0 for t in tickers}}
        for t, wi in zip(valid_tickers, w):
            row[t] = float(wi)
        weights_rows.append(row)

    weights_df = pd.DataFrame(weights_rows).set_index("date")
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_df.sort_index(inplace=True)
    return weights_df
