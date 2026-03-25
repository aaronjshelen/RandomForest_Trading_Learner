"""
Vectorized multi-stock backtester.

Design principles:
- All returns computed with a 1-day lag on weights (weights decided at close
  of day t, applied to returns of day t+1) — no lookahead.
- Transaction costs applied on turnover (absolute change in weight per ticker).
- Single initial capital; portfolio value compounded daily.
"""

import numpy as np
import pandas as pd


def run(
    price_data: dict[str, pd.DataFrame],
    weights_df: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_capital: float = 100_000,
    rebalance_freq: int = 1,
    vol_target: float = 0.0,
    vol_lookback: int = 20,
    regime_filter: bool = False,
    regime_ticker: str = "SPY",
    regime_ma: int = 200,
    regime_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Run the backtest.

    Parameters
    ----------
    price_data       : Dict ticker → OHLCV DataFrame
    weights_df       : Date × ticker weight matrix (from portfolio.optimizer)
    transaction_cost : One-way cost as fraction of notional traded
    initial_capital  : Starting portfolio value in dollars

    Returns
    -------
    DataFrame with columns:
        gross_return   : Daily return before costs
        turnover       : Sum of |Δweight| across all tickers
        cost           : Transaction cost for the day
        net_return     : gross_return - cost
        portfolio_value: Cumulative compounded portfolio value
    """
    tickers = weights_df.columns.tolist()

    # Build aligned close price matrix
    close_matrix = pd.DataFrame({
        t: price_data[t]["Close"] for t in tickers if t in price_data
    })
    close_matrix.index = pd.to_datetime(close_matrix.index)
    close_matrix.sort_index(inplace=True)

    # Daily returns for each ticker
    daily_returns = close_matrix.pct_change()

    # Align weights and returns on common dates
    common_dates = weights_df.index.intersection(daily_returns.index)
    weights = weights_df.loc[common_dates, tickers].fillna(0.0)
    rets = daily_returns.loc[common_dates, tickers].fillna(0.0)

    # Apply rebalance frequency: only update weights every N days,
    # hold previous weights on non-rebalance days (drastically cuts turnover)
    if rebalance_freq > 1:
        rebalance_mask = pd.Series(False, index=weights.index)
        rebalance_mask.iloc[::rebalance_freq] = True
        weights = weights.where(rebalance_mask, other=np.nan).ffill().fillna(0.0)

    # Regime filter: zero out all weights on bear-market days.
    # Bear = SPY close < SPY N-day MA (computed from regime_prices or price_data).
    # Uses shift(1) so today's regime decision uses yesterday's close — no lookahead.
    if regime_filter:
        spy_close = (
            regime_prices
            if regime_prices is not None
            else pd.Series(
                price_data[regime_ticker]["Close"],
                index=pd.to_datetime(price_data[regime_ticker].index),
            )
        )
        spy_close.index = pd.to_datetime(spy_close.index)
        spy_ma = spy_close.rolling(regime_ma, min_periods=regime_ma // 2).mean()
        in_regime = (spy_close > spy_ma).astype(float)           # 1 = bull, 0 = bear
        in_regime = in_regime.shift(1).reindex(weights.index, method="ffill").fillna(0.0)
        weights = weights.multiply(in_regime, axis=0)

    # Shift weights by 1 day: weights decided at close of t applied at t+1
    weights_shifted = weights.shift(1).fillna(0.0)

    # Volatility targeting: scale weights down when realized vol exceeds target.
    # Step 1: compute unscaled portfolio returns as a vol proxy.
    # Step 2: compute rolling realized vol (annualized), lagged 1 day (no lookahead).
    # Step 3: scale = vol_target / realized_vol, capped at 1.0 (never leverage).
    # Step 4: apply scale to weights_shifted before computing P&L and turnover.
    if vol_target > 0:
        proxy_returns = (weights_shifted * rets).sum(axis=1)
        realized_vol = (
            proxy_returns
            .rolling(vol_lookback, min_periods=max(5, vol_lookback // 2))
            .std()
            .shift(1)          # lag by 1: use vol known at start of day
            * np.sqrt(252)     # annualize
        ).fillna(1.0)
        scale = (vol_target / realized_vol).clip(upper=1.0)
        weights_shifted = weights_shifted.multiply(scale, axis=0)

    # Gross portfolio return each day
    gross_return = (weights_shifted * rets).sum(axis=1)

    # Turnover: sum of absolute weight changes (use scaled weights)
    turnover = weights_shifted.diff().abs().sum(axis=1).fillna(0.0)

    # Cost: one-way transaction cost on each leg of the trade
    cost = turnover * transaction_cost

    net_return = gross_return - cost

    portfolio_value = (1 + net_return).cumprod() * initial_capital

    result = pd.DataFrame({
        "gross_return": gross_return,
        "turnover": turnover,
        "cost": cost,
        "net_return": net_return,
        "portfolio_value": portfolio_value,
    }, index=common_dates)

    return result


def benchmark_buy_and_hold(
    price_data: dict[str, pd.DataFrame],
    ticker: str,
    initial_capital: float = 100_000,
) -> pd.Series:
    """
    Compute buy-and-hold portfolio value for a benchmark ticker (e.g., SPY).
    """
    close = price_data[ticker]["Close"].sort_index()
    daily_rets = close.pct_change().fillna(0.0)
    return (1 + daily_rets).cumprod() * initial_capital
