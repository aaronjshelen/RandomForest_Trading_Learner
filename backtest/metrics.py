"""
Performance metrics for strategy evaluation.
"""

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    excess = returns - rf / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def max_drawdown(portfolio_value: pd.Series) -> float:
    roll_max = portfolio_value.cummax()
    drawdown = (portfolio_value - roll_max) / roll_max
    return float(drawdown.min())


def cagr(portfolio_value: pd.Series, periods: int = 252) -> float:
    n_days = len(portfolio_value)
    if n_days < 2:
        return 0.0
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
    years = n_days / periods
    return float(total_return ** (1 / years) - 1)


def calmar(portfolio_value: pd.Series, periods: int = 252) -> float:
    md = max_drawdown(portfolio_value)
    if md == 0:
        return 0.0
    return cagr(portfolio_value, periods) / abs(md)


def win_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def summarize(
    strategy_result: pd.DataFrame,
    benchmark_value: pd.Series,
    rf: float = 0.045,
    periods: int = 252,
) -> pd.DataFrame:
    """
    Print and return a comparison table of strategy vs. benchmark.

    Parameters
    ----------
    strategy_result : Output of backtester.run()
    benchmark_value : Buy-and-hold portfolio value (pd.Series)
    rf              : Annual risk-free rate (default 4.5%)
    periods         : Trading days per year
    """
    s_ret = strategy_result["net_return"]
    b_ret = benchmark_value.pct_change().fillna(0.0)

    metrics = {
        "Total Return": [
            f"{(strategy_result['portfolio_value'].iloc[-1] / strategy_result['portfolio_value'].iloc[0] - 1):.1%}",
            f"{(benchmark_value.iloc[-1] / benchmark_value.iloc[0] - 1):.1%}",
        ],
        "CAGR": [
            f"{cagr(strategy_result['portfolio_value'], periods):.1%}",
            f"{cagr(benchmark_value, periods):.1%}",
        ],
        "Sharpe Ratio": [
            f"{sharpe(s_ret, rf, periods):.2f}",
            f"{sharpe(b_ret, rf, periods):.2f}",
        ],
        "Max Drawdown": [
            f"{max_drawdown(strategy_result['portfolio_value']):.1%}",
            f"{max_drawdown(benchmark_value):.1%}",
        ],
        "Calmar Ratio": [
            f"{calmar(strategy_result['portfolio_value'], periods):.2f}",
            f"{calmar(benchmark_value, periods):.2f}",
        ],
        "Win Rate": [
            f"{win_rate(s_ret):.1%}",
            f"{win_rate(b_ret):.1%}",
        ],
        "Avg Daily Cost": [
            f"{strategy_result['cost'].mean():.4%}",
            "N/A",
        ],
    }

    df = pd.DataFrame(metrics, index=["Strategy", "Benchmark"]).T
    print("\n" + "=" * 50)
    print("  Performance Summary")
    print("=" * 50)
    print(df.to_string())
    print("=" * 50 + "\n")
    return df
