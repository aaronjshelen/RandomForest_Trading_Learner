"""
Plotting utilities. Saves figures to plots/output/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = Path(__file__).parent / "output"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot: {path}")
    plt.close(fig)


def equity_curve(
    portfolio_value: pd.Series,
    benchmark_value: pd.Series,
    title: str = "Equity Curve",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    # Normalize both to 100 at start
    strat_norm = portfolio_value / portfolio_value.iloc[0] * 100
    bench_norm = benchmark_value / benchmark_value.iloc[0] * 100

    ax.plot(strat_norm.index, strat_norm, label="Strategy", linewidth=1.5, color="#2196F3")
    ax.plot(bench_norm.index, bench_norm, label="Buy & Hold (SPY)", linewidth=1.5,
            color="#FF9800", linestyle="--", alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Normalized Value (base=100)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "equity_curve.png")


def drawdown_chart(portfolio_value: pd.Series, title: str = "Drawdown") -> None:
    roll_max = portfolio_value.cummax()
    drawdown = (portfolio_value - roll_max) / roll_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown, 0, color="#F44336", alpha=0.5, label="Drawdown")
    ax.plot(drawdown.index, drawdown, color="#F44336", linewidth=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "drawdown.png")


def feature_importance(importance: pd.Series, top_n: int = 20, title: str = "Feature Importance") -> None:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top[::-1].plot.barh(ax=ax, color="#4CAF50", edgecolor="white")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "feature_importance.png")


def signal_distribution(signals_df: pd.DataFrame) -> None:
    label_map = {-1: "Sell", 0: "Hold", 1: "Buy"}
    grouped = signals_df.groupby("ticker")["signal"].value_counts(normalize=True).unstack(fill_value=0)
    grouped.rename(columns=label_map, inplace=True)
    for col in ["Sell", "Hold", "Buy"]:
        if col not in grouped.columns:
            grouped[col] = 0.0
    grouped = grouped[["Sell", "Hold", "Buy"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.plot.bar(ax=ax, color=["#F44336", "#9E9E9E", "#4CAF50"], edgecolor="white")
    ax.set_title("Signal Distribution by Ticker", fontsize=14)
    ax.set_ylabel("Fraction of days")
    ax.set_xlabel("")
    ax.legend(title="Signal")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "signal_distribution.png")


def buy_sell_chart(
    signals_df: pd.DataFrame,
    price_data: dict,
    tickers: list | None = None,
) -> None:
    """
    For each ticker, plot the close price with Buy (▲) and Sell (▼) markers.

    Parameters
    ----------
    signals_df : Output of run_walk_forward(); columns [date, ticker, signal, ...]
    price_data : Dict of ticker → OHLCV DataFrame
    tickers    : Subset of tickers to plot. Defaults to all in signals_df.
    """
    if tickers is None:
        tickers = sorted(signals_df["ticker"].unique())

    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        if ticker not in price_data:
            continue

        close = price_data[ticker]["Close"].sort_index()
        close.index = pd.to_datetime(close.index)

        ticker_signals = signals_df[signals_df["ticker"] == ticker].copy()
        ticker_signals["date"] = pd.to_datetime(ticker_signals["date"])

        buys  = ticker_signals[ticker_signals["signal"] ==  1]
        sells = ticker_signals[ticker_signals["signal"] == -1]

        # Align signal dates to close prices (nearest available date)
        buy_prices  = close.reindex(buys["date"],  method="nearest")
        sell_prices = close.reindex(sells["date"], method="nearest")

        ax.plot(close.index, close, color="#455A64", linewidth=1.0, label="Close", zorder=1)

        ax.scatter(
            buy_prices.index, buy_prices.values,
            marker="^", color="#4CAF50", s=60, zorder=3, label="Buy",
        )
        ax.scatter(
            sell_prices.index, sell_prices.values,
            marker="v", color="#F44336", s=60, zorder=3, label="Sell",
        )

        ax.set_title(f"{ticker} — Buy / Sell Signals", fontsize=13)
        ax.set_ylabel("Price ($)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "buy_sell_signals.png")


def weights_over_time(weights_df: pd.DataFrame, title: str = "Portfolio Weights Over Time") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    weights_df.plot.area(ax=ax, stacked=True, alpha=0.75, linewidth=0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "weights_over_time.png")
