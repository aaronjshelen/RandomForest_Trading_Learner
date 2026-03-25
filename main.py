"""
Random Forest Strategy Learner — main entry point.

Usage
-----
# Default config (tickers/dates from config.py)
python main.py

# Custom tickers and date range
python main.py --tickers AAPL MSFT GOOGL JPM XOM --start 2015-01-01 --end 2024-12-31

# Switch to risk parity optimizer
python main.py --optimizer risk_parity

# Verbose fold output
python main.py --verbose
"""

import argparse
import sys

import pandas as pd
import config
from data.fetcher import load_all, fetch
from features.labels import label_distribution
from model.random_forest import build_dataset, run_walk_forward, get_feature_importance
from model.walk_forward import WalkForwardSplitter
from portfolio.optimizer import compute_weights
from backtest.backtester import run as run_backtest, benchmark_buy_and_hold
from backtest.metrics import summarize
from plots import plotter


def parse_args():
    parser = argparse.ArgumentParser(description="RF Strategy Learner")
    parser.add_argument("--tickers", nargs="+", default=config.TICKERS)
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.END_DATE)
    parser.add_argument(
        "--optimizer",
        choices=["mean_variance", "risk_parity"],
        default=config.OPTIMIZER,
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    parser.add_argument("--verbose", action="store_true", help="Print fold-level details")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  RF Strategy Learner")
    print("=" * 60)
    print(f"  Tickers   : {args.tickers}")
    print(f"  Period    : {args.start} → {args.end}")
    print(f"  Optimizer : {args.optimizer}")
    print("=" * 60 + "\n")

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("Step 1/6: Building feature/label dataset...")
    X, y = build_dataset(args.tickers, args.start, args.end)
    print(f"  Dataset shape: {X.shape[0]} rows × {X.shape[1]} features")

    print("\n  Label distribution:")
    all_labels_dist = label_distribution(y)
    print(all_labels_dist.to_string())

    # ── 2. Walk-forward modelling ─────────────────────────────────────────────
    print("\nStep 2/6: Running walk-forward RF model...")
    splitter = WalkForwardSplitter(
        min_train=config.MIN_TRAIN_DAYS,
        test_size=config.TEST_DAYS,
        step=config.STEP_DAYS,
        gap=config.LABEL_GAP,
    )
    signals_df, final_model = run_walk_forward(
        X, y, splitter, config.RF_PARAMS, verbose=args.verbose
    )
    print(f"  Total signal rows generated: {len(signals_df)}")

    # ── 3. Portfolio weights ──────────────────────────────────────────────────
    print("\nStep 3/6: Computing portfolio weights...")
    price_data = load_all(args.tickers, args.start, args.end)
    weights_df = compute_weights(
        signals_df,
        price_data,
        mode=args.optimizer,
        lookback=config.COV_LOOKBACK,
        max_weight=config.MAX_WEIGHT,
        signal_scaling=config.SIGNAL_STRENGTH_SCALING,
    )
    print(f"  Weight matrix: {weights_df.shape[0]} dates × {weights_df.shape[1]} tickers")

    # ── 4. Backtest ───────────────────────────────────────────────────────────
    print("\nStep 4/6: Running backtest...")
    print("  Fetching benchmark (SPY)...")
    benchmark_prices = load_all([config.BENCHMARK], args.start, args.end)
    spy_close = benchmark_prices[config.BENCHMARK]["Close"]
    spy_close.index = pd.to_datetime(spy_close.index)

    if config.REGIME_FILTER:
        print(f"  Regime filter: ON  (SPY > {config.REGIME_MA}-day MA → trade, else cash)")
    else:
        print("  Regime filter: OFF")

    result = run_backtest(
        price_data,
        weights_df,
        transaction_cost=config.TRANSACTION_COST,
        initial_capital=config.INITIAL_CAPITAL,
        rebalance_freq=config.REBALANCE_FREQ,
        vol_target=config.VOL_TARGET,
        vol_lookback=config.VOL_LOOKBACK,
        regime_filter=config.REGIME_FILTER,
        regime_ticker=config.BENCHMARK,
        regime_ma=config.REGIME_MA,
        regime_prices=spy_close,
    )
    benchmark_value = benchmark_buy_and_hold(
        benchmark_prices, config.BENCHMARK, config.INITIAL_CAPITAL
    )
    # Align benchmark to backtest dates
    benchmark_value = benchmark_value.reindex(result.index, method="ffill")

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    print("\nStep 5/6: Computing performance metrics...")
    metrics_df = summarize(result, benchmark_value)

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Step 6/6: Saving plots...")
        plotter.equity_curve(result["portfolio_value"], benchmark_value)
        plotter.drawdown_chart(result["portfolio_value"])

        importance = get_feature_importance(final_model, X.columns.tolist())
        plotter.feature_importance(importance)
        plotter.signal_distribution(signals_df)
        plotter.weights_over_time(weights_df)
        plotter.buy_sell_chart(signals_df, price_data, tickers=args.tickers)
    else:
        print("Step 6/6: Plots skipped (--no-plots).")

    print("\nDone.")
    return result, metrics_df, signals_df, weights_df


if __name__ == "__main__":
    main()
