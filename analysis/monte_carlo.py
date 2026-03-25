"""
Monte Carlo simulation for the RF Strategy Learner.

Runs the full pipeline once to get the strategy's actual daily net returns,
then bootstraps those returns (random resampling with replacement) to generate
N alternative equity paths. This answers:

  "Given these returns happened, what would outcomes look like under
   different orderings? How lucky or unlucky was our specific path?"

Usage
-----
python analysis/monte_carlo.py                        # static plots
python analysis/monte_carlo.py --live                 # animated live drawing
python analysis/monte_carlo.py --live --batch 5       # draw 5 paths per frame
python analysis/monte_carlo.py --sims 2000 --capital 1000000
"""

import sys
import argparse
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Sound support — Windows only, gracefully skipped on other platforms
try:
    import winsound
    _SOUND_AVAILABLE = True
except ImportError:
    _SOUND_AVAILABLE = False


def _tick(freq: int = 200, duration_ms: int = 25) -> None:
    """Play a short tick sound in a background thread (non-blocking)."""
    if _SOUND_AVAILABLE:
        threading.Thread(
            target=winsound.Beep, args=(freq, duration_ms), daemon=True
        ).start()

# Make sure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.fetcher import load_all
from features.labels import label_distribution
from model.random_forest import build_dataset, run_walk_forward
from model.walk_forward import WalkForwardSplitter
from portfolio.optimizer import compute_weights
from backtest.backtester import run as run_backtest, benchmark_buy_and_hold
from backtest.metrics import sharpe, max_drawdown, cagr, calmar

OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "output"


# ── Core simulation ───────────────────────────────────────────────────────────

def run_simulations(
    net_returns: np.ndarray,
    n_sims: int,
    initial_capital: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Bootstrap resample the daily net returns N times.

    Parameters
    ----------
    net_returns     : 1-D array of actual daily net returns
    n_sims          : Number of simulated paths
    initial_capital : Starting portfolio value

    Returns
    -------
    paths : (n_sims, len(net_returns)) array of portfolio values
    """
    rng = np.random.default_rng(seed)
    n = len(net_returns)
    paths = np.empty((n_sims, n))

    for i in range(n_sims):
        sampled = rng.choice(net_returns, size=n, replace=True)
        paths[i] = (1 + sampled).cumprod() * initial_capital

    return paths


def compute_sim_metrics(paths: np.ndarray, periods: int = 252) -> pd.DataFrame:
    """Compute CAGR, Sharpe, Max Drawdown, and Final Value for every path."""
    records = []
    for path in paths:
        pv = pd.Series(path)
        ret = pv.pct_change().fillna(0.0)
        records.append({
            "final_value": path[-1],
            "cagr":        cagr(pv, periods),
            "sharpe":      sharpe(ret, rf=0.045, periods=periods),
            "max_dd":      max_drawdown(pv),
        })
    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_paths(
    paths: np.ndarray,
    actual_path: pd.Series,
    initial_capital: float,
    n_sims: int,
) -> plt.Figure:
    """
    Plot all simulated equity paths with percentile bands and the actual path.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(paths.shape[1])

    # Percentile bands
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    # Draw all paths (thin, transparent)
    sample_paths = paths[::max(1, n_sims // 300)]  # draw at most ~300 lines
    for path in sample_paths:
        ax.plot(x, path, color="#90A4AE", linewidth=0.3, alpha=0.25)

    # Percentile fills
    ax.fill_between(x, p5,  p95, color="#1565C0", alpha=0.10, label="5–95th pct")
    ax.fill_between(x, p25, p75, color="#1565C0", alpha=0.20, label="25–75th pct")

    # Median
    ax.plot(x, p50, color="#1565C0", linewidth=1.8, linestyle="--", label="Median sim")

    # Actual backtest path (normalize to same start)
    actual_norm = actual_path.values / actual_path.iloc[0] * initial_capital
    # Trim/pad to match simulation length
    actual_plot = actual_norm[:paths.shape[1]]
    ax.plot(np.arange(len(actual_plot)), actual_plot,
            color="#F44336", linewidth=2.0, label="Actual backtest", zorder=5)

    ax.axhline(initial_capital, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title(f"Monte Carlo Simulation — {n_sims:,} Paths (Bootstrap Resampling)",
                 fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_distributions(metrics_df: pd.DataFrame, actual_metrics: dict) -> plt.Figure:
    """
    4-panel histogram grid: CAGR, Sharpe, Max Drawdown, Final Value.
    Actual backtest value shown as a red vertical line on each panel.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    panels = [
        ("cagr",        "CAGR",              "{:.1%}",  actual_metrics["cagr"]),
        ("sharpe",      "Sharpe Ratio",       "{:.2f}",  actual_metrics["sharpe"]),
        ("max_dd",      "Max Drawdown",       "{:.1%}",  actual_metrics["max_dd"]),
        ("final_value", "Final Value ($)",    "${:,.0f}", actual_metrics["final_value"]),
    ]

    for idx, (col, title, fmt, actual_val) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        data = metrics_df[col].dropna()

        ax.hist(data, bins=60, color="#1565C0", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(actual_val, color="#F44336", linewidth=2.0,
                   label=f"Actual: {fmt.format(actual_val)}")

        # 5th / 95th percentile markers
        p5  = np.percentile(data, 5)
        p95 = np.percentile(data, 95)
        ax.axvline(p5,  color="#FF9800", linewidth=1.2, linestyle="--",
                   label=f"5th pct: {fmt.format(p5)}")
        ax.axvline(p95, color="#4CAF50", linewidth=1.2, linestyle="--",
                   label=f"95th pct: {fmt.format(p95)}")

        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Monte Carlo — Metric Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(metrics_df: pd.DataFrame, actual_metrics: dict, n_sims: int) -> None:
    print("\n" + "=" * 58)
    print(f"  Monte Carlo Summary  ({n_sims:,} simulations)")
    print("=" * 58)
    rows = [
        ("CAGR",         "cagr",        "{:.1%}"),
        ("Sharpe",       "sharpe",      "{:.2f}"),
        ("Max Drawdown", "max_dd",      "{:.1%}"),
        ("Final Value",  "final_value", "${:,.0f}"),
    ]
    header = f"{'Metric':<16} {'Actual':>10} {'5th%':>10} {'Median':>10} {'95th%':>10}"
    print(header)
    print("-" * 58)
    for label, col, fmt in rows:
        data = metrics_df[col].dropna()
        print(
            f"{label:<16} "
            f"{fmt.format(actual_metrics[col]):>10} "
            f"{fmt.format(np.percentile(data, 5)):>10} "
            f"{fmt.format(np.percentile(data, 50)):>10} "
            f"{fmt.format(np.percentile(data, 95)):>10}"
        )

    beat_pct = (metrics_df["cagr"] > actual_metrics["cagr"]).mean()
    print("-" * 58)
    print(f"  Paths beating actual CAGR : {beat_pct:.1%} of simulations")

    worse_dd = (metrics_df["max_dd"] < actual_metrics["max_dd"]).mean()
    print(f"  Paths with worse drawdown : {worse_dd:.1%} of simulations")
    print("=" * 58 + "\n")


# ── Live animation ───────────────────────────────────────────────────────────

def plot_paths_live(
    paths: np.ndarray,
    actual_path: pd.Series,
    initial_capital: float,
    n_sims: int,
    batch: int = 10,
) -> plt.Figure:
    """
    Draw Monte Carlo paths one batch at a time so the user sees them
    accumulate live. After all paths are drawn, overlay the percentile
    bands and the actual backtest path as a finale.

    Parameters
    ----------
    paths          : (n_sims, n_days) array of simulated portfolio values
    actual_path    : Actual backtest portfolio value series
    initial_capital: Starting capital (used for the zero-return reference line)
    n_sims         : Total number of simulations
    batch          : Paths added per animation frame (lower = slower/smoother)
    """
    # Colors
    COLOR_FRESH = "#1A237E"   # dark navy  — freshly added batch
    COLOR_RESTED = "#90A4AE"  # light grey — all previous batches

    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(paths.shape[1])

    ax.axhline(initial_capital, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title(f"Monte Carlo — Live Drawing ({n_sims:,} paths)", fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(True, alpha=0.3)

    prev_batch_lines = []  # line objects from the previous batch

    # Draw in batches
    for start in range(0, n_sims, batch):
        # Fade previous batch to rested colour
        for line in prev_batch_lines:
            line.set_color(COLOR_RESTED)
            line.set_alpha(0.30)
            line.set_linewidth(0.35)

        # Draw new batch in fresh (dark) colour
        new_lines = []
        for path in paths[start : start + batch]:
            (ln,) = ax.plot(x, path, color=COLOR_FRESH, linewidth=0.9, alpha=0.85)
            new_lines.append(ln)

        prev_batch_lines = new_lines
        count = min(start + batch, n_sims)
        ax.set_title(
            f"Monte Carlo — Drawing paths... {count:,} / {n_sims:,}", fontsize=14
        )
        fig.canvas.draw()
        _tick()
        plt.pause(0.001)

    # Fade the very last batch to rested colour too
    for line in prev_batch_lines:
        line.set_color(COLOR_RESTED)
        line.set_alpha(0.30)
        line.set_linewidth(0.35)

    # ── Finale: overlay percentile bands ─────────────────────────────────────
    ax.set_title("Monte Carlo — Adding percentile bands...", fontsize=14)
    fig.canvas.draw()
    plt.pause(0.3)

    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    ax.fill_between(x, p5,  p95, color="#1565C0", alpha=0.12, label="5–95th pct")
    fig.canvas.draw()
    plt.pause(0.2)

    ax.fill_between(x, p25, p75, color="#1565C0", alpha=0.22, label="25–75th pct")
    fig.canvas.draw()
    plt.pause(0.2)

    ax.plot(x, p50, color="#1565C0", linewidth=1.8, linestyle="--", label="Median sim")
    fig.canvas.draw()
    plt.pause(0.3)

    # ── Finale: actual backtest path ──────────────────────────────────────────
    ax.set_title("Monte Carlo — Adding actual backtest path...", fontsize=14)
    fig.canvas.draw()
    plt.pause(0.3)

    actual_norm = actual_path.values / actual_path.iloc[0] * initial_capital
    actual_plot = actual_norm[: paths.shape[1]]
    ax.plot(
        np.arange(len(actual_plot)), actual_plot,
        color="#F44336", linewidth=2.2, label="Actual backtest", zorder=5,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"Monte Carlo Simulation — {n_sims:,} Paths (Bootstrap Resampling)", fontsize=14
    )
    fig.canvas.draw()
    plt.pause(0.5)

    plt.ioff()
    return fig


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulation")
    parser.add_argument("--sims",    type=int,   default=1000)
    parser.add_argument("--capital", type=float, default=float(config.INITIAL_CAPITAL))
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--live",    action="store_true",
                        help="Animate paths being drawn live")
    parser.add_argument("--batch",   type=int,   default=10,
                        help="Paths drawn per frame in live mode (default 10)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Monte Carlo — RF Strategy Learner")
    print("=" * 60)
    print(f"  Simulations : {args.sims:,}")
    print(f"  Capital     : ${args.capital:,.0f}")
    print("=" * 60)

    # ── Run the full pipeline to get net returns ──────────────────────────────
    print("\nRunning pipeline to generate strategy returns...")

    X, y = build_dataset(config.TICKERS, config.START_DATE, config.END_DATE)

    splitter = WalkForwardSplitter(
        min_train=config.MIN_TRAIN_DAYS,
        test_size=config.TEST_DAYS,
        step=config.STEP_DAYS,
        gap=config.LABEL_GAP,
    )
    signals_df, _ = run_walk_forward(X, y, splitter, config.RF_PARAMS, verbose=False)

    price_data = load_all(config.TICKERS, config.START_DATE, config.END_DATE)
    weights_df = compute_weights(
        signals_df, price_data,
        mode=config.OPTIMIZER,
        lookback=config.COV_LOOKBACK,
        max_weight=config.MAX_WEIGHT,
    )
    result = run_backtest(
        price_data, weights_df,
        transaction_cost=config.TRANSACTION_COST,
        initial_capital=args.capital,
        rebalance_freq=config.REBALANCE_FREQ,
        vol_target=config.VOL_TARGET,
        vol_lookback=config.VOL_LOOKBACK,
    )

    net_returns = result["net_return"].values
    actual_pv   = result["portfolio_value"]

    # Actual metrics
    actual_metrics = {
        "cagr":        cagr(actual_pv),
        "sharpe":      sharpe(result["net_return"], rf=0.045),
        "max_dd":      max_drawdown(actual_pv),
        "final_value": actual_pv.iloc[-1],
    }

    # ── Run simulations ───────────────────────────────────────────────────────
    print(f"\nRunning {args.sims:,} Monte Carlo simulations...")
    paths = run_simulations(net_returns, args.sims, args.capital, seed=args.seed)
    metrics_df = compute_sim_metrics(paths)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(metrics_df, actual_metrics, args.sims)

    # ── Save plots ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving plots...")
    if args.live:
        print("  Live animation mode — watch the window...")
        fig1 = plot_paths_live(paths, actual_pv, args.capital, args.sims, batch=args.batch)
        print("  Animation complete. Close the window to continue.")
        plt.show(block=True)
    else:
        fig1 = plot_paths(paths, actual_pv, args.capital, args.sims)

    path1 = OUTPUT_DIR / "mc_paths.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {path1}")

    fig2 = plot_distributions(metrics_df, actual_metrics)
    path2 = OUTPUT_DIR / "mc_distributions.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
