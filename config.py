TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
BENCHMARK = "SPY"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Label generation
FORWARD_HORIZON = 5     # trading days ahead to measure return
BUY_THRESHOLD = 0.015   # >+1.5% forward return → Buy
SELL_THRESHOLD = -0.015 # <-1.5% forward return → Sell
LABEL_GAP = 5           # days gap between train end and test start (equals horizon)

# Walk-forward CV
MIN_TRAIN_DAYS = 504    # ~2 years minimum training window
TEST_DAYS = 63          # ~1 quarter per test fold
STEP_DAYS = 63          # advance by 1 quarter each fold

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 25,
    "max_features": "log2",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

# Backtesting
INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001  # 0.1% per trade (one-way)
REBALANCE_FREQ = 5        # only rebalance every N trading days (1 = daily)
VOL_TARGET = 0.15         # target annualized portfolio volatility (15%)
VOL_LOOKBACK = 20         # days of returns used to estimate realized vol

# Regime filter: only trade when SPY is above its N-day moving average
# - not too much success with this
REGIME_FILTER = False      # set False to disable
REGIME_MA = 200           # 200-day MA is the standard institutional filter

# Portfolio optimizer: "mean_variance" or "risk_parity"
OPTIMIZER = "risk_parity"
MAX_WEIGHT = 0.35         # max allocation to any single stock
COV_LOOKBACK = 40         # days of returns for covariance estimate
SIGNAL_STRENGTH_SCALING = False  # scale risk-parity weights by RF confidence score
