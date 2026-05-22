"""
app/strategies/sma_crossover.py — SMA Crossover Strategy
=========================================================
PURPOSE
-------
This is the first concrete trading strategy in the project.  It implements the
classic "dual moving average crossover" rule, one of the oldest systematic
trading strategies in existence.

THE IDEA IN PLAIN ENGLISH
--------------------------
We calculate two Simple Moving Averages on the closing price:
  - A FAST SMA that averages the last N days (e.g. 10 days)
  - A SLOW SMA that averages the last M days (e.g. 30 days)

The fast SMA reacts quickly to recent price changes; the slow SMA is smoother
and lags behind.  When the fast SMA crosses above the slow SMA it suggests
that recent momentum is turning upward → we buy.  When it crosses below →
we sell.

WHY DOES THIS WORK (IN THEORY)?
--------------------------------
Moving average crossovers are trend-following signals.  The logic is:
  "If the short-term average is now above the long-term average, the asset is
   in an uptrend — ride it."
They work well in strongly trending markets but produce many false signals
(whipsaws) in sideways/choppy markets.  The backtest engine (Stage 2) will
let you measure exactly how well or poorly the strategy performs on real data.

HOW generate_signals() WORKS — STEP BY STEP
--------------------------------------------
Given a price series like: [10, 11, 12, 11, 10, 9, 10, 12, 14, 16]

1. calc_sma(data, fast_period) → Series of fast averages (fewer NaNs at start)
2. calc_sma(data, slow_period) → Series of slow averages (more NaNs at start)
3. (fast > slow)               → Boolean Series: True where fast is above slow
   .astype(int)                → Convert True/False to 1/0
4. .diff()                     → Difference from the previous bar:
       0 → 1  means the fast just crossed ABOVE the slow  →  +1 (buy)
       1 → 0  means the fast just crossed BELOW the slow  →  -1 (sell)
       no change                                          →   0 (hold)
5. .fillna(0)                  → First bar has no previous value; treat as hold.

PARAMETERS
----------
fast_period : int  — lookback window for the fast SMA (default 10 days)
slow_period : int  — lookback window for the slow SMA (default 30 days)

The larger the gap between fast and slow, the fewer signals you get but each
signal represents a more established trend.  This is a key thing to experiment
with during backtesting.

HOW IT FITS IN
--------------
    SMACrossover.generate_signals(data)
        ↓ calls
    calc_sma() from indicators.py  (twice)
        ↓ returns signals to
    (future) engine.run()  which converts signals → positions → P&L
"""

import pandas as pd

from app.data.indicators import calc_sma
from app.strategies.base import Strategy


class SMACrossover(Strategy):
    """Buy when the fast SMA crosses above the slow SMA; sell on the reverse cross."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30) -> None:
        super().__init__(name="SMACrossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast = calc_sma(data, self.fast_period)
        slow = calc_sma(data, self.slow_period)

        above = (fast > slow).astype(int)
        signals = above.diff().fillna(0).astype(int)
        # diff() == +1 means fast just crossed above slow → buy
        # diff() == -1 means fast just crossed below slow → sell

        return signals
