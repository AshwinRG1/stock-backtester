"""
app/strategies/base.py — Strategy Interface (Abstract Base Class)
=================================================================
PURPOSE
-------
This file defines the contract that every trading strategy in the project must
follow.  It does not contain any trading logic itself — it only says *what
shape* a strategy must have so that the rest of the system (the backtesting
engine, the CLI, etc.) can call any strategy in exactly the same way.

WHAT IS AN ABSTRACT BASE CLASS (ABC)?
--------------------------------------
An ABC is a class you cannot instantiate directly.  It acts as a blueprint.
If you write `Strategy(name="x")` Python will raise a TypeError — you must
first subclass it and implement every @abstractmethod.

This is useful here because:
  - It forces every strategy author to implement generate_signals().
  - The engine can accept any Strategy subclass without knowing which one it is.
  - If a method is missing, Python raises an error at import time (fast
    feedback) rather than at runtime deep in a backtest.

THE SIGNAL CONTRACT
-------------------
generate_signals() must return a pd.Series of integers where each value is:
    +1  — Buy:  enter or stay in a long position
     0  — Hold: do nothing
    -1  — Sell: exit the position

The series must be the same length as the input DataFrame and share its index.
NaN values are not allowed — use 0 for bars where the indicator is still
warming up (e.g. the first N bars of an N-period SMA are undefined).

HOW IT FITS IN
--------------
    base.Strategy  ←  subclassed by  →  sma_crossover.SMACrossover
                                     →  (future) rsi_reversion.RSIReversion
                                     →  (future) macd_momentum.MACDMomentum

    (future) engine.py calls strategy.generate_signals(data) without caring
    which concrete subclass it received.  This pattern is called polymorphism.
"""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Contract:
    - Subclasses must implement generate_signals().
    - generate_signals() receives a DataFrame with OHLCV columns and a DatetimeIndex.
    - It must return a Series of the same length as the input, indexed identically.
    - Valid signal values are integers: +1 (buy), -1 (sell), 0 (hold).
    - NaN values are not allowed in the returned Series; use 0 for warmup periods.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Compute trading signals for the given OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex. Expected columns include at
            minimum 'Close'. Additional columns (Open, High, Low, Volume) may
            be present and used by the strategy.

        Returns
        -------
        pd.Series
            Integer series aligned to ``data.index`` where each value is one of:
            +1  — enter long (buy)
             0  — stay flat (hold / no position change)
            -1  — exit long (sell)

        Raises
        ------
        ValueError
            If the returned Series contains values outside {-1, 0, 1} or has a
            different length / index than ``data``.
        """


# ---------------------------------------------------------------------------
# HOW TO ADD A NEW STRATEGY (e.g. rsi_reversion.py)
# ---------------------------------------------------------------------------
#
# 1. CREATE the file app/strategies/rsi_reversion.py
#
# 2. IMPORT the base class and any indicators you need:
#
#       from app.strategies.base import Strategy
#       from app.data.indicators import calc_rsi
#       import pandas as pd
#
# 3. SUBCLASS Strategy and implement generate_signals():
#
#       class RSIReversion(Strategy):
#           def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
#               super().__init__(name="RSIReversion")
#               self.period = period
#               self.oversold = oversold
#               self.overbought = overbought
#
#           def generate_signals(self, data: pd.DataFrame) -> pd.Series:
#               rsi = calc_rsi(data, self.period)
#               signals = pd.Series(0, index=data.index)
#               signals[rsi < self.oversold] = 1    # buy when oversold
#               signals[rsi > self.overbought] = -1  # sell when overbought
#               return signals.fillna(0).astype(int)
#
# 4. REGISTER (optional) by importing in app/strategies/__init__.py so callers
#    can do `from app.strategies import RSIReversion` without knowing the module:
#
#       from app.strategies.rsi_reversion import RSIReversion
#
# 5. ADD TESTS in tests/test_rsi_reversion.py following the same three-test
#    pattern used in tests/test_sma_crossover.py.
#
# No changes to base.py are required.
# ---------------------------------------------------------------------------