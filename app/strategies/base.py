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