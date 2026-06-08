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

        # Suppress the warmup boundary: the first slow_period bars contain NaN
        # SMAs which pandas coerces to False (→ 0). The first valid comparison
        # at bar slow_period-1 would diff() against that 0, producing a spurious
        # crossover signal even when no real crossover occurred.
        signals.iloc[: self.slow_period] = 0

        return signals
