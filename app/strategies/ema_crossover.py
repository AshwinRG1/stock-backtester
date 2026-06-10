import pandas as pd

from app.data.indicators import calc_ema
from app.strategies.base import Strategy


class EMACrossover(Strategy):
    """Buy when the fast EMA crosses above the slow EMA; sell on the reverse cross.

    Identical logic to SMACrossover but uses exponential weighting, so the
    fast line reacts more strongly to recent price moves.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30) -> None:
        super().__init__(name="EMACrossover")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast = calc_ema(data, self.fast_period)
        slow = calc_ema(data, self.slow_period)

        above = (fast > slow).astype(int)
        signals = above.diff().fillna(0).astype(int)

        # Suppress warmup: same boundary issue as SMACrossover — the first
        # slow_period bars have NaN EMAs which produce a spurious diff() signal.
        signals.iloc[: self.slow_period] = 0

        return signals
