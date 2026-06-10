import pandas as pd

from app.data.indicators import calc_bbands
from app.strategies.base import Strategy


class BollingerReversion(Strategy):
    """Buy when price touches the lower band (oversold); sell when it returns to the middle.

    Mean-reversion thesis: a close more than 2 standard deviations below the
    20-day average is statistically stretched and likely to snap back to the mean.
    """

    def __init__(self, period: int = 20, std: float = 2.0) -> None:
        super().__init__(name="BollingerReversion")
        self.period = period
        self.std = std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        upper, middle, lower = calc_bbands(data, self.period, self.std)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[data["Close"] < lower] = 1
        signals[data["Close"] > middle] = -1

        return signals.fillna(0).astype(int)
