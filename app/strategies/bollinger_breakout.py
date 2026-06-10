import pandas as pd

from app.data.indicators import calc_bbands
from app.strategies.base import Strategy


class BollingerBreakout(Strategy):
    """Buy when price breaks above the upper Bollinger Band; sell when it falls back below the middle.

    A close above the upper band signals strong upward momentum. Exiting at the
    middle band (20-day average) captures the move before a full reversal.
    """

    def __init__(self, period: int = 20, std: float = 2.0) -> None:
        super().__init__(name="BollingerBreakout")
        self.period = period
        self.std = std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        upper, middle, _ = calc_bbands(data, self.period, self.std)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[data["Close"] > upper] = 1
        signals[data["Close"] < middle] = -1

        return signals.fillna(0).astype(int)
