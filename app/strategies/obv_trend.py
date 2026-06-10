import pandas as pd

from app.data.indicators import calc_obv, calc_sma_of
from app.strategies.base import Strategy


class OBVTrend(Strategy):
    """Buy when OBV crosses above its SMA; sell when it crosses below.

    On-Balance Volume tracks whether volume is flowing into or out of a stock.
    When OBV rises above its own moving average, buyers are becoming dominant.
    """

    def __init__(self, sma_period: int = 20) -> None:
        super().__init__(name="OBVTrend")
        self.sma_period = sma_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        obv = calc_obv(data)
        obv_sma = calc_sma_of(obv, self.sma_period)

        above = (obv > obv_sma).astype(int)
        signals = above.diff().fillna(0).astype(int)

        signals.iloc[: self.sma_period] = 0

        return signals
