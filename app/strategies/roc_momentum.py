import pandas as pd

from app.data.indicators import calc_roc
from app.strategies.base import Strategy


class ROCMomentum(Strategy):
    """Buy on strong positive momentum; sell on equal negative momentum.

    Rate of Change measures the percentage gain or loss over `period` bars.
    Only moves exceeding `threshold` percent generate a signal, filtering out noise.
    """

    def __init__(self, period: int = 10, threshold: float = 2.0) -> None:
        super().__init__(name="ROCMomentum")
        self.period = period
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        roc = calc_roc(data, self.period)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[roc > self.threshold] = 1
        signals[roc < -self.threshold] = -1

        return signals.fillna(0).astype(int)
