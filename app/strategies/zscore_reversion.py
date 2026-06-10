import pandas as pd

from app.strategies.base import Strategy


class ZScoreReversion(Strategy):
    """Buy when price is statistically cheap; sell when it is statistically expensive.

    The z-score measures how many standard deviations the current Close is
    above or below its rolling mean. A z-score below -threshold means the
    price has fallen unusually far and is expected to revert toward the mean.

    Uses pure pandas rolling math — no external indicator library needed.
    """

    def __init__(self, period: int = 20, threshold: float = 2.0) -> None:
        super().__init__(name="ZScoreReversion")
        self.period = period
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        rolling_mean = close.rolling(self.period).mean()
        rolling_std = close.rolling(self.period).std()

        # std is 0 when all prices in the window are identical → NaN z-score,
        # which fillna(0) below converts to a hold signal.
        zscore = (close - rolling_mean) / rolling_std

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[zscore < -self.threshold] = 1
        signals[zscore > self.threshold] = -1

        return signals.fillna(0).astype(int)
