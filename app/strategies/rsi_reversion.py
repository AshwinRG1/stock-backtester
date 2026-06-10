import pandas as pd

from app.data.indicators import calc_rsi
from app.strategies.base import Strategy


class RSIReversion(Strategy):
    """Buy when RSI is oversold; sell when overbought.

    oversold  (default 30) — RSI below this → +1 (buy)
    overbought (default 70) — RSI above this → -1 (sell)
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70) -> None:
        super().__init__(name="RSIReversion")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi = calc_rsi(data, self.period)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[rsi < self.oversold] = 1
        signals[rsi > self.overbought] = -1

        # NaN comparisons return False, so warmup bars already stay 0.
        # fillna(0) is a safety net; astype(int) enforces the contract.
        return signals.fillna(0).astype(int)
