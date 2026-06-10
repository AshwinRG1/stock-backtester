import pandas as pd

from app.data.indicators import calc_macd, calc_macd_signal
from app.strategies.base import Strategy


class MACDCrossover(Strategy):
    """Buy when the MACD line crosses above the signal line; sell on the reverse cross.

    Default parameters follow the standard 12/26/9 configuration.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        super().__init__(name="MACDCrossover")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        macd_line = calc_macd(data)
        signal_line = calc_macd_signal(data)

        above = (macd_line > signal_line).astype(int)
        signals = above.diff().fillna(0).astype(int)

        # Warmup: slow EMA needs `slow` bars, then the signal EMA needs `signal`
        # more bars on top of that before the first valid crossover can be detected.
        signals.iloc[: self.slow + self.signal - 1] = 0

        return signals
