import pandas as pd
import pandas_ta as ta


def calc_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Simple moving average of the Close price."""
    return ta.sma(data["Close"], length=period)


def calc_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index of the Close price."""
    return ta.rsi(data["Close"], length=period)


def calc_macd(data: pd.DataFrame) -> pd.Series:
    """MACD line (12/26/9). Returns the MACD line only (not signal or histogram)."""
    result = ta.macd(data["Close"])
    return result.iloc[:, 0]
