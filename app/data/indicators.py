import pandas as pd
import pandas_ta as ta


def calc_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Simple moving average of the Close price."""
    return ta.sma(data["Close"], length=period)


def calc_ema(data: pd.DataFrame, period: int) -> pd.Series:
    """Exponential moving average of the Close price."""
    return ta.ema(data["Close"], length=period)


def calc_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index of the Close price."""
    return ta.rsi(data["Close"], length=period)


def calc_macd(data: pd.DataFrame) -> pd.Series:
    """MACD line (12/26/9). Returns the MACD line only (not signal or histogram)."""
    result = ta.macd(data["Close"])
    return result.iloc[:, 0]


def calc_macd_signal(data: pd.DataFrame) -> pd.Series:
    """MACD signal line (9-period EMA of the MACD line). Use with calc_macd() for crossover."""
    result = ta.macd(data["Close"])
    return result.iloc[:, 2]


def calc_bbands(
    data: pd.DataFrame, period: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands of the Close price. Returns (upper, middle, lower)."""
    result = ta.bbands(data["Close"], length=period, std=std)
    # Column order from pandas-ta: BBL, BBM, BBU, BBB, BBP
    lower = result.iloc[:, 0]
    middle = result.iloc[:, 1]
    upper = result.iloc[:, 2]
    return upper, middle, lower


def calc_roc(data: pd.DataFrame, period: int = 10) -> pd.Series:
    """Rate of Change: percentage price change over `period` bars."""
    return ta.roc(data["Close"], length=period)


def calc_obv(data: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume weighted by price direction."""
    return ta.obv(data["Close"], data["Volume"])


def calc_sma_of(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average of any Series (e.g. OBV)."""
    return ta.sma(series, length=period)
