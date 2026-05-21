import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a ticker via yfinance.

    Parameters
    ----------
    ticker : str
        The stock symbol (e.g. 'AAPL').
    period : str
        Lookback window accepted by yfinance (e.g. '1y', '6mo', '3mo').
    interval : str
        Bar interval accepted by yfinance (e.g. '1d', '1h', '15m').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex (timezone-naive UTC). Raises ValueError if no data
        is returned.
    """
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' (period={period}, interval={interval})")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Flatten MultiIndex columns that yfinance may produce when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    return df
