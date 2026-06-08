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

    # Flatten MultiIndex columns before selection — newer yfinance versions return
    # a MultiIndex even for a single ticker, so string-key selection would KeyError
    # if we try to select columns first.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    return df
