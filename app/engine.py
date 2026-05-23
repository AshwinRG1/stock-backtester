from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    Holds all outputs produced by a single backtest run.

    equity_curve : pd.Series
        Portfolio value in dollars at the close of each bar.
        Starts at initial_capital, rises and falls with the strategy.

    trades : pd.DataFrame
        One row per completed round-trip trade with columns:
            entry_date  — bar where we bought
            exit_date   — bar where we sold
            entry_price — Close price on entry
            exit_price  — Close price on exit
            pnl         — profit/loss in dollars for that trade
            return_pct  — pnl as a percentage of the entry value

    total_return : float
        Percentage gain/loss over the full backtest period.
        e.g. 0.25 means +25%.

    sharpe_ratio : float
        Annualised risk-adjusted return. Ratio of mean daily return to
        std of daily return, scaled by sqrt(252) trading days.
        > 1.0 is generally considered acceptable.
    """
    equity_curve: pd.Series
    trades: pd.DataFrame
    total_return: float
    sharpe_ratio: float
    ticker: str = ""


def run_engine(data: pd.DataFrame, strategy, initial_capital: float = 10_000.0, ticker: str = "") -> BacktestResult:
    """
    Simulate a trading strategy against historical OHLCV data.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame returned by fetch_ohlcv(). Must have a DatetimeIndex
        and at minimum a 'Close' column.

    strategy : Strategy
        Any subclass of app.strategies.base.Strategy. generate_signals() is
        called here so the caller only needs to pass the strategy object.

    initial_capital : float
        Starting portfolio value in dollars (default $10,000).

    Returns
    -------
    BacktestResult
        Populated result object. Pass this to metrics.summary() (Stage 3).
    """

    # --- signals & positions ------------------------------------------------
    signals = strategy.generate_signals(data)

    # Convert +1/-1/0 signals into a binary held/flat position series.
    # replace(-1, 0) makes it binary; ffill() holds the position between
    # signals; fillna(0) covers the warmup period before the first signal.
    positions = signals.replace(-1, 0).ffill().fillna(0)

    # --- daily returns -------------------------------------------------------
    # pct_change() gives the market's return each bar.
    # shift(1) aligns yesterday's position to today's return — prevents
    # look-ahead bias (you act on today's signal at tomorrow's open).
    daily_market_returns = data["Close"].pct_change()
    strategy_returns = positions.shift(1) * daily_market_returns

    # --- equity curve --------------------------------------------------------
    # Compound daily returns starting from initial_capital.
    equity_curve = initial_capital * (1 + strategy_returns.fillna(0)).cumprod()

    # --- trade log -----------------------------------------------------------
    # A trade starts on a +1 signal and ends on a -1 signal.
    trades = []
    entry_date = None
    entry_price = None

    for date, signal in signals.items():
        if signal == 1 and entry_date is None:
            entry_date = date
            entry_price = data.loc[date, "Close"]
        elif signal == -1 and entry_date is not None:
            exit_price = data.loc[date, "Close"]
            pnl = (exit_price - entry_price) / entry_price * initial_capital
            trades.append({
                "entry_date":  entry_date,
                "exit_date":   date,
                "entry_price": entry_price,
                "exit_price":  exit_price,
                "pnl":         round(pnl, 2),
                "return_pct":  round((exit_price - entry_price) / entry_price * 100, 2),
            })
            entry_date = None
            entry_price = None

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "entry_price", "exit_price", "pnl", "return_pct"]
    )

    # --- summary metrics -----------------------------------------------------
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital

    # Annualised Sharpe ratio (assumes 252 trading days, risk-free rate = 0)
    clean = strategy_returns.dropna()
    if clean.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((clean.mean() / clean.std()) * np.sqrt(252))

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades_df,
        total_return=round(total_return, 4),
        sharpe_ratio=round(sharpe_ratio, 4),
        ticker=ticker,
    )
