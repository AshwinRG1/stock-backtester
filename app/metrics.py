"""
app/metrics.py — Performance Metrics
======================================
PURPOSE
-------
This file takes a completed BacktestResult and computes/displays performance
statistics.  It answers the core question: "Was this strategy actually good?"

Raw returns alone are misleading — a strategy that returned 20% but lost 60%
at its worst point is very different from one that returned 20% with smooth,
steady gains.  Metrics give you the full picture.

METRICS EXPLAINED
-----------------
Total Return
    The simplest measure: how much did $1 grow (or shrink) over the period?
    Formula: (final_value - initial_value) / initial_value
    Limitation: ignores how bumpy the ride was.

Max Drawdown
    The largest peak-to-trough decline in portfolio value, expressed as a %.
    If the equity curve hit $12,000 then fell to $9,000, drawdown = -25%.
    This is the most important risk metric — it tells you the worst loss you
    would have experienced if you entered at the peak.

Sharpe Ratio
    Risk-adjusted return: mean daily return divided by std of daily return,
    annualised by multiplying by sqrt(252 trading days).
    > 1.0  acceptable
    > 2.0  good
    > 3.0  excellent
    Computed in the engine and passed through here for display.

Win Rate
    Percentage of completed trades that were profitable (pnl > 0).
    Requires at least one completed round-trip trade to be meaningful.

Average Trade Return
    Mean return % across all completed trades.  Useful paired with win rate —
    a strategy can have a low win rate but still be profitable if winners are
    much larger than losers.

HOW IT FITS IN
--------------
    run_engine()       →  BacktestResult
    metrics.summary()  →  printed report to terminal
    (future) metrics.summary() can also return a dict for plotting/export
"""

import numpy as np
import pandas as pd

from app.engine import BacktestResult


def _max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the maximum drawdown of an equity curve.

    At each bar we calculate how far the portfolio has fallen from its
    all-time high up to that point.  The worst such drop is the max drawdown.

    Returns a negative float, e.g. -0.25 means a 25% drawdown.
    """
    rolling_peak = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_peak) / rolling_peak
    return float(drawdowns.min())


def _win_rate(trades: pd.DataFrame) -> float | None:
    """Return the fraction of trades with positive pnl, or None if no trades."""
    if trades.empty:
        return None
    return float((trades["pnl"] > 0).sum() / len(trades))


def _avg_trade_return(trades: pd.DataFrame) -> float | None:
    """Return mean return_pct across all trades, or None if no trades."""
    if trades.empty:
        return None
    return float(trades["return_pct"].mean())


def summary(result: BacktestResult, initial_capital: float = 10_000.0) -> dict:
    """
    Print a formatted performance report and return metrics as a dict.

    Parameters
    ----------
    result : BacktestResult
        Output of engine.run_engine().

    initial_capital : float
        Starting capital used in the backtest — needed to display dollar P&L.

    Returns
    -------
    dict
        All computed metrics keyed by name, for downstream use (plotting, etc).
    """
    mdd = _max_drawdown(result.equity_curve)
    win_rate = _win_rate(result.trades)
    avg_trade_ret = _avg_trade_return(result.trades)
    final_value = result.equity_curve.iloc[-1]
    dollar_pnl = final_value - initial_capital
    num_trades = len(result.trades)

    # --- build report --------------------------------------------------------
    ticker_line = f"  Ticker          : {result.ticker}" if result.ticker else ""

    win_rate_line = (
        f"  Win Rate        : {win_rate:.1%}  ({int(win_rate * num_trades)}/{num_trades} trades)"
        if win_rate is not None
        else "  Win Rate        : n/a  (no completed trades)"
    )
    avg_ret_line = (
        f"  Avg Trade Ret   : {avg_trade_ret:+.2f}%"
        if avg_trade_ret is not None
        else "  Avg Trade Ret   : n/a"
    )

    report = f"""
╔══════════════════════════════════════════╗
║         BACKTEST PERFORMANCE SUMMARY     ║
╚══════════════════════════════════════════╝
{ticker_line}
  Initial Capital : ${initial_capital:>10,.2f}
  Final Value     : ${final_value:>10,.2f}
  Dollar P&L      : ${dollar_pnl:>+10,.2f}

  Total Return    : {result.total_return:>+.2%}
  Max Drawdown    : {mdd:>.2%}
  Sharpe Ratio    : {result.sharpe_ratio:.4f}

  Num Trades      : {num_trades}
{win_rate_line}
{avg_ret_line}
══════════════════════════════════════════
"""
    print(report)

    if not result.trades.empty:
        print("  Trade Log:")
        print(result.trades.to_string(index=False))
        print()

    return {
        "total_return":    result.total_return,
        "max_drawdown":    round(mdd, 4),
        "sharpe_ratio":    result.sharpe_ratio,
        "final_value":     round(final_value, 2),
        "dollar_pnl":      round(dollar_pnl, 2),
        "num_trades":      num_trades,
        "win_rate":        round(win_rate, 4) if win_rate is not None else None,
        "avg_trade_return": round(avg_trade_ret, 4) if avg_trade_ret is not None else None,
    }
