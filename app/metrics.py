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


def _cagr(equity_curve: pd.Series, initial_capital: float) -> float:
    """
    Compound Annual Growth Rate over the full backtest period.

    Derives the number of years from the DatetimeIndex span of the equity
    curve, then solves: final = initial * (1 + cagr) ^ years  for cagr.
    """
    final_value = equity_curve.iloc[-1]
    num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if num_years <= 0:
        return 0.0
    return float((final_value / initial_capital) ** (1 / num_years) - 1)


def compute_metrics(result: BacktestResult, initial_capital: float = 10_000.0) -> dict:
    """Compute all performance metrics — side-effect free.

    Returns the dict shape used by both summary() (CLI) and the
    Phase 4 API route. Visualizer.py and any future caller that
    wants metrics without stdout noise should call this directly.
    """
    mdd           = _max_drawdown(result.equity_curve)
    win_rate      = _win_rate(result.trades)
    avg_trade_ret = _avg_trade_return(result.trades)
    cagr          = _cagr(result.equity_curve, initial_capital)
    final_value   = float(result.equity_curve.iloc[-1])
    dollar_pnl    = final_value - initial_capital
    num_trades    = len(result.trades)

    return {
        "total_return":     result.total_return,
        "cagr":             round(cagr, 4),
        "max_drawdown":     round(mdd, 4),
        "sharpe_ratio":     result.sharpe_ratio,
        "final_value":      round(final_value, 2),
        "dollar_pnl":       round(dollar_pnl, 2),
        "num_trades":       num_trades,
        "win_rate":         round(win_rate, 4) if win_rate is not None else None,
        "avg_trade_return": round(avg_trade_ret, 4) if avg_trade_ret is not None else None,
    }


def summary(result: BacktestResult, initial_capital: float = 10_000.0) -> dict:
    """Print a formatted performance report and return the metrics dict.

    Thin wrapper over compute_metrics() that adds CLI presentation. Use
    compute_metrics() directly from any non-CLI caller (API, visualizer)
    to avoid stdout noise.
    """
    m = compute_metrics(result, initial_capital)

    ticker_line = f"  Ticker          : {result.ticker}" if result.ticker else ""

    # round() not int(): m['win_rate'] is rounded to 4 dp in the dict, so
    # `int(win_rate * num_trades)` truncates 16.9992 → 16 instead of 17.
    win_rate_line = (
        f"  Win Rate        : {m['win_rate']:.1%}  ({round(m['win_rate'] * m['num_trades'])}/{m['num_trades']} trades)"
        if m["win_rate"] is not None
        else "  Win Rate        : n/a  (no completed trades)"
    )
    avg_ret_line = (
        f"  Avg Trade Ret   : {m['avg_trade_return']:+.2f}%"
        if m["avg_trade_return"] is not None
        else "  Avg Trade Ret   : n/a"
    )

    report = f"""
╔══════════════════════════════════════════╗
║         BACKTEST PERFORMANCE SUMMARY     ║
╚══════════════════════════════════════════╝
{ticker_line}
  Initial Capital : ${initial_capital:>10,.2f}
  Final Value     : ${m['final_value']:>10,.2f}
  Dollar P&L      : ${m['dollar_pnl']:>+10,.2f}

  Total Return    : {m['total_return']:>+.2%}
  Avg Annual Ret  : {m['cagr']:>+.2%}  (CAGR)
  Max Drawdown    : {m['max_drawdown']:>.2%}
  Sharpe Ratio    : {m['sharpe_ratio']:.4f}

  Num Trades      : {m['num_trades']}
{win_rate_line}
{avg_ret_line}
══════════════════════════════════════════
"""
    print(report)

    if not result.trades.empty:
        print("  Trade Log:")
        print(result.trades.to_string(index=False))
        print()

    return m
