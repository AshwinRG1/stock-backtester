import pandas as pd
import pytest

from app.engine import BacktestResult
from app.metrics import _avg_trade_return, _cagr, _max_drawdown, _win_rate, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_equity_curve(values: list[float], start: str = "2020-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def make_trades(pnls: list[float], return_pcts: list[float]) -> pd.DataFrame:
    n = len(pnls)
    return pd.DataFrame({
        "entry_date":  pd.date_range("2020-01-01", periods=n, freq="W"),
        "exit_date":   pd.date_range("2020-01-08", periods=n, freq="W"),
        "entry_price": [100.0] * n,
        "exit_price":  [100.0 + r for r in return_pcts],
        "pnl":         pnls,
        "return_pct":  return_pcts,
    })


def empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["entry_date", "exit_date", "entry_price", "exit_price", "pnl", "return_pct"]
    )


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_flat_curve_is_zero():
    """Flat equity curve never falls from its peak — drawdown must be 0."""
    curve = make_equity_curve([10_000.0] * 20)
    assert _max_drawdown(curve) == 0.0


def test_max_drawdown_always_rising_is_zero():
    """Monotonically rising curve has no drawdown."""
    curve = make_equity_curve([float(v) for v in range(10_000, 10_020)])
    assert _max_drawdown(curve) == 0.0


def test_max_drawdown_known_value():
    """Peak 10 000 → trough 7 500 should give exactly -0.25."""
    curve = make_equity_curve([10_000.0, 9_000.0, 7_500.0, 8_000.0, 10_000.0])
    assert _max_drawdown(curve) == pytest.approx(-0.25)


def test_max_drawdown_returns_negative_float():
    """Any real decline must produce a negative return value."""
    curve = make_equity_curve([10_000.0, 8_000.0, 9_000.0])
    assert _max_drawdown(curve) < 0.0


# ---------------------------------------------------------------------------
# _win_rate
# ---------------------------------------------------------------------------

def test_win_rate_empty_trades_is_none():
    """With no trades there is no win rate — must return None."""
    assert _win_rate(empty_trades()) is None


def test_win_rate_all_winners():
    trades = make_trades([500.0, 300.0, 100.0], [5.0, 3.0, 1.0])
    assert _win_rate(trades) == 1.0


def test_win_rate_all_losers():
    trades = make_trades([-100.0, -200.0], [-1.0, -2.0])
    assert _win_rate(trades) == 0.0


def test_win_rate_half_and_half():
    """2 winners out of 4 trades → exactly 0.5."""
    trades = make_trades([100.0, 200.0, -50.0, -75.0], [1.0, 2.0, -0.5, -0.75])
    assert _win_rate(trades) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _avg_trade_return
# ---------------------------------------------------------------------------

def test_avg_trade_return_empty_trades_is_none():
    assert _avg_trade_return(empty_trades()) is None


def test_avg_trade_return_known_value():
    """Mean of [10.0, 20.0, 30.0] return_pct must equal 20.0."""
    trades = make_trades([0.0, 0.0, 0.0], [10.0, 20.0, 30.0])
    assert _avg_trade_return(trades) == pytest.approx(20.0)


def test_avg_trade_return_mixed_signs():
    """Mean of [10.0, -10.0] must equal 0.0."""
    trades = make_trades([100.0, -100.0], [10.0, -10.0])
    assert _avg_trade_return(trades) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _cagr
# ---------------------------------------------------------------------------

def test_cagr_single_bar_returns_zero():
    """A one-bar curve has no time span — CAGR cannot be computed, returns 0."""
    curve = make_equity_curve([10_000.0])
    assert _cagr(curve, 10_000.0) == 0.0


def test_cagr_flat_curve_is_zero():
    """No growth over the period → CAGR is 0%."""
    curve = make_equity_curve([10_000.0] * 366)
    assert _cagr(curve, 10_000.0) == pytest.approx(0.0, abs=1e-6)


def test_cagr_double_in_one_year():
    """Doubling capital over ~365 days → CAGR ≈ 100%."""
    curve = make_equity_curve([10_000.0] + [20_000.0] * 364)
    assert _cagr(curve, 10_000.0) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _make_result(curve_values, trade_pnls=None, trade_rets=None) -> BacktestResult:
    curve = make_equity_curve(curve_values)
    trades = (
        make_trades(trade_pnls, trade_rets)
        if trade_pnls is not None
        else empty_trades()
    )
    total_return = round((curve.iloc[-1] - 10_000.0) / 10_000.0, 4)
    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        total_return=total_return,
        sharpe_ratio=1.0,
        ticker="TEST",
    )


def test_summary_returns_expected_keys():
    """summary() must return a dict that contains every required metric key."""
    result = _make_result(
        [10_000.0, 10_500.0, 10_200.0, 10_800.0],
        [500.0, -300.0],
        [5.0, -3.0],
    )
    metrics = summary(result, initial_capital=10_000.0)
    expected = {
        "total_return", "cagr", "max_drawdown", "sharpe_ratio",
        "final_value", "dollar_pnl", "num_trades", "win_rate", "avg_trade_return",
    }
    assert set(metrics.keys()) == expected


def test_summary_no_trades_nulls_trade_metrics():
    """win_rate and avg_trade_return must be None when the trade log is empty."""
    result = _make_result([10_000.0, 10_000.0, 10_000.0])
    metrics = summary(result, initial_capital=10_000.0)
    assert metrics["win_rate"] is None
    assert metrics["avg_trade_return"] is None


def test_summary_final_value_matches_equity_curve():
    """final_value must equal the last point of the equity curve."""
    result = _make_result([10_000.0, 10_500.0, 11_000.0])
    metrics = summary(result, initial_capital=10_000.0)
    assert metrics["final_value"] == pytest.approx(11_000.0)


def test_summary_num_trades_is_correct():
    """num_trades must reflect the actual number of rows in the trade log."""
    result = _make_result(
        [10_000.0, 10_200.0, 10_400.0],
        [200.0, 400.0],
        [2.0, 4.0],
    )
    metrics = summary(result, initial_capital=10_000.0)
    assert metrics["num_trades"] == 2


def test_summary_dollar_pnl_is_final_minus_initial():
    """dollar_pnl must equal final_value - initial_capital."""
    result = _make_result([10_000.0, 10_500.0, 12_000.0])
    metrics = summary(result, initial_capital=10_000.0)
    assert metrics["dollar_pnl"] == pytest.approx(metrics["final_value"] - 10_000.0)
