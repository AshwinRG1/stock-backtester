import pandas as pd

from app.engine import run_engine
from app.strategies.base import Strategy
from app.strategies.sma_crossover import SMACrossover

INITIAL_CAPITAL = 10_000.0


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


class FixedSignals(Strategy):
    """Test helper: returns a pre-defined signal list, bypassing indicator logic."""

    def __init__(self, signals: list[int]) -> None:
        super().__init__(name="Fixed")
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals, index=data.index, dtype=int)


def test_equity_curve_starts_at_initial_capital():
    """Equity curve first value should equal initial_capital."""
    closes = [10.0] * 20 + [10.0 + i for i in range(40)]
    data = make_ohlcv(closes)
    strategy = SMACrossover(fast_period=5, slow_period=10)
    result = run_engine(data, strategy, initial_capital=INITIAL_CAPITAL)
    assert result.equity_curve.iloc[0] == INITIAL_CAPITAL, (
        f"Expected {INITIAL_CAPITAL}, got {result.equity_curve.iloc[0]}"
    )


def test_equity_curve_length_matches_input():
    """Equity curve should have the same length as the input data."""
    closes = [10.0] * 20 + [10.0 + i for i in range(40)]
    data = make_ohlcv(closes)
    strategy = SMACrossover(fast_period=5, slow_period=10)
    result = run_engine(data, strategy, initial_capital=INITIAL_CAPITAL)
    assert len(result.equity_curve) == len(data), (
        f"Expected length {len(data)}, got {len(result.equity_curve)}"
    )


def test_no_signals_produces_no_trades():
    """All-zero signals should result in an empty trade log."""
    data = make_ohlcv([10.0] * 20)
    result = run_engine(data, FixedSignals([0] * 20), INITIAL_CAPITAL)
    assert result.trades.empty


def test_trade_log_has_correct_columns():
    """Trade DataFrame must have the expected columns even when empty."""
    data = make_ohlcv([10.0] * 20)
    result = run_engine(data, FixedSignals([0] * 20), INITIAL_CAPITAL)
    expected = {"entry_date", "exit_date", "entry_price", "exit_price", "pnl", "return_pct"}
    assert set(result.trades.columns) == expected


def test_total_return_matches_equity_curve():
    """total_return must equal (final_value - initial) / initial, rounded to 4dp."""
    closes = [10.0] * 20 + [10.0 + i for i in range(40)]
    data = make_ohlcv(closes)
    result = run_engine(data, SMACrossover(fast_period=5, slow_period=10), INITIAL_CAPITAL)
    expected = round((result.equity_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL, 4)
    assert result.total_return == expected


def test_sharpe_zero_when_no_market_movement():
    """Zero volatility in returns should yield a Sharpe ratio of 0.0."""
    data = make_ohlcv([10.0] * 30)
    result = run_engine(data, FixedSignals([0] * 30), INITIAL_CAPITAL)
    assert result.sharpe_ratio == 0.0


def test_ticker_stored_in_result():
    """Ticker passed to run_engine should appear in BacktestResult."""
    data = make_ohlcv([10.0] * 20)
    result = run_engine(data, FixedSignals([0] * 20), INITIAL_CAPITAL, ticker="AAPL")
    assert result.ticker == "AAPL"


def test_known_trade_pnl_and_return():
    """Buy at 10.0, sell at 20.0 should produce pnl=10000 and return_pct=100."""
    # closes[5]=10.0 (buy signal), closes[15]=20.0 (sell signal)
    closes = [10.0] * 5 + [10.0 + i for i in range(15)]
    signals = [0] * 5 + [1] + [0] * 9 + [-1] + [0] * 4
    result = run_engine(make_ohlcv(closes), FixedSignals(signals), INITIAL_CAPITAL)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 10.0
    assert trade["exit_price"] == 20.0
    assert trade["pnl"] == round((20.0 - 10.0) / 10.0 * INITIAL_CAPITAL, 2)
    assert trade["return_pct"] == 100.0


def test_open_trade_not_in_log():
    """A buy with no subsequent sell should not appear in the trade log."""
    signals = [0] * 5 + [1] + [0] * 14
    result = run_engine(make_ohlcv([10.0] * 20), FixedSignals(signals), INITIAL_CAPITAL)
    assert result.trades.empty
