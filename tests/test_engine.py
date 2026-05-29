import pandas as pd

from app.engine import run_engine
from app.strategies.sma_crossover import SMACrossover

INITIAL_CAPITAL = 10_000.0


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def make_signals(values: list[int]) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index)


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
    """Equity curve must have one value per bar in the input DataFrame."""
    closes = [10.0] * 20 + [10.0 + i for i in range(40)]
    data = make_ohlcv(closes)
    strategy = SMACrossover(fast_period=5, slow_period=10)
    result = run_engine(data, strategy, initial_capital=INITIAL_CAPITAL)
    assert len(result.equity_curve) == len(data)


def test_no_trades_when_signals_are_zero():
    """All-zero signals should produce no trades and a flat equity curve."""
    closes = [10.0] * 60
    data = make_ohlcv(closes)

    # Patch generate_signals to return all zeros
    class FlatStrategy:
        def generate_signals(self, d):
            return make_signals([0] * len(d))

    result = run_engine(data, FlatStrategy(), initial_capital=INITIAL_CAPITAL)
    assert result.trades.empty, "Expected no trades on all-zero signals"
    assert (result.equity_curve == INITIAL_CAPITAL).all(), (
        "Expected flat equity curve when never in market"
    )


def test_known_buy_sell_produces_one_trade():
    """A single +1 followed by a single -1 should log exactly one trade."""
    # 60 flat bars; inject buy at bar 20, sell at bar 40
    closes = [10.0] * 60
    data = make_ohlcv(closes)

    signals = [0] * 60
    signals[20] = 1
    signals[40] = -1

    class ManualStrategy:
        def generate_signals(self, d):
            return make_signals(signals)

    result = run_engine(data, ManualStrategy(), initial_capital=INITIAL_CAPITAL)
    assert len(result.trades) == 1, f"Expected 1 trade, got {len(result.trades)}"
    assert result.trades.iloc[0]["entry_price"] == 10.0
    assert result.trades.iloc[0]["exit_price"] == 10.0


def test_profitable_trade_increases_equity():
    """Buying low and selling high should leave final equity above initial_capital."""
    # Price rises from 10 to 20 while we're in the trade
    closes = [10.0] * 20 + [10.0 + i for i in range(1, 21)] + [30.0] * 20
    data = make_ohlcv(closes)

    signals = [0] * len(closes)
    signals[5] = 1   # buy when price is still 10
    signals[39] = -1  # sell after price has risen

    class ManualStrategy:
        def generate_signals(self, d):
            return make_signals(signals)

    result = run_engine(data, ManualStrategy(), initial_capital=INITIAL_CAPITAL)
    assert result.equity_curve.iloc[-1] > INITIAL_CAPITAL, (
        "Expected equity to grow after a profitable trade"
    )


def test_losing_trade_decreases_equity():
    """Buying high and selling low should leave final equity below initial_capital."""
    # Price falls from 20 to 10 while we're in the trade
    closes = [20.0] * 20 + [20.0 - i for i in range(1, 21)] + [10.0] * 20
    data = make_ohlcv(closes)

    signals = [0] * len(closes)
    signals[5] = 1    # buy when price is 20
    signals[39] = -1  # sell after price has fallen

    class ManualStrategy:
        def generate_signals(self, d):
            return make_signals(signals)

    result = run_engine(data, ManualStrategy(), initial_capital=INITIAL_CAPITAL)
    assert result.equity_curve.iloc[-1] < INITIAL_CAPITAL, (
        "Expected equity to fall after a losing trade"
    )


def test_no_position_before_first_buy_signal():
    """Equity curve should stay flat until the first buy signal fires."""
    closes = [10.0] * 30 + [20.0] * 30
    data = make_ohlcv(closes)

    signals = [0] * 60
    signals[30] = 1  # buy at bar 30

    class ManualStrategy:
        def generate_signals(self, d):
            return make_signals(signals)

    result = run_engine(data, ManualStrategy(), initial_capital=INITIAL_CAPITAL)
    # All bars before the buy signal should be flat (shift(1) means effect
    # is felt from bar 31 onward, so bars 0–30 should equal initial_capital)
    assert (result.equity_curve.iloc[:31] == INITIAL_CAPITAL).all(), (
        "Equity should not move before the first buy signal"
    )
