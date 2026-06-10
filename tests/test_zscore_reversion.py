import pandas as pd

from app.strategies.zscore_reversion import ZScoreReversion


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1} with no NaN values."""
    prices = [100.0 + i * 0.5 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = ZScoreReversion()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [100.0 + i * 0.5 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = ZScoreReversion()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_known_extreme_drop():
    """A sharp crash after a stable period pushes the z-score well below -2.

    25 flat bars give the rolling window a tight mean (100) and near-zero std.
    When price crashes to 50, the z-score collapses to around -4, guaranteeing
    a +1 buy signal.
    """
    flat = [100.0] * 25
    crash = [50.0] * 10
    data = make_ohlcv(flat + crash)
    strategy = ZScoreReversion(period=20, threshold=2.0)
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) when z-score drops below -2"
