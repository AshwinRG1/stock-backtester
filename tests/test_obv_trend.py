import pandas as pd

from app.strategies.obv_trend import OBVTrend


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1} with no NaN values."""
    prices = [float(i) for i in range(1, 61)]
    data = make_ohlcv(prices)
    strategy = OBVTrend()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [float(i) for i in range(1, 61)]
    data = make_ohlcv(prices)
    strategy = OBVTrend()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_when_obv_rises_above_its_sma():
    """Flat prices keep OBV flat; rising prices then push OBV above its lagging SMA.

    During the flat period OBV stays at 0 (no price direction change, no volume
    added). Once prices rise every bar, OBV accumulates volume and quickly
    overtakes the SMA that is still averaging in the flat-OBV period.
    """
    flat = [10.0] * 25
    rising = [10.0 + i for i in range(25)]
    data = make_ohlcv(flat + rising)
    strategy = OBVTrend(sma_period=20)
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) as OBV crosses above its SMA"
