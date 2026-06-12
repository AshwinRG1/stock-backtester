from pathlib import Path

import app.engine as engine
import app.data.fetcher as fetcher
import app.metrics as metrics
from app.strategies.sma_crossover import SMACrossover
from app.visualizer import plot_results


TICKER = "^NDX"  # Example ticker, can be changed to any stock symbol
PERIOD = "40y"     # Lookback window for historical data (e.g. '1y', '6mo', '3mo')
INTERVAL = "1d"   # Bar interval (e.g. '1d',
INITIAL_CAPITAL = 10_000.0  # Starting capital for backtesting
REPORT_PATH = Path(__file__).parent / "report.html"


data = fetcher.fetch_ohlcv(TICKER, period=PERIOD, interval=INTERVAL)
strategy = SMACrossover(fast_period=50, slow_period=200)
result = engine.run_engine(data, strategy, initial_capital=INITIAL_CAPITAL, ticker=TICKER)
metrics.summary(result, initial_capital=INITIAL_CAPITAL)

html = plot_results(result, data, initial_capital=INITIAL_CAPITAL, strategy_name="SMACrossover(50, 200)")
REPORT_PATH.write_text(html, encoding="utf-8")
print(f"Report saved → {REPORT_PATH.resolve()}")
