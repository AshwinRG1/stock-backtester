import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from app.engine import BacktestResult

_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Backtest Report{% if ticker %} — {{ ticker }}{% endif %}</title>
  <style>
    body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#333}
    h1{margin-bottom:4px}
    .sub{color:#888;font-size:0.95rem;margin:0 0 24px}
    h2{margin-top:32px}
    table{border-collapse:collapse;width:400px}
    th,td{padding:8px 16px;text-align:left;border-bottom:1px solid #e5e5e5;font-size:0.95rem}
    th{background:#f9f9f9;font-weight:600}
    .pos{color:#2e7d32}
    .neg{color:#c62828}
  </style>
</head>
<body>
  <h1>Backtest Report{% if ticker %} — {{ ticker }}{% endif %}</h1>
  {% if strategy_name %}<p class="sub">Strategy: {{ strategy_name }}</p>{% endif %}
  {{ chart_html }}
  <h2>Performance Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {% for row in rows %}
    <tr><td>{{ row.label }}</td><td class="{{ row.cls }}">{{ row.value }}</td></tr>
    {% endfor %}
  </table>
</body>
</html>
"""

# Maximum candles to render before switching to a coarser frequency.
_DAILY_LIMIT  = 365        # <= 1 year  → daily
_WEEKLY_LIMIT = 5 * 365   # 1–5 years  → weekly
                           # > 5 years  → monthly


def _resample_ohlcv(data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return a downsampled copy of data and the pandas frequency string used.

    Keeps chart density under ~500 candles regardless of the lookback window,
    which is the threshold where browser rendering stays smooth.
    """
    num_days = (data.index[-1] - data.index[0]).days
    if num_days <= _DAILY_LIMIT:
        return data.copy(), "D"
    freq = "W" if num_days <= _WEEKLY_LIMIT else "ME"
    resampled = data.resample(freq).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    return resampled, freq


def plot_results(
    result: BacktestResult,
    data: pd.DataFrame,
    initial_capital: float = 10_000.0,
    strategy_name: str = "",
) -> str:
    """Generate an interactive HTML backtest report.

    Parameters
    ----------
    result         : BacktestResult returned by run_engine()
    data           : the same OHLCV DataFrame passed to run_engine()
    initial_capital: starting capital used in the backtest
    strategy_name  : optional label shown in the report header

    Returns
    -------
    str
        Self-contained HTML string — write to disk or return from an API endpoint.
    """
    candle_data, freq = _resample_ohlcv(data)

    # Resample equity curve and B&H to match candle frequency so all panels
    # have the same x-axis density and zoom together cleanly.
    buy_and_hold = initial_capital * (data["Close"] / data["Close"].iloc[0])
    if freq != "D":
        equity_plot  = result.equity_curve.resample(freq).last().dropna()
        bah_plot     = buy_and_hold.resample(freq).last().dropna()
    else:
        equity_plot  = result.equity_curve
        bah_plot     = buy_and_hold

    # Drawdown for the chart — derived from the resampled equity curve.
    # True max drawdown for the metrics table is still computed from daily data below.
    rolling_peak_plot = equity_plot.cummax()
    drawdown_plot = ((equity_plot - rolling_peak_plot) / rolling_peak_plot * 100).round(2)

    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.30, 0.20],
        vertical_spacing=0.02,
        subplot_titles=("Price & Trades", "Equity Curve", "Drawdown"),
    )

    # Panel 1 — candlestick + entry/exit markers
    fig.add_trace(
        go.Candlestick(
            x=candle_data.index,
            open=candle_data["Open"], high=candle_data["High"],
            low=candle_data["Low"],   close=candle_data["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    if not result.trades.empty:
        fig.add_trace(
            go.Scatter(
                x=result.trades["entry_date"],
                y=result.trades["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#2e7d32"),
                name="Entry",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.trades["exit_date"],
                y=result.trades["exit_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#c62828"),
                name="Exit",
            ),
            row=1, col=1,
        )

    # Panel 2 — equity curve vs buy-and-hold
    fig.add_trace(
        go.Scatter(
            x=equity_plot.index, y=equity_plot.round(2),
            name="Strategy",
            line=dict(color="#1565c0", width=2),
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bah_plot.index, y=bah_plot.round(2),
            name="Buy & Hold",
            line=dict(color="#888", width=1.5, dash="dash"),
        ),
        row=2, col=1,
    )

    # Panel 3 — drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown_plot.index, y=drawdown_plot,
            fill="tozeroy",
            fillcolor="rgba(198,40,40,0.15)",
            line=dict(color="#c62828", width=1),
            name="Drawdown",
            showlegend=False,
        ),
        row=3, col=1,
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    # Default view: most recent 2 years. User can zoom/pan to full history.
    default_start = data.index[-1] - pd.DateOffset(years=2)

    range_buttons = [
        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
        dict(count=2,  label="2Y",  step="year",  stepmode="backward"),
        dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
        dict(step="all", label="All"),
    ]

    fig.update_layout(
        height=820,
        template="plotly_white",
        dragmode="pan",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
        modebar=dict(remove=["select2d", "lasso2d"]),
    )
    fig.update_yaxes(title_text="Price",        row=1, col=1)
    fig.update_yaxes(title_text="Value ($)",    row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    # Dynamic date labels: format changes automatically as the user zooms.
    # dtickrange boundaries are in milliseconds (for sub-month) or "M<n>" strings.
    fig.update_xaxes(
        rangeslider_visible=False,
        tickformatstops=[
            dict(dtickrange=[None, 604800000], value="%b %d, %Y"),   # < 1 week  → Jan 15, 2024
            dict(dtickrange=[604800000, "M1"], value="%b %d, %Y"),   # week–month → Jan 15, 2024
            dict(dtickrange=["M1", "M12"],     value="%b %Y"),       # month–year → Jan 2024
            dict(dtickrange=["M12", None],     value="%Y"),          # > 1 year   → 2024
        ],
    )
    if freq == "D":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # range + rangeselector go on row 1 only — it is the master axis when
    # shared_xaxes=True, so updating its range propagates to rows 2 and 3.
    fig.update_xaxes(
        range=[default_start, data.index[-1]],
        rangeselector=dict(buttons=range_buttons),
        row=1, col=1,
    )

    chart_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"scrollZoom": True, "displaylogo": False},
    )

    # ------------------------------------------------------------------
    # Metrics — always computed from full daily data, not the resampled chart data
    # ------------------------------------------------------------------
    num_trades = len(result.trades)
    win_rate   = float((result.trades["pnl"] > 0).mean()) if num_trades else None

    num_years  = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 365.25
    cagr       = (result.equity_curve.iloc[-1] / initial_capital) ** (1 / num_years) - 1 if num_years > 0 else 0.0

    rolling_peak_full = result.equity_curve.cummax()
    mdd = float(((result.equity_curve - rolling_peak_full) / rolling_peak_full).min())

    def _pct(v: float) -> str: return f"{v:+.2%}"
    def _cls(v: float) -> str: return "pos" if v >= 0 else "neg"

    rows = [
        {"label": "Total Return", "value": _pct(result.total_return), "cls": _cls(result.total_return)},
        {"label": "CAGR",         "value": _pct(cagr),                "cls": _cls(cagr)},
        {"label": "Sharpe Ratio", "value": f"{result.sharpe_ratio:.4f}", "cls": ""},
        {"label": "Max Drawdown", "value": f"{mdd:.2%}",               "cls": "neg"},
        {"label": "Num Trades",   "value": str(num_trades),             "cls": ""},
        {"label": "Win Rate",     "value": f"{win_rate:.1%}" if win_rate is not None else "n/a", "cls": ""},
    ]

    return Template(_REPORT_TEMPLATE).render(
        chart_html=chart_html,
        ticker=result.ticker,
        strategy_name=strategy_name,
        rows=rows,
    )
