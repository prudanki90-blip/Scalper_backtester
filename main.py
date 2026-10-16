
import os
from io import StringIO
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

APP_VERSION = "0.2.0"

PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "AUD/USD",
    "USD/JPY",
    "XAU/USD",
]

st.set_page_config(
    page_title="Scalper Backtester",
    page_icon="📈",
    layout="wide",
)

# ---------------- DATA ----------------

def make_demo_data(pair, bars=600):
    """Generate synthetic candles for testing, not real prices."""
    rng = np.random.default_rng(sum(map(ord, pair)))

    start = datetime.now(timezone.utc).replace(
        second=0, microsecond=0
    ) - timedelta(minutes=bars)

    timestamps = pd.date_range(
        start=start,
        periods=bars,
        freq="min",
        tz="UTC",
    )

    base = {
        "EUR/USD": 1.08,
        "GBP/USD": 1.27,
        "AUD/USD": 0.66,
        "USD/JPY": 150.0,
        "XAU/USD": 2650.0,
    }[pair]

    volatility = 0.0012 if pair == "XAU/USD" else 0.00035
    returns = rng.normal(0, volatility, bars)
    close = base * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]

    spread = np.abs(
        rng.normal(0, base * volatility / 2, bars)
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": np.maximum(open_price, close) + spread,
        "low": np.minimum(open_price, close) - spread,
        "close": close,
        "volume": 0,
    })


def normalize_candles(frame):
    """Standardize and validate historical OHLC candle data."""
    if frame is None or frame.empty:
        raise ValueError("The supplied candle data is empty.")

    df = frame.copy()
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    aliases = {
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "vol": "volume",
    }

    df = df.rename(columns=aliases)

    required = [
        "timestamp", "open", "high", "low", "close"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    if "volume" not in df.columns:
        df["volume"] = 0

    df["volume"] = pd.to_numeric(
        df["volume"], errors="coerce"
    ).fillna(0)

    df = df.dropna(subset=required)

    # Exclude invalid prices and impossible OHLC candles.
    price_columns = ["open", "high", "low", "close"]
    df = df[(df[price_columns] > 0).all(axis=1)]

    valid_ohlc = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    )

    df = df[valid_ohlc]
    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} valid candles found. "
            "At least 30 are required."
        )

    return df


def load_twelvedata(pair, interval, outputsize):
    """Load historical candles from Twelve Data."""
    api_key = os.getenv("TWELVEDATA_API_KEY", "")

    try:
        api_key = api_key or st.secrets.get(
            "TWELVEDATA_API_KEY", ""
        )
    except Exception:
        pass

    if not api_key:
        raise ValueError(
            "Twelve Data API key is missing. Add "
            "TWELVEDATA_API_KEY in Streamlit Secrets."
        )

    response = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": pair,
            "interval": interval,
            "outputsize": min(max(outputsize, 30), 5000),
            "apikey": api_key,
            "format": "JSON",
            "timezone": "UTC",
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    if "values" not in payload:
        message = payload.get(
            "message",
            payload.get("code", "No candle data returned."),
        )
        raise ValueError(f"Twelve Data: {message}")

    frame = pd.DataFrame(payload["values"]).rename(
        columns={"datetime": "timestamp"}
    )

    return normalize_candles(frame)


# ---------------- INDICATORS ----------------

def add_indicators(data):
    df = data.copy()
    delta = df["close"].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    df["bb_mid"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)

    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std

    df["sma_fast"] = df["close"].rolling(9).mean()
    df["sma_slow"] = df["close"].rolling(21).mean()

    return df


def generate_signals(data, strategy, rsi_low, rsi_high):
    df = add_indicators(data)
    df["signal"] = "HOLD"

    if strategy == "RSI + Bollinger":
        buy = (
            (df["rsi"] < rsi_low)
            & (df["close"] < df["bb_lower"])
        )
        sell = (
            (df["rsi"] > rsi_high)
            & (df["close"] > df["bb_upper"])
        )

    elif strategy == "Moving-average crossover":
        buy = (
            (df["sma_fast"] > df["sma_slow"])
            & (
                df["sma_fast"].shift(1)
                <= df["sma_slow"].shift(1)
            )
        )
        sell = (
            (df["sma_fast"] < df["sma_slow"])
            & (
                df["sma_fast"].shift(1)
                >= df["sma_slow"].shift(1)
            )
        )

    else:
        buy = df["rsi"] < rsi_low
        sell = df["rsi"] > rsi_high

    df.loc[buy.fillna(False), "signal"] = "BUY"
    df.loc[sell.fillna(False), "signal"] = "SELL"

    return df


# ---------------- BACKTEST ----------------

def run_backtest(
    data, starting_balance, risk_pct, stop_pct, target_pct
):
    """
    Simplified one-position simulation.

    This is not a broker-accurate forex P&L model. It does not
    account for spread, slippage, commissions, swaps or contract
    sizes, and it checks exits using candle closing prices.
    """
    df = data.reset_index(drop=True)
    balance = float(starting_balance)
    trades = []
    equity = []
    position = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        signal = row["signal"]

        if position is not None:
            reason = None

            if position["side"] == "LONG":
                if price <= position["stop"]:
                    reason = "Stop loss"
                elif price >= position["target"]:
                    reason = "Take profit"
                elif signal == "SELL":
                    reason = "Opposite signal"
            else:
                if price >= position["stop"]:
                    reason = "Stop loss"
                elif price <= position["target"]:
                    reason = "Take profit"
                elif signal == "BUY":
                    reason = "Opposite signal"

            if reason:
                direction = (
                    1 if position["side"] == "LONG" else -1
                )
                pnl = (
                    (price - position["entry"])
                    * direction
                    * position["units"]
                )
                balance += pnl

                trades.append({
                    "entry_time": position["time"],
                    "exit_time": row["timestamp"],
                    "side": position["side"],
                    "entry": position["entry"],
                    "exit": price,
                    "units": position["units"],
                    "pnl": pnl,
                    "exit_reason": reason,
                    "balance": balance,
                })

                position = None

        if (
            position is None
            and signal in ("BUY", "SELL")
            and balance > 0
        ):
            risk_amount = balance * risk_pct / 100
            stop_distance = price * stop_pct / 100

            if stop_distance <= 0:
                continue

            units = risk_amount / stop_distance
            side = "LONG" if signal == "BUY" else "SHORT"

            position = {
                "side": side,
                "entry": price,
                "time": row["timestamp"],
                "units": units,
                "stop": price * (
                    1 - stop_pct / 100
                    if side == "LONG"
                    else 1 + stop_pct / 100
                ),
                "target": price * (
                    1 + target_pct / 100
                    if side == "LONG"
                    else 1 - target_pct / 100
                ),
            }

        equity.append({
            "timestamp": row["timestamp"],
            "equity": balance,
        })

    if position is not None and len(df):
        row = df.iloc[-1]
        price = float(row["close"])
        direction = (
            1 if position["side"] == "LONG" else -1
        )

        pnl = (
            (price - position["entry"])
            * direction
            * position["units"]
        )
        balance += pnl

        trades.append({
            "entry_time": position["time"],
            "exit_time": row["timestamp"],
            "side": position["side"],
            "entry": position["entry"],
            "exit": price,
            "units": position["units"],
            "pnl": pnl,
            "exit_reason": "End of data",
            "balance": balance,
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)

    if not equity_df.empty:
        equity_df.loc[
            equity_df.index[-1], "equity"
        ] = balance

    return trades_df, equity_df, balance


# ---------------- DASHBOARD ----------------

st.title("📈 Scalper Backtester")
st.caption(
    f"Version {APP_VERSION} | Analysis and simulation only"
)

with st.sidebar:
    st.header("Configuration")

    pair = st.selectbox("Market", PAIRS)

    source = st.radio(
        "Data source",
        [
            "Synthetic demo data",
            "Upload CSV",
            "Twelve Data API",
        ],
    )

    interval = st.selectbox(
        "Candle interval",
        ["1min", "5min", "15min", "30min", "1h", "4h", "1day"],
        index=1,
    )

    outputsize = st.slider(
        "Candles to load", 100, 2000, 600, 100
    )

    strategy = st.selectbox(
        "Signal strategy",
        [
            "RSI + Bollinger",
            "Moving-average crossover",
            "RSI thresholds",
        ],
    )

    rsi_low = st.slider("RSI lower threshold", 10, 45, 30)
    rsi_high = st.slider("RSI upper threshold", 55, 90, 70)

    starting_balance = st.number_input(
        "Starting simulated balance",
        min_value=100.0,
        value=10000.0,
        step=500.0,
    )

    risk_pct = st.slider(
        "Risk budget per trade (%)",
        0.1, 3.0, 1.0, 0.1,
    )

    stop_pct = st.slider(
        "Stop distance (%)",
        0.05, 2.0, 0.3, 0.05,
    )

    target_pct = st.slider(
        "Target distance (%)",
        0.05, 5.0, 0.6, 0.05,
    )

    run_button = st.button(
        "Load data and run analysis",
        type="primary",
        use_container_width=True,
    )


# ---------------- CSV IMPORT ----------------

uploaded = None
pasted_csv = ""
import_method = "Select a file"

if source == "Upload CSV":
    st.subheader("Import your candle data")

    st.write(
        "Choose a CSV file from your phone or paste its contents. "
        "Required columns: timestamp, open, high, low, close. "
        "At least 30 valid candles are required."
    )

    import_method = st.radio(
        "Choose import method",
        ["Select a file", "Paste CSV text"],
        key="csv_import_method",
    )

    if import_method == "Select a file":
        uploaded = st.file_uploader(
            "Browse your device for a CSV file",
            type=["csv"],
            key="scalper_csv_upload",
        )

        if uploaded is not None:
            st.success(
                f"Selected file: {uploaded.name}"
            )

    else:
        pasted_csv = st.text_area(
            "Paste your CSV data here",
            height=250,
            key="pasted_csv_data",
            placeholder=(
                "timestamp,open,high,low,close,volume\n"
                "2026-10-15 10:00:00,1.1600,1.1610,"
                "1.1590,1.1605,100"
            ),
        )


# ---------------- LOAD AND ANALYZE ----------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_settings" not in st.session_state:
    st.session_state.last_settings = None

settings = (
    pair, source, interval, outputsize, strategy,
    rsi_low, rsi_high, starting_balance,
    risk_pct, stop_pct, target_pct,
)

# Run automatically for the initial demo view.
# Subsequent runs happen when the user presses the button.
should_run = (
    run_button
    or st.session_state.analysis_result is None
)

if should_run:
    try:
        if source == "Synthetic demo data":
            candles = make_demo_data(pair, outputsize)
            source_note = (
                "Synthetic demo candles — NOT real market data."
            )

        elif source == "Upload CSV":
            if import_method == "Select a file":
                if uploaded is None:
                    st.info(
                        "Select a CSV file, or choose "
                        "'Paste CSV text' above."
                    )
                    st.stop()

                raw_csv = pd.read_csv(uploaded)

            else:
                if not pasted_csv.strip():
                    st.info(
                        "Paste your CSV contents into the text box first."
                    )
                    st.stop()

                raw_csv = pd.read_csv(
                    StringIO(pasted_csv.strip())
                )

            candles = normalize_candles(raw_csv)
            source_note = "User-provided candle data."

        else:
            candles = load_twelvedata(
                pair, interval, outputsize
            )
            source_note = "Data returned by Twelve Data API."

        analyzed = generate_signals(
            candles, strategy, rsi_low, rsi_high
        )

        trades, equity, final_balance = run_backtest(
            analyzed,
            starting_balance,
            risk_pct,
            stop_pct,
            target_pct,
        )

        st.session_state.analysis_result = (
            analyzed,
            trades,
            equity,
            final_balance,
            source_note,
            pair,
            strategy,
        )

        st.session_state.last_settings = settings

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()


if st.session_state.analysis_result is None:
    st.info(
        "Choose a data source and press "
        "'Load data and run analysis' to begin."
    )
    st.stop()

(
    analyzed, trades, equity, final_balance,
    source_note, result_pair, result_strategy,
) = st.session_state.analysis_result

if settings != st.session_state.last_settings:
    st.warning(
        "Your settings have changed since the last analysis. "
        "Press 'Load data and run analysis' to update the results."
    )

st.info(source_note)

latest = analyzed.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{latest['close']:.5f}")
c2.metric("Latest signal", latest["signal"])
c3.metric("RSI (14)", f"{latest['rsi']:.1f}")
c4.metric(
    "Simulated balance",
    f"{final_balance:,.2f}",
    f"{final_balance - starting_balance:,.2f}",
)

tab_chart, tab_signals, tab_backtest, tab_data, tab_setup = (
    st.tabs([
        "Chart", "Signals", "Backtest", "Data", "Setup"
    ])
)


# ---------------- CHARTS ----------------

with tab_chart:
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=analyzed["timestamp"],
        open=analyzed["open"],
        high=analyzed["high"],
        low=analyzed["low"],
        close=analyzed["close"],
        name="Candles",
    ))

    for column, label, dash in [
        ("bb_upper", "BB upper", "dot"),
        ("bb_mid", "BB middle", "solid"),
        ("bb_lower", "BB lower", "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=analyzed["timestamp"],
            y=analyzed[column],
            name=label,
            line={"width": 1, "dash": dash},
        ))

    for signal, symbol in [
        ("BUY", "triangle-up"),
        ("SELL", "triangle-down"),
    ]:
        points = analyzed[analyzed["signal"] == signal]

        fig.add_trace(go.Scatter(
            x=points["timestamp"],
            y=points["close"],
            mode="markers",
            name=signal,
            marker={"symbol": symbol, "size": 10},
        ))

    fig.update_layout(
        height=520,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    rsi_fig = go.Figure(go.Scatter(
        x=analyzed["timestamp"],
        y=analyzed["rsi"],
        name="RSI",
    ))

    rsi_fig.add_hline(y=rsi_low, line_dash="dash")
    rsi_fig.add_hline(y=rsi_high, line_dash="dash")
    rsi_fig.update_layout(height=250, yaxis_range=[0, 100])

    st.plotly_chart(rsi_fig, use_container_width=True)


# ---------------- SIGNALS ----------------

with tab_signals:
    st.write(f"Market: **{result_pair}**")
    st.write(f"Strategy: **{result_strategy}**")

    rows = analyzed[
        analyzed["signal"] != "HOLD"
    ].tail(100)

    rows = rows.sort_values(
        "timestamp", ascending=False
    )

    if rows.empty:
        st.warning(
            "No BUY/SELL signals in this dataset."
        )
    else:
        st.dataframe(
            rows[["timestamp", "close", "rsi", "signal"]],
            use_container_width=True,
            hide_index=True,
        )


# ---------------- BACKTEST RESULTS ----------------

with tab_backtest:
    total = len(trades)
    wins = (
        int((trades["pnl"] > 0).sum())
        if total else 0
    )

    win_rate = wins / total * 100 if total else 0
    net_pnl = (
        float(trades["pnl"].sum())
        if total else 0
    )

    if not equity.empty:
        peak = equity["equity"].cummax()
        drawdown = (
            (equity["equity"] - peak)
            / peak.replace(0, np.nan)
        )
        max_dd = float(drawdown.min() * 100)
    else:
        max_dd = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closed trades", total)
    m2.metric("Win rate", f"{win_rate:.1f}%")
    m3.metric("Net simulated P&L", f"{net_pnl:,.2f}")
    m4.metric("Max drawdown", f"{max_dd:.2f}%")

    if not equity.empty:
        eqfig = go.Figure(go.Scatter(
            x=equity["timestamp"],
            y=equity["equity"],
            name="Equity",
        ))

        eqfig.update_layout(height=300)
        st.plotly_chart(eqfig, use_container_width=True)

    if trades.empty:
        st.write("No completed trades in this run.")
    else:
        st.dataframe(
            trades.sort_values(
                "exit_time", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download trade log CSV",
            trades.to_csv(index=False).encode("utf-8"),
            "backtest_trades.csv",
            "text/csv",
        )


# ---------------- CANDLE DATA ------------
