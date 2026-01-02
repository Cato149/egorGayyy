import datetime as dt
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
from dash.dash_table import DataTable
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

INDICATORS = ["SMA", "EMA", "RSI", "MACD", "BOLLINGER", "ATR"]


def parse_tickers(raw_value: str) -> List[str]:
    return [item.strip().upper() for item in raw_value.split(",") if item.strip()]


@lru_cache(maxsize=64)
def normalize_price_columns(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(level) for level in levels if level]).strip()
            for levels in df.columns.to_list()
        ]
    df = df.rename(columns=str.title)

    for base in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if base in df.columns:
            continue
        matches = [col for col in df.columns if col.startswith(f"{base}_")]
        if len(matches) == 1:
            df = df.rename(columns={matches[0]: base})
    return df


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    data = normalize_price_columns(data)
    data = data.dropna()
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["Sma_20"] = df["Close"].rolling(window=20).mean()
    df["Ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["Rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["Macd"] = ema12 - ema26
    df["Macd_Signal"] = df["Macd"].ewm(span=9, adjust=False).mean()

    rolling_std = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["Sma_20"] + (rolling_std * 2)
    df["Bollinger_Lower"] = df["Sma_20"] - (rolling_std * 2)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["Atr_14"] = true_range.rolling(window=14).mean()
    return df


def indicator_scores(df: pd.DataFrame) -> Dict[str, float]:
    scores = {}
    returns = df["Close"].pct_change().shift(-1)

    if "Sma_20" in df:
        signal = np.where(df["Close"] > df["Sma_20"], 1, -1)
        scores["SMA"] = np.nanmean(signal * returns)

    if "Ema_20" in df:
        signal = np.where(df["Close"] > df["Ema_20"], 1, -1)
        scores["EMA"] = np.nanmean(signal * returns)

    if "Rsi_14" in df:
        signal = np.where(df["Rsi_14"] < 30, 1, np.where(df["Rsi_14"] > 70, -1, 0))
        scores["RSI"] = np.nanmean(signal * returns)

    if "Macd" in df and "Macd_Signal" in df:
        signal = np.where(df["Macd"] > df["Macd_Signal"], 1, -1)
        scores["MACD"] = np.nanmean(signal * returns)

    if "Bollinger_Upper" in df and "Bollinger_Lower" in df:
        signal = np.where(
            df["Close"] < df["Bollinger_Lower"],
            1,
            np.where(df["Close"] > df["Bollinger_Upper"], -1, 0),
        )
        scores["BOLLINGER"] = np.nanmean(signal * returns)

    if "Atr_14" in df:
        atr_median = df["Atr_14"].median()
        signal = np.where(df["Atr_14"] < atr_median, 1, -1)
        scores["ATR"] = np.nanmean(signal * returns)

    return {name: score for name, score in scores.items() if not np.isnan(score)}


def run_tournament(scores: Dict[str, float], winners: int = 4) -> List[Tuple[str, float]]:
    if not scores:
        return []
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores[:winners]


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df[
        [
            "Sma_20",
            "Ema_20",
            "Rsi_14",
            "Macd",
            "Macd_Signal",
            "Bollinger_Upper",
            "Bollinger_Lower",
            "Atr_14",
        ]
    ].copy()
    target = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = features.dropna()
    target = target.loc[features.index]
    return features, target


def predict_growth_probability(df: pd.DataFrame) -> float:
    features, target = build_features(df)
    if len(features) < 50:
        return float("nan")

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target.values

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    model.fit(X[:-1], y[:-1])
    last_features = scaler.transform(features.iloc[[-1]])
    probability = model.predict_proba(last_features)[0, 1]
    return float(probability)


def plot_indicators(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Sma_20"], name="SMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Ema_20"], name="EMA 20"))
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Bollinger_Upper"],
            name="Bollinger Upper",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Bollinger_Lower"],
            name="Bollinger Lower",
            line=dict(dash="dot"),
        )
    )
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def plot_candles(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
            )
        ]
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def tradingview_iframe(symbol: str) -> html.Iframe:
    base = "https://s.tradingview.com/widgetembed/"
    src = (
        f"{base}?symbol={symbol}&interval=D&hidesidetoolbar=1&symboledit=1"
        "&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=light"
    )
    return html.Iframe(src=src, style={"width": "100%", "height": "520px", "border": "0"})


def data_table(df: pd.DataFrame) -> DataTable:
    safe_df = df.copy()
    safe_df.columns = [str(col) for col in safe_df.columns]
    return DataTable(
        data=safe_df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in safe_df.columns],
        page_size=5,
        style_table={"overflowX": "auto"},
        style_header={"fontWeight": "bold"},
    )


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Нейросетевой анализ рынка акций"),
                        html.P(
                            "Пайплайн: загрузка данных → индикаторы → турнир → нейросеть → свечи."
                        ),
                    ],
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Label("Тикеры (через запятую)"),
                        dbc.Input(id="tickers-input", value="AAPL, MSFT"),
                        dbc.Label("Период"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=dt.date.today() - dt.timedelta(days=365 * 2),
                            end_date=dt.date.today(),
                        ),
                        dbc.Label("TradingView символ"),
                        dbc.Input(id="tv-symbol", value="NASDAQ:AAPL"),
                    ],
                    width=3,
                ),
            ],
            className="mb-4",
        ),
        dcc.Tabs(
            [
                dcc.Tab(label="Источник данных", value="data"),
                dcc.Tab(label="Индикаторы", value="indicators"),
                dcc.Tab(label="Турнир", value="tournament"),
                dcc.Tab(label="Нейросеть", value="neural"),
                dcc.Tab(label="Свечи", value="candles"),
            ],
            value="data",
            id="tabs",
        ),
        html.Div(id="tabs-content"),
    ],
    fluid=True,
)


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
    Input("tickers-input", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("tv-symbol", "value"),
)
def render_tab(tab, tickers_raw, start_date, end_date, tv_symbol):
    tickers = parse_tickers(tickers_raw)
    if not tickers:
        return dbc.Alert("Добавьте хотя бы один тикер.", color="warning")

    if tab == "data":
        content = []
        for ticker in tickers:
            data = fetch_prices(ticker, start_date, end_date)
            content.append(html.H5(ticker))
            content.append(html.P(f"Строк: {len(data)}"))
            content.append(data_table(data.tail().reset_index()))
        return html.Div(content)

    if tab == "indicators":
        content = []
        for ticker in tickers:
            data = add_indicators(fetch_prices(ticker, start_date, end_date))
            content.append(html.H5(ticker))
            content.append(dcc.Graph(figure=plot_indicators(data)))
            content.append(data_table(data.tail().reset_index()))
        return html.Div(content)

    if tab == "tournament":
        content = []
        for ticker in tickers:
            data = add_indicators(fetch_prices(ticker, start_date, end_date))
            scores = indicator_scores(data)
            winners = run_tournament(scores, winners=4)
            content.append(html.H5(ticker))
            if winners:
                winner_df = pd.DataFrame(winners, columns=["Индикатор", "Скор"])
                content.append(data_table(winner_df))
            else:
                content.append(dbc.Alert("Недостаточно данных для оценки.", color="warning"))
        return html.Div(content)

    if tab == "neural":
        content = []
        for ticker in tickers:
            data = add_indicators(fetch_prices(ticker, start_date, end_date))
            probability = predict_growth_probability(data)
            content.append(html.H5(ticker))
            if np.isnan(probability):
                content.append(dbc.Alert("Недостаточно данных для обучения модели.", color="warning"))
            else:
                content.append(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Вероятность роста"),
                                html.H3(f"{probability * 100:.2f}%"),
                            ]
                        ),
                        className="mb-3",
                    )
                )
        return html.Div(content)

    if tab == "candles":
        content = []
        for ticker in tickers:
            data = fetch_prices(ticker, start_date, end_date)
            content.append(html.H5(f"{ticker} свечи"))
            content.append(dcc.Graph(figure=plot_candles(data, ticker)))
        content.append(html.H5("TradingView"))
        content.append(tradingview_iframe(tv_symbol))
        return html.Div(content)

    return html.Div()


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
