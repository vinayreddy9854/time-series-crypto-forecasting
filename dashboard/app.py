import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from dash.dependencies import Input, Output
import joblib
import requests

def fetch_crypto_data(crypto_id='bitcoin', days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': str(days)}
    response = requests.get(url, params=params)
    data = response.json()

    prices = data['prices']
    timestamps = [entry[0] for entry in prices]
    price_values = [entry[1] for entry in prices]

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='ms'),
        'price': price_values
    })

    df['rolling_mean_3'] = df['price'].rolling(window=3).mean()
    df['rolling_std_3'] = df['price'].rolling(window=3).std()
    df['rolling_mean_7'] = df['price'].rolling(window=7).mean()
    df['rolling_std_7'] = df['price'].rolling(window=7).std()
    df['rolling_mean_14'] = df['price'].rolling(window=14).mean()
    df['rolling_std_14'] = df['price'].rolling(window=14).std()
    df['lag_1'] = df['price'].shift(1)

    return df.dropna().reset_index(drop=True)
    
app = dash.Dash(__name__)
model = load_model("models/lstm_model.keras")
y_scaler = joblib.load('models/y_scaler.save')
df = fetch_crypto_data('bitcoin', 365)
features = ['price', 'lag_1', 'rolling_mean_3', 'rolling_std_3',
            'rolling_mean_7', 'rolling_std_7']
lookback = 60
historical_predictions = []
timestamps = []

if len(df) > lookback:
    for i in range(len(df) - lookback):
        input_data = df[features].iloc[i:i+lookback].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(input_data)
        reshaped = np.reshape(scaled, (1, lookback, 6))
        predicted = model.predict(reshaped)
        predicted_price = y_scaler.inverse_transform(predicted)[0][0]
        historical_predictions.append(predicted_price)
        timestamps.append(df['timestamp'].iloc[i + lookback])
else:
    predicted_price = None

risk_level = "Insufficient Data"
investment_advice = "Not Available"

if len(df) >= 25:
    prediction_data = df[features].values[-lookback:]
    scaler = MinMaxScaler()
    normalized_input = scaler.fit_transform(prediction_data)
    input_sequence = np.reshape(normalized_input, (1, lookback, 6))
    predicted_normalized = model.predict(input_sequence)
    predicted_price = y_scaler.inverse_transform(predicted_normalized)[0][0]

    volatility = df['rolling_std_7'].iloc[-1]
    price_trend = df['price'].iloc[-1] - df['price'].iloc[-2]
    vol_mean = df['rolling_std_7'].mean()

    if volatility > vol_mean * 1.5 and price_trend < 0:
        risk_level = "High Risk"
        investment_advice = "Sell"
    elif volatility > vol_mean and price_trend >= 0:
        risk_level = "Moderate Risk"
        investment_advice = "Hold"
    else:
        risk_level = "Low Risk"
        investment_advice = "Buy"

app.layout = html.Div([
    html.H1("Cryptocurrency Price Prediction Dashboard"),

    html.Div([
        html.Div([
            html.H3(f"Predicted Price: {predicted_price:.2f}" if predicted_price else "Predicted Price: Not available"),
            dcc.Graph(id='candlestick-chart')
        ], className="six columns"),

        html.Div([
            dcc.Graph(id='moving-averages-chart'),
            dcc.Graph(id='volatility-chart')
        ], className="six columns"),
    ], className="row"),

   

    html.Div([
        html.H3("Historical vs Predicted Prices"),
        dcc.Graph(id='historical-vs-predicted-chart')
    ]),

     html.Div([
        html.H3(f"Risk Assessment: {risk_level}"),
        html.H4(f"Investment Strategy: {investment_advice}")
    ], className="row"),
])

@app.callback(
    Output('candlestick-chart', 'figure'),
    [Input('candlestick-chart', 'id')]
)
def update_candlestick_chart(_):
    trace_candle = go.Candlestick(
        x=df['timestamp'],
        open=df['price'] - df['rolling_std_7'],
        high=df['price'] + df['rolling_std_7'],
        low=df['price'] - df['rolling_std_7'],
        close=df['price'],
        name="Price"
    )
    layout = go.Layout(title="Cryptocurrency Price (Candlestick)", xaxis=dict(title="Date"), yaxis=dict(title="Price"))
    return {'data': [trace_candle], 'layout': layout}

@app.callback(
    Output('moving-averages-chart', 'figure'),
    [Input('moving-averages-chart', 'id')]
)
def update_moving_averages_chart(_):
    traces = [
        go.Scatter(x=df['timestamp'], y=df['rolling_mean_7'], mode='lines', name='7-Day MA'),
        go.Scatter(x=df['timestamp'], y=df['rolling_mean_14'], mode='lines', name='14-Day MA')
    ]
    layout = go.Layout(title="Moving Averages (7 & 14 Days)", xaxis=dict(title="Date"), yaxis=dict(title="Price"))
    return {'data': traces, 'layout': layout}

@app.callback(
    Output('volatility-chart', 'figure'),
    [Input('volatility-chart', 'id')]
)
def update_volatility_chart(_):
    trace_volatility = go.Scatter(
        x=df['timestamp'],
        y=df['rolling_std_7'],
        mode='lines',
        name='7-Day Volatility',
        line=dict(color='red')
    )
    layout = go.Layout(title="Volatility (7-Day Std Dev)", xaxis=dict(title="Date"), yaxis=dict(title="Volatility"))
    return {'data': [trace_volatility], 'layout': layout}

@app.callback(
    Output('historical-vs-predicted-chart', 'figure'),
    [Input('historical-vs-predicted-chart', 'id')]
)
def update_historical_vs_predicted(_):
    trace_actual = go.Scatter(
        x=timestamps,
        y=df['price'].iloc[-len(historical_predictions):],
        mode='lines',
        name='Actual Price'
    )
    trace_predicted = go.Scatter(
        x=timestamps,
        y=historical_predictions,
        mode='lines',
        name='Predicted Price'
    )
    layout = go.Layout(title="Actual vs Predicted Prices", xaxis=dict(title="Date"), yaxis=dict(title="Price"))
    return {'data': [trace_actual, trace_predicted], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
