import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import datetime as dt
import numpy as np
import yfinance as yf

# Fetch stock data using yfinance
START_DATE = '2018-01-01'
END_DATE = '2024-01-01'
NVDA_stock = yf.download('NVDA', start=START_DATE, end=END_DATE)
AMD_stock = yf.download('AMD', start=START_DATE, end=END_DATE)
TSLA_stock = yf.download('TSLA', start=START_DATE, end=END_DATE)

# Index is already DatetimeIndex
# 1. Interactive Candlestick Chart for NVDA
fig_candlestick = go.Figure(data=[go.Candlestick(x=NVDA_stock.index,
                                                open=NVDA_stock['Open'],
                                                high=NVDA_stock['High'],
                                                low=NVDA_stock['Low'],
                                                close=NVDA_stock['Close'],
                                                name='NVDA')])

fig_candlestick.update_layout(
    title='NVDA Interactive Candlestick Chart',
    yaxis_title='Stock Price (USD)',
    xaxis_title='Date',
    template='plotly_white',
    height=600
)

fig_candlestick.show()

# 2. Multi-stock Comparison with Interactive Features
fig_comparison = go.Figure()

fig_comparison.add_trace(go.Scatter(
    x=NVDA_stock.index,
    y=NVDA_stock['Close'],
    mode='lines',
    name='NVDA',
    line=dict(color='#2E8B57', width=2)
))

fig_comparison.add_trace(go.Scatter(
    x=AMD_stock.index,
    y=AMD_stock['Close'],
    mode='lines',
    name='AMD',
    line=dict(color='#DC143C', width=2)
))

fig_comparison.add_trace(go.Scatter(
    x=TSLA_stock.index,
    y=TSLA_stock['Close'],
    mode='lines',
    name='TSLA',
    line=dict(color='#4169E1', width=2)
))

fig_comparison.update_layout(
    title='Multi-Stock Price Comparison',
    xaxis_title='Date',
    yaxis_title='Close Price (USD)',
    template='plotly_white',
    height=500,
    hovermode='x unified'
)

fig_comparison.show()

# 3. Interactive Dashboard with Subplots
fig_dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Stock Price Trends', 'Volume Analysis', 'Price Distribution', 'Daily Returns'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Stock Price Trends
fig_dashboard.add_trace(
    go.Scatter(x=NVDA_stock.index, y=NVDA_stock['High'], name='High', line=dict(color='#2E8B57')),
    row=1, col=1
)
fig_dashboard.add_trace(
    go.Scatter(x=NVDA_stock.index, y=NVDA_stock['Low'], name='Low', line=dict(color='#DC143C')),
    row=1, col=1
)
fig_dashboard.add_trace(
    go.Scatter(x=NVDA_stock.index, y=NVDA_stock['Close'], name='Close', line=dict(color='#4169E1')),
    row=1, col=1
)

# Volume Analysis (if available)
if 'Volume' in NVDA_stock.columns:
    fig_dashboard.add_trace(
        go.Bar(x=NVDA_stock.index, y=NVDA_stock['Volume'], name='Volume', marker_color='#20B2AA'),
        row=1, col=2
    )

# Price Distribution
fig_dashboard.add_trace(
    go.Histogram(x=NVDA_stock['Close'], name='Price Distribution', nbinsx=30, marker_color='#9370DB'),
    row=2, col=1
)

# Daily Returns
daily_returns = NVDA_stock['Close'].pct_change().dropna()
fig_dashboard.add_trace(
    go.Scatter(x=daily_returns.index, y=daily_returns, name='Daily Returns', line=dict(color='#FF6347')),
    row=2, col=2
)

fig_dashboard.update_layout(
    title='Interactive Stock Analysis Dashboard',
    height=800,
    template='plotly_white',
    showlegend=True
)

fig_dashboard.show()

# 4. Correlation Heatmap
# Calculate correlation between different stocks
correlation_data = pd.DataFrame({
    'NVDA': NVDA_stock['Close'],
    'AMD': AMD_stock['Close'],
    'TSLA': TSLA_stock['Close']
}).corr()

fig_heatmap = px.imshow(
    correlation_data,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu',
    title='Stock Price Correlation Matrix'
)

fig_heatmap.update_layout(
    height=500,
    template='plotly_white'
)

fig_heatmap.show()

# 5. Moving Averages
def calculate_moving_averages(data, windows=[20, 50]):
    """Calculate moving averages for given windows"""
    result = data.copy()
    for window in windows:
        result[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return result

NVDA_with_ma = calculate_moving_averages(NVDA_stock)

fig_ma = go.Figure()

fig_ma.add_trace(go.Scatter(
    x=NVDA_with_ma.index,
    y=NVDA_with_ma['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='#4169E1', width=2)
))

fig_ma.add_trace(go.Scatter(
    x=NVDA_with_ma.index,
    y=NVDA_with_ma['MA_20'],
    mode='lines',
    name='20-day MA',
    line=dict(color='#FF6347', width=2)
))

fig_ma.add_trace(go.Scatter(
    x=NVDA_with_ma.index,
    y=NVDA_with_ma['MA_50'],
    mode='lines',
    name='50-day MA',
    line=dict(color='#2E8B57', width=2)
))

fig_ma.update_layout(
    title='NVDA Price with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    template='plotly_white',
    height=500,
    hovermode='x unified'
)

fig_ma.show()

print("Interactive visualizations created successfully!")
print("Features included:")
print("1. Interactive candlestick chart")
print("2. Multi-stock comparison")
print("3. Comprehensive dashboard with subplots")
print("4. Correlation heatmap")
print("5. Moving averages analysis") 