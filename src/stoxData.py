import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

# Set modern plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

#import stock data
NVDA_stock = pd.read_csv('./NVDA.csv', index_col = "Date")
NVDA_stock.head()
AMD_stock =  pd.read_csv('./AMD.csv', index_col = "Date")
AMD_stock.head()
TSLA_stock =  pd.read_csv('./TSLA.csv', index_col = "Date")
TSLA_stock.head()

#set values from imported files
x_dates_NVDA = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in NVDA_stock.index.values]

# Create comprehensive visualization dashboard
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Stock Prediction Analysis Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Stock Price Trends
ax1 = axes[0, 0]
ax1.plot(x_dates_NVDA, NVDA_stock["High"], label="High", color='#2E8B57', linewidth=2)
ax1.plot(x_dates_NVDA, NVDA_stock["Low"], label="Low", color='#DC143C', linewidth=2)
ax1.plot(x_dates_NVDA, NVDA_stock["Close"], label="Close", color='#4169E1', linewidth=2)
ax1.set_title('NVDA Stock Price Trends', fontweight='bold')
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Plot 2: Volume Analysis
ax2 = axes[0, 1]
if 'Volume' in NVDA_stock.columns:
    ax2.bar(x_dates_NVDA, NVDA_stock["Volume"], alpha=0.7, color='#20B2AA')
    ax2.set_title('NVDA Trading Volume', fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
else:
    ax2.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Volume Data', fontweight='bold')

# Plot 3: Price Distribution
ax3 = axes[1, 0]
ax3.hist(NVDA_stock["Close"], bins=30, alpha=0.7, color='#9370DB', edgecolor='black')
ax3.set_title('NVDA Close Price Distribution', fontweight='bold')
ax3.set_xlabel("Price (USD)")
ax3.set_ylabel("Frequency")
ax3.grid(True, alpha=0.3)

# Plot 4: Daily Returns
ax4 = axes[1, 1]
daily_returns = NVDA_stock["Close"].pct_change().dropna()
ax4.plot(x_dates_NVDA[1:], daily_returns, color='#FF6347', alpha=0.7)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.set_title('NVDA Daily Returns', fontweight='bold')
ax4.set_xlabel("Date")
ax4.set_ylabel("Daily Return")
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.tight_layout()
plt.show()

target_y = NVDA_stock['Close']
X_feat = NVDA_stock.iloc[:,0:3]

#Feature Scaling
sc = StandardScaler()
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns = X_feat.columns, data = X_ft, index= X_feat.index)

#Create windows 
def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps+1):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i+ n_steps-1, -1])


    return np.array(X), np.array(y)


X1, y1 = lstm_split(X_feat.values, 2)

train_split = 0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = X_ft.index

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]

X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

print(X1.shape, X_train.shape, X_test.shape, y_test.shape)


lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation="relu"))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer="adam")
lstm.summary()


history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)


y_pred = lstm.predict(X_test)


rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("RSME: ", rmse)
print("MAPE: ", mape)

# Model Performance Visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('LSTM Model Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Training History
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], color='#2E8B57', linewidth=2)
ax1.set_title('Training Loss Over Time', fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
ax2 = axes[0, 1]
test_dates = x_dates_NVDA[split_idx+1:split_idx+1+len(y_test)]
ax2.plot(test_dates, y_test, label='Actual', color='#4169E1', linewidth=2)
ax2.plot(test_dates, y_pred.flatten(), label='Predicted', color='#DC143C', linewidth=2)
ax2.set_title('Actual vs Predicted Values', fontweight='bold')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = y_test - y_pred.flatten()
ax3.scatter(y_test, residuals, alpha=0.6, color='#9370DB')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
ax3.set_title('Residual Plot', fontweight='bold')
ax3.set_xlabel("Actual Values")
ax3.set_ylabel("Residuals")
ax3.grid(True, alpha=0.3)

# Plot 4: Prediction Error Distribution
ax4 = axes[1, 1]
ax4.hist(residuals, bins=20, alpha=0.7, color='#20B2AA', edgecolor='black')
ax4.set_title('Prediction Error Distribution', fontweight='bold')
ax4.set_xlabel("Prediction Error")
ax4.set_ylabel("Frequency")
ax4.grid(True, alpha=0.3)

# Add performance metrics as text
fig.text(0.02, 0.02, f'RMSE: {rmse:.4f}\nMAPE: {mape:.4f}', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

plt.tight_layout()
plt.show()

# Multi-stock comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Multi-Stock Comparison', fontsize=16, fontweight='bold')

# NVDA
x_dates_NVDA = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in NVDA_stock.index.values]
axes[0].plot(x_dates_NVDA, NVDA_stock["Close"], color='#2E8B57', linewidth=2)
axes[0].set_title('NVDA', fontweight='bold')
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Close Price")
axes[0].grid(True, alpha=0.3)

# AMD
x_dates_AMD = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in AMD_stock.index.values]
axes[1].plot(x_dates_AMD, AMD_stock["Close"], color='#DC143C', linewidth=2)
axes[1].set_title('AMD', fontweight='bold')
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Close Price")
axes[1].grid(True, alpha=0.3)

# TSLA
x_dates_TSLA = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in TSLA_stock.index.values]
axes[2].plot(x_dates_TSLA, TSLA_stock["Close"], color='#4169E1', linewidth=2)
axes[2].set_title('TSLA', fontweight='bold')
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Close Price")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()