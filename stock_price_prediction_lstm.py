# stock_price_prediction_lstm.py
# Created by Smit Chovatiya

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf

# Step 1: Load historical stock price data
ticker = 'AAPL'  # Apple Inc.
data = yf.download(ticker, start='2015-01-01', end='2025-04-20')
closing_prices = data['Close'].values.reshape(-1, 1)

# Step 2: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Step 3: Prepare training data
X_train = []
y_train = []

window_size = 60  # number of previous days to consider

for i in range(window_size, len(scaled_data)):
    X_train.append(scaled_data[i - window_size:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # (samples, time steps, features)

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Step 6: Prepare test data (last 60 days of the original dataset)
test_data = scaled_data[-(window_size + 30):]  # last few data points for prediction
X_test = []

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 7: Predict and unscale the values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Step 8: Visualize the result
real_prices = closing_prices[-len(predictions):]

plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Real Stock Price', color='blue')
plt.plot(predictions, label='Predicted Stock Price', color='orange')
plt.title(f'{ticker} Stock Price Prediction using LSTM')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
