# Stock Market Price Prediction using LSTM

This mini project uses advanced Python libraries such as NumPy, Pandas, Matplotlib, Keras, and TensorFlow to predict stock prices using an LSTM (Long Short-Term Memory) neural network.

## Technologies Used
- NumPy
- Pandas
- Matplotlib
- Keras
- TensorFlow
- scikit-learn
- yfinance (for live data)

## How It Works
1. Fetch historical stock data using yFinance.
2. Normalize and prepare data using a window of past 60 days.
3. Build and train an LSTM model using Keras.
4. Predict future stock prices.
5. Visualize real vs. predicted stock prices.

## Output Example

![Output Chart](screenshot.png)

## Run It Yourself

```bash
pip install -r requirements.txt
python stock_price_prediction_lstm.py
