import numpy as np
import matplotlib.pyplot as plt
import utils
from stock_plotter import StockPlotter
from models import Models
from model_evaluator import ModelEvaluator
from time_series_visualizer import TimeSeriesVisualizer
from backtest import BacktestStrategy
from stock_prediction import StockPrediction

#To get data from pre-loaded csv file 
ticker = "AAPL"
start = "2022-01-01"
end = "2025-01-01"
stock_data = utils.get_csv(ticker, start, end)
print(type(stock_data)) 
print(stock_data) 
print(ticker)

df = utils.add_technical_indicators(stock_data, "Close") #df not scaled
df = df.dropna()
print(df)

df_scaled, scaler = utils.prepare_features(df) #returns scaled df
print(type(df_scaled))
print(df_scaled)
sequence_length = 15
X, y = utils.create_sequences(df_scaled[19:], sequence_length) #returns sequences ndarray
print(f"\nX type {type(X)}, size {X.size}, shape {X.shape} (samples, timesteps, features)")
print(f"{X[:3]}usw...")
print(f"\nX type {type(y)}, size {y.size}, shape {y.shape} (samples,)")
print(f"{y[:3]}usw...")

train_size = int(len(X) * 0.8) #splits in train and test
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

price_test_start = stock_data.iloc[(len(X_test))]["Close"]
date_test_start = stock_data.iloc[(len(X_test))]["Date"]
print(price_test_start)
print(date_test_start)

#Draws plot
plotter = StockPlotter(stock_data, ticker)
fig1 = plotter.plot()

# **Understanding This Chart:**
# - **Closing Price**: The daily closing price trend over time.
# - **RSI (Relative Strength Index)**: Identifies overbought (>70) and oversold (<30) conditions.
# - **MACD (Moving Average Convergence Divergence)**: A trend-following momentum indicator.
# **Use this analysis to identify trading opportunities and assess market trends.**

# Initialize Backtest Strategy
backtest_strategy = BacktestStrategy("Advanced Strategy", ticker)

# Generate backtest results
backtest_results = backtest_strategy.run_backtest(df, "Close")
print(backtest_results)
backtest_strategy.plot_backtest_results(backtest_results)
print(backtest_results["Portfolio Value"].iloc[-1])

portfolio_test_start = backtest_results.iloc[(-len(X_test)+1)]["Portfolio Value"]
date_test_start = backtest_results.iloc[-(len(X_test)+1)]["Date"]
print(portfolio_test_start)
print(date_test_start)

# **Understanding This Chart:**
#  - The green line represents the portfolio value over time.
#  - This backtest simulates an automated trading strategy based on technical indicators.
# **Trading Rules Applied:**
#  - Buy when RSI is rising and price is higher than 3 days ago (ensures uptrend confirmation).
#  - Sell only if price has increased by at least 2% since buying (prevents premature exits).
#  - MACD and Moving Averages are used for additional trade confirmation.
# **Key Notes:**
#  - Portfolio value changes based on executed trades.
#  - The system avoids unnecessary trades by ensuring multiple confirmations before buying or selling.
#  - Unlike basic strategies, this approach prevents buying in a downtrend and avoids frequent stop-outs.
# This backtest provides a realistic view of potential trading performance, but real-world trading includes slippage, execution delays, and psychological factors.

n_features = X_train.shape[2] #the number of third dimension of X_train, 11 features
lstm_model = Models.LSTM_Model(sequence_length, n_features) #initializes model
best_params = lstm_model.tune_hyperparameters(X_train, y_train) #tunes hyperparameters, creates the model
history_lstm = lstm_model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1) #stores training history

print(f"X_test shape: {X_test.shape}")  # Check for consistent shape
X_test = np.array(X_test, dtype=np.float32)  # Ensure fixed dtype, to avoid error message

y_pred_lstm = lstm_model.predict(X_test).flatten() #flatten to take away the second dimension from ndarray (xxx, 1) to (xxx,)
lstm_model.summary()
# Plot training & validation loss
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("LSTM Training Loss vs Validation Loss")
plt.show()

xgb_model = Models.XGB_Model(n_estimators=150, max_depth=8, learning_rate=0.05, gamma=0.3, subsample=0.9)
xgb_model.summary()  # ✅ Now the method exists and will print model details

xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
y_pred_xgb = predictions.flatten()

fig7 = xgb_model.plot_training_history(metric="rmse")

gp_model = Models.GP_Model()
gp_model.fit(X_train, y_train)

# Predict with confidence intervals
y_pred_gp, sigma_gp = gp_model.predict(X_test, return_std=True)
#gp_model = GaussianProcessModel()
fig8 = gp_model.validate(X_train, y_train)

models = {
    'LSTM': lstm_model,
    'XGBoost': xgb_model,
    'Gaussian Process': y_pred_gp
}
evaluator = ModelEvaluator(models, X_test, y_test, ticker)

# Evaluate models
evaluation_results, y_predictions = evaluator.evaluate()

# Print results
for model, metrics in evaluation_results.items():
    print(f"\nModel: {model}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

print(len(X_test))

std_devs = {}  # Ensure std_devs exists even if sigma_gp is not available

if 'sigma_gp' in globals():
    std_devs['GP'] = sigma_gp  # Add Gaussian Process std deviation if available# Assuming you have predictions from LSTM, XGBoost, and Gaussian Process
    
predictions = {
    'LSTM': y_pred_lstm,
    'XGBoost': y_pred_xgb,
    'Gaussian Process': y_pred_gp
}
if 'sigma_gp' in globals():
    std_devs = {'GP': sigma_gp}

visualizer = TimeSeriesVisualizer(y_test, predictions, ticker, std_devs)

fig3 = visualizer.plot_predictions()

# **Understanding This Chart:**
# - **Blue Line (Actual)**: Represents true stock price movements.
# - **Red Dashed Line (Predicted)**: Model’s predicted values.
# - **Pink Shaded Area (Confidence Interval)**: Represents prediction uncertainty (if available).
# **Use this visualization to evaluate model accuracy and prediction confidence levels.**

module_name = "LSTM"
prediction = StockPrediction(y_pred_lstm, price_test_start, date_test_start, ticker, portfolio_test_start, module_name)
backtest_results = prediction.run_backtest()
prediction.plot_predictions_and_backtest(backtest_results)

module_name = "XGBoost"
prediction = StockPrediction(y_pred_xgb, price_test_start, date_test_start, ticker, portfolio_test_start, module_name)
backtest_results = prediction.run_backtest()
prediction.plot_predictions_and_backtest(backtest_results)

module_name = "Gaussian Process"
prediction = StockPrediction(y_pred_gp, price_test_start, date_test_start, ticker, portfolio_test_start, module_name)
backtest_results = prediction.run_backtest()
prediction.plot_predictions_and_backtest(backtest_results)