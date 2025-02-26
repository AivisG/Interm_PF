import numpy as np

import utils
from stock_plotter import StockPlotter
from models import Models
from model_evaluator import ModelEvaluator
from time_series_visualizer import TimeSeriesVisualizer
from backtest import BacktestStrategy

from matplotlib.backends.backend_pdf import PdfPages

#To get data from pre-loaded csv file 
ticker = "AAPL"
start = "2024-01-01"
end = "2025-01-01"
stock_data = utils.get_csv(ticker, start, end)
print(stock_data.head()) 
print(ticker)

pdf_path = f"results/results_{ticker}_{start}_to_{end}.pdf"
pdf_pages = PdfPages(pdf_path)

df = utils.add_technical_indicators(stock_data)

# Drop NaNs before scaling
df = df.dropna().reset_index(drop=True)

# Prepare data (features + target)
df_scaled, scaler_features, scaler_target = utils.prepare_features(df)

# Create sequences (using the scaled DataFrame)
sequence_length = 15  # Example: 15 days of past data
df_scaled = df_scaled.drop(columns=['Date'], errors='ignore')
X, y = utils.create_sequences(df_scaled, target_column='Close', sequence_length=sequence_length)

print("X shape:", X.shape)  # Expected: (num_samples, sequence_length, num_features)
print("y shape:", y.shape)  # Expected: (num_samples,)

print("Original DataFrame (df) Head:\n", df.head(17))
print("Scaled DataFrame (df_scaled) Head:\n", df_scaled.head(17))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

# Print the first sequence
print("First sequence (X_train[0]):\n", X_train[0])

# Print the last timestep in the first sequence
print("Last timestep in first sequence (X_train[0][-1]):\n", X_train[0][-1])

# Print expected target
print("Expected Next Close Price (y_train[0]):", y_train[0])

# Find the actual row in df_scaled that should match X_train[0][-1]
first_sequence_check = df_scaled.iloc[:15]  # The first 15 rows for comparison
expected_y_train_check = df_scaled.iloc[15]["Close"]  # The expected y_train[0] value

print("First 15 rows from df_scaled:\n", first_sequence_check)
print("Expected y_train[0] from df_scaled:", expected_y_train_check)

# Check if the last timestep in X_train[0] corresponds to y_train[0]
print("Last timestep in first sequence (X_train[0][-1]):\n", X_train[0][-1])  # Last row in first sequence
print("Corresponding y_train[0]:", y_train[0])  # Target value

# Extract the actual closing price from the last timestep in X_train[0]
last_close_price_in_X = X_train[0][-1][df_scaled.columns.get_loc("Close")]

print("Last Close Price in X_train[0][-1]:", last_close_price_in_X)
print("Expected Next Close Price (y_train[0]):", y_train[0])

# Initialize plotter
plotter = StockPlotter(stock_data, ticker)

# Generate figures and save them
fig1 = plotter.plot_closing_price()
pdf_pages.savefig(fig1)  # Save Closing Price plot

fig2 = plotter.plot_rsi()
pdf_pages.savefig(fig2)  # Save RSI plot

fig3 = plotter.plot_macd()
pdf_pages.savefig(fig3)  # Save MACD plot

# Initialize Backtest Strategy
backtest_strategy = BacktestStrategy("Advanced Strategy")

# Generate backtest results
backtest_results = backtest_strategy.run_backtest(df)
fig4 = plotter.plot_backtest_results(backtest_results)
pdf_pages.savefig(fig4)  # Save Backtest plot

n_features = X_train.shape[2]
lstm_model = Models.LSTM_Model(sequence_length, n_features)

# Train the model
history_lstm = lstm_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1)

# Make predictions
#y_pred_lstm = lstm_model.predict(X_test).flatten()
y_pred_lstm = lstm_model.predict(X_test)
print(y_pred_lstm)# Get predictions in scaled format
y_pred_lstm = scaler_target.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten() # Convert back to real prices for MaxScaler
print(y_pred_lstm)
print(y_test)

print(f"Number of Features in X_train: {n_features}")
print("Shape of X_train:", X_train.shape)

xgb_model = Models.XGB_Model(n_estimators=150, max_depth=8, learning_rate=0.05, gamma=0.3, subsample=0.9)

xgb_model.summary()  # ✅ Now the method exists and will print model details

xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
y_pred_xgb = predictions.flatten()

gp_model = Models.GP_Model()
gp_model.fit(X_train, y_train)

# Predict with confidence intervals
y_pred_gp, sigma_gp = gp_model.predict(X_test, return_std=True)

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

fig2 = evaluator.plot_metrics()
pdf_pages.savefig(fig2)
print(len(X_test))

from matplotlib.backends.backend_pdf import PdfPages

# Create a PDF file
pdf_pages = PdfPages("time_series_plots.pdf")

# Ensure std_devs exists
std_devs = {}  
if 'sigma_gp' in globals():
    std_devs['GP'] = sigma_gp

# Predictions dictionary
predictions = {
    'LSTM': y_pred_lstm,
    'XGBoost': y_pred_xgb,
    'Gaussian Process': y_pred_gp
}

# Initialize visualizer
visualizer = TimeSeriesVisualizer(y_test, predictions, ticker, std_devs)

# Generate separate plots for each model and save them
figs = visualizer.plot_predictions()
for fig in figs:
    pdf_pages.savefig(fig)

# Generate and save the comparison plot
fig_compare = visualizer.compare_models()
pdf_pages.savefig(fig_compare)

fig6 = utils.plot_training_history(history_lstm)  # Call only once and store result

if fig6 is not None:  # Ensure fig7 is valid before saving
    pdf_pages.savefig(fig6)
    print("Backtest results for LSTM saved to PDF (Page 6).")
else:
    print("Error: plot_training_history did not return a figure.")

fig7 = xgb_model.plot_training_history(metric="rmse")

if fig7 is not None:  # Ensure fig7 is valid before saving
    pdf_pages.savefig(fig7)
    print("Backtest results for XGBoost saved to PDF (Page 7).")
else:
    print("Error: plot_training_history did not return a figure.")

#gp_model = GaussianProcessModel()
fig8 = gp_model.validate(X_train, y_train)
pdf_pages.savefig(fig8)  # ✅ Save to PDF

# Close the PDF file
pdf_pages.close()
print("PDF successfully closed")

#Falls nur ein Teill von Code wird ausgefuert und kein pdf erzeugt wird, sonst Fehlermeldung
pdf_pages = PdfPages("output.pdf")

if pdf_pages.get_pagecount() > 0:  # Direkter Vergleich ohne len()
    pdf_pages.close()
else:
    pdf_pages._file.close()  # Datei korrekt schließen, um die Warnung zu vermeiden

