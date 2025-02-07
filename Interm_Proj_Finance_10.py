import numpy as np
import matplotlib.pyplot as plt

import utils
import stock_data_fetcher
from stock_plotter import StockPlotter
from models import Models
from model_evaluator import ModelEvaluator
from time_series_visualizer import TimeSeriesVisualizer
from backtest import BacktestStrategy

from matplotlib.backends.backend_pdf import PdfPages

ticker = "AAPL"
start = "2020-02-01"
end = "2025-01-01"
interval = "1d"
pdf_path = f"results/results_{ticker}_{start}_to_{end}.pdf"

pdf_pages = PdfPages(pdf_path)

stock_fetcher = stock_data_fetcher.StockDataFetcher(ticker, start, end, interval)
stock_data = stock_fetcher.fetch_data()

df = utils.add_technical_indicators(stock_data)

df_scaled, scaler = utils.prepare_features(df)
sequence_length = 15
X, y = utils.create_sequences(df_scaled[19:], sequence_length)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

X_train, y_train = utils.augment_data(X_train, y_train)

plotter = StockPlotter(stock_data, ticker)
fig1 = plotter.plot()
pdf_pages.savefig(fig1)  # Save stock plot to PDF

n_features = X_train.shape[2]
lstm_model = Models.LSTM_Model(sequence_length, n_features)
best_params = lstm_model.tune_hyperparameters(X_train, y_train)
history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)
y_pred_lstm = lstm_model.predict(X_test).flatten()

models = {
    'LSTM1': lstm_model,
    #'LSTM2': lstm_model  # Duplicate model with different names
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

std_devs = {}  # Ensure std_devs exists even if sigma_gp is not available
    
predictions = {
    'LSTM': y_pred_lstm
}
visualizer = TimeSeriesVisualizer(y_test, predictions, ticker, std_devs)

fig3 = visualizer.plot_predictions()
pdf_pages.savefig(fig3)
fig4 = visualizer.compare_models()
pdf_pages.savefig(fig4)

backtester = BacktestStrategy(initial_balance=10000, transaction_cost=0.001)

results_lstm = backtester.backtest(y_test, y_pred_lstm, model_name="LSTM")
metrics_lstm = backtester.evaluate_performance(results_lstm)

fig5 = backtester.plot_results(results_lstm, model_name="LSTM")  
pdf_pages.savefig(fig5)

# results_xgb = backtester.backtest(y_test, y_pred_xgb, model_name="XGBoost")
# metrics_xgb = backtester.evaluate_performance(results_xgb)
# fig4 = backtester.plot_results(results_xgb, model_name="XGBoost")
# pdf_pages.savefig(fig4)
# print("Backtest results for XGBoost saved to PDF (Page 4).")

pdf_pages.close()
print(f"PDF successfully saved in: {pdf_path}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


