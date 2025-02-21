#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import utils
import stock_data_fetcher
from stock_plotter import StockPlotter
from models import Models
from model_evaluator import ModelEvaluator
from time_series_visualizer import TimeSeriesVisualizer
from transformer_model import TimeSeriesTransformer
from lstm_transformer_model import LSTMTransformer
from backtest import BacktestStrategy

from matplotlib.backends.backend_pdf import PdfPages


# #To get data from yfinance:
# ticker = "AAPL"
# start = "2020-01-01"
# end = "2025-01-01"
# interval = "1d"
# 
# stock_fetcher = stock_data_fetcher.StockDataFetcher(ticker, start, end, interval)
# stock_data = stock_fetcher.fetch_data()

# In[2]:


#To get data from pre-loaded csv file 
ticker = "AAPL"
start = "2022-01-01"
end = "2025-01-01"
stock_data = utils.get_csv(ticker, start, end)
print(stock_data.head()) 
print(ticker)


# In[3]:


pdf_path = f"results/results_{ticker}_{start}_to_{end}.pdf"
pdf_pages = PdfPages(pdf_path)

df = utils.add_technical_indicators(stock_data)

df_scaled, scaler = utils.prepare_features(df)
sequence_length = 15
X, y = utils.create_sequences(df_scaled[19:], sequence_length)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)


# In[4]:


#Draws plot
plotter = StockPlotter(stock_data, ticker)
fig1 = plotter.plot()
pdf_pages.savefig(fig1)  # Save stock plot to PDF


# In[5]:


n_features = X_train.shape[2]
lstm_model = Models.LSTM_Model(sequence_length, n_features)
best_params = lstm_model.tune_hyperparameters(X_train, y_train)
history_lstm = lstm_model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)
y_pred_lstm = lstm_model.predict(X_test).flatten()


# In[8]:


xgb_model = Models.XGB_Model(n_estimators=150, max_depth=8, learning_rate=0.05, gamma=0.3, subsample=0.9)

xgb_model.summary()  # ✅ Now the method exists and will print model details

xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
y_pred_xgb = predictions.flatten()


# In[9]:


gp_model = Models.GP_Model()
gp_model.fit(X_train, y_train)

# Predict with confidence intervals
y_pred_gp, sigma_gp = gp_model.predict(X_test, return_std=True)


# In[10]:


models = {
    'LSTM1': lstm_model,
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


# In[11]:


std_devs = {}  # Ensure std_devs exists even if sigma_gp is not available

if 'sigma_gp' in globals():
    std_devs['GP'] = sigma_gp  # Add Gaussian Process std deviation if available# Assuming you have predictions from LSTM, XGBoost, and Gaussian Process
    
'''predictions = {
    'LSTM': y_pred_lstm,
    'XGBoost': y_pred_xgb,
    'Gaussian Prpcess': y_pred_gp,
    'Transformer': y_pred_transformer
}'''
predictions = {
    'LSTM': y_pred_lstm,
    'XGBoost': y_pred_xgb,
    'Gaussian Process': y_pred_gp
}
if 'sigma_gp' in globals():
    std_devs = {'GP': sigma_gp}

visualizer = TimeSeriesVisualizer(y_test, predictions, ticker, std_devs)

fig3 = visualizer.plot_predictions()
pdf_pages.savefig(fig3)
fig4 = visualizer.compare_models()
pdf_pages.savefig(fig4)


# In[12]:


backtester = BacktestStrategy(initial_balance=10000, transaction_cost=0.001)

results_lstm = backtester.backtest(y_test, y_pred_lstm, model_name="LSTM")
metrics_lstm = backtester.evaluate_performance(results_lstm)
results_xgb = backtester.backtest(y_test, y_pred_xgb, model_name="XGBoost")
metrics_xgb = backtester.evaluate_performance(results_xgb)
results_gp = backtester.backtest(y_test, y_pred_xgb, model_name="Gaussian Process")
metrics_gp = backtester.evaluate_performance(results_gp)


# In[13]:


import matplotlib.pyplot as plt
import numpy as np

# Generate individual figures
fig_lstm = backtester.plot_results(results_lstm, model_name="LSTM")
fig_xgb = backtester.plot_results(results_xgb, model_name="XGBoost")
fig_gp = backtester.plot_results(results_gp, model_name="Gaussian Process")

# Save figures to the PDF
pdf_pages.savefig(fig_lstm)
pdf_pages.savefig(fig_xgb)
pdf_pages.savefig(fig_gp)
print("Backtest results saved to PDF.")


# In[14]:


fig6 = utils.plot_training_history(history_lstm)  # Call only once and store result

if fig6 is not None:  # Ensure fig7 is valid before saving
    pdf_pages.savefig(fig6)
    print("Backtest results for LSTM saved to PDF (Page 6).")
else:
    print("Error: plot_training_history did not return a figure.")


# In[15]:


fig7 = xgb_model.plot_training_history(metric="rmse")
#fig7 = utils.plot_training_history(history_lstm)  # Call only once and store result

if fig7 is not None:  # Ensure fig7 is valid before saving
    pdf_pages.savefig(fig7)
    print("Backtest results for XGBoost saved to PDF (Page 7).")
else:
    print("Error: plot_training_history did not return a figure.")


# In[ ]:


#gp_model = GaussianProcessModel()
fig8 = gp_model.validate(X_train, y_train)
pdf_pages.savefig(fig8)  # ✅ Save to PDF


# In[ ]:


# Close the PDF file
pdf_pages.close()
print(f"PDF successfully closed")

#Falls nur ein Teill von Code wird ausgefuert und kein pdf erzeugt wird, sonst Fehlermeldung
pdf_pages = PdfPages("output.pdf")

if pdf_pages.get_pagecount() > 0:  # Direkter Vergleich ohne len()
    pdf_pages.close()
else:
    pdf_pages._file.close()  # Datei korrekt schließen, um die Warnung zu vermeiden

