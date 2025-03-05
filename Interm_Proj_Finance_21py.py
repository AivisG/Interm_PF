import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from pypdf import PdfReader, PdfWriter
from sklearn.model_selection import train_test_split
import pandas_datareader.data as web
import datetime

import utils
from stock_plotter import StockPlotter
from models import Models
from model_evaluator import ModelEvaluator
from time_series_visualizer import TimeSeriesVisualizer
from backtest import BacktestStrategy
from xgb_model import XGBoostModel
from xgb_model_val import XGBoostModelVal

'''--------------------
Get data from yahoo finance 
--------------------'''
# #To get data from yfinance:
# ticker = "AAPL"
# start = "2020-01-01"
# end = "2025-01-01"
# interval = "1d"
# 
# stock_fetcher = stock_data_fetcher.StockDataFetcher(ticker, start, end, interval)
# stock_data = stock_fetcher.fetch_data()

'''--------------------
Get data from pre-loaded csv file
--------------------'''
#ticker = "JPM"
#start = "2024-01-01"
#end = "2025-01-01"
#stock_data = utils.get_csv(ticker, start, end)

'''--------------------
Get data from pandas datareader stooq
--------------------'''
start = datetime.datetime(2024, 1, 1)
end = datetime.datetime(2025, 1, 1)
ticker = "JPM"

# Fetch AAPL data from Stooq (since Google Finance is not available in pandas_datareader)
stock_data = web.DataReader(ticker, "stooq", start, end)

stock_data = stock_data.reset_index()
#df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime
#df = df.sort_values(by="Date", ascending=True)  # Sort correctly

'''--------------------
Data preparation
--------------------'''
df = utils.add_technical_indicators(stock_data, "Close") #df not scaled
df = df.dropna()
print(df)

df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime
df = df.sort_values(by="Date", ascending=True)  # Sort correctly

df_scaled, scaler = utils.prepare_features(df) # Returns scaled df
sequence_length = 15
X, y = utils.create_sequences(df_scaled[19:], sequence_length) #returns sequences ndarray

train_size = int(len(X) * 0.8) #splits in train and test
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

price_test_start = stock_data.iloc[(len(X_test))]["Close"]
date_test_start = stock_data.iloc[(len(X_test))]["Date"]
print(price_test_start)
print(date_test_start)

'''--------------------
PDF path
--------------------'''
# Fix filename by formatting dates without spaces or colons
start_str = start.strftime("%Y-%m-%d")  # Format as 'YYYY-MM-DD'
end_str = end.strftime("%Y-%m-%d")      # Same for end date

pdf_path = f"results/results_{ticker}_{start_str}_to_{end_str}.pdf"
pdf_pages = PdfPages(pdf_path)

'''--------------------
Data plot
--------------------'''
plotter = StockPlotter(stock_data, ticker, start, end)
fig1_3 = plotter.plot()
pdf_pages.savefig(fig1_3)  # Save stock plot to PDF

'''--------------------
Backtest plot
--------------------'''
# Initialize Backtest Strategy
backtest_strategy = BacktestStrategy("Advanced Strategy", ticker)

# Generate backtest results
backtest_results = backtest_strategy.run_backtest(df, "Close")
print(backtest_results)

# Get the figure without showing it
fig4 = backtest_strategy.plot_backtest_results(backtest_results)

# Save the figure to PDF
pdf_pages.savefig(fig4)
plt.show()

print(backtest_results["Portfolio Value"].iloc[-1])

portfolio_test_start = backtest_results.iloc[(-len(X_test)+1)]["Portfolio Value"]
date_test_start = backtest_results.iloc[-(len(X_test)+1)]["Date"]
print(portfolio_test_start)
print(date_test_start)

'''--------------------
 LSTM model 
--------------------'''
# Ensure correct feature dimension
n_features = X_train.shape[2]  # 11 features

lstm_model = Models.LSTM_Model(sequence_length, n_features)  # Initialize model

# Tune hyperparameters and create the model
best_params = lstm_model.tune_hyperparameters(X_train, y_train)

# Train the model and store training history
history_lstm = lstm_model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

# Ensure X_test is of the correct dtype
X_test = np.array(X_test, dtype=np.float32)

# Make predictions and flatten output
y_pred_lstm = lstm_model.predict(X_test).flatten()
lstm_model.summary()

# Create the plot
fig5, ax = plt.subplots(figsize=(8, 6))
ax.plot(history_lstm.history['loss'], label='Training Loss', color='blue')
ax.plot(history_lstm.history['val_loss'], label='Validation Loss', color='orange', linestyle='dashed')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Fig 5 LSTM Training Loss vs Validation Loss")

# Save the figure to PDF without closing
pdf_pages.savefig(fig5)

# Show the plot
plt.show()

all_features = df.columns
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
features_10 = df.select_dtypes(include=['float64', 'int64']).drop(columns=["Close"]).columns

# Drop "Close" and "Date" to create X (features)
X = df.drop(columns=["Close", "Date"]).values  

# Set y (target) as "Close" column
y = df["Close"].values  


'''--------------------
 XGB model
--------------------'''
xgb_model = XGBoostModel(n_estimators=200, max_depth=5, learning_rate=0.05)

xgb_model.fit(X_train, y_train, verbose=False)

xgb_model.evaluate(X_test, y_test)

y_pred_xgb = xgb_model.predict(X_test)

fig = xgb_model.plot_training_history()
plt.show()  

'''--------------------
 XGB model with validation for plot
--------------------'''
# Split data into train (70%), validation (15%), and test (15%)
X_train_val, X_temp, y_train_val, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test_val, y_val, y_test_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_val = pd.DataFrame(X_train_val, columns=features_10)  # Replace `feature_names` with actual feature names
X_test_val = pd.DataFrame(X_test_val, columns=features_10)

# Initialize the model with validation support
xgb_model_val = XGBoostModelVal(n_estimators=200, max_depth=5, learning_rate=0.05, early_stopping_rounds=10)

# Fit the model using both training and validation data
xgb_model_val.fit(X_train_val, y_train_val, X_val=X_val, y_val=y_val, verbose=True)

# Evaluate the model on test data
xgb_model_val.evaluate(X_test_val, y_test_val)

# Make predictions
y_pred_xgb_val = xgb_model_val.predict(X_test_val)

# Plot training history including validation performance
fig6 = xgb_model_val.plot_training_history()
plt.show()

pdf_pages.savefig(fig6)

'''--------------------
 Gaussian model 
--------------------'''
gp_model = Models.GP_Model()
gp_model.fit(X_train, y_train)

# Predict with confidence intervals
y_pred_gp, sigma_gp = gp_model.predict(X_test, return_std=True)
#gp_model = GaussianProcessModel()
fig7 = gp_model.validate(X_train, y_train)
pdf_pages.savefig(fig7)  

'''--------------------
 Model evaluation
--------------------'''
models = {
    'LSTM': lstm_model,
    'XGBoost': xgb_model,
    'Gaussian Process': y_pred_gp
}
evaluator = ModelEvaluator(models, X_test, y_test, ticker)

evaluation_results, y_predictions = evaluator.evaluate()

# Print results
for model, metrics in evaluation_results.items():
    print(f"\nModel: {model}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

fig8 = evaluator.plot_metrics()
pdf_pages.savefig(fig8)
print(len(X_test))

'''--------------------
 Model visualization
--------------------'''
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

fig9_11 = visualizer.plot_predictions()
pdf_pages.savefig(fig9_11)

# Close the PDF file
pdf_pages.close()

# Falls nur ein Teill von Code wird ausgefuert und kein pdf erzeugt wird, sonst Fehlermeldung
pdf_pages = PdfPages("output.pdf")

if pdf_pages.get_pagecount() > 0:  # Direkter Vergleich ohne len()
    pdf_pages.close()
else:
    pdf_pages._file.close()  # Datei korrekt schließen, um die Warnung zu vermeiden

'''--------------------
 Add explanatory text to PDF file
--------------------'''
# Define file paths
main_pdf = pdf_path  # The PDF created in your code
extra_pdf = "text_fig.pdf"  # The external PDF you want to add
output_pdf = pdf_path  # The final merged output

# Ensure the files exist
if not os.path.exists(main_pdf):
    print(f"Error: {main_pdf} does not exist.")
    exit()
if not os.path.exists(extra_pdf):
    print(f"Error: {extra_pdf} does not exist.")
    exit()

# Open both PDFs
reader_main = PdfReader(main_pdf)
reader_extra = PdfReader(extra_pdf)
writer = PdfWriter()

# Copy all pages from the main PDF
for page in reader_main.pages:
    writer.add_page(page)

# Append all pages from the extra PDF
for page in reader_extra.pages:
    writer.add_page(page)

# Save the final merged PDF
with open(output_pdf, "wb") as output_file:
    writer.write(output_file)

print(f"Successfully merged! Final PDF saved as: {output_pdf}")

