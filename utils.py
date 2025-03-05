import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from matplotlib.backends.backend_pdf import PdfPages

def calculate_rsi(close_prices, window=14):
    
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0) 
    loss = -delta.where(delta < 0, 0) 
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(close_prices, short_window=12, long_window=26, signal_window=9):
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()  
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()    
    macd = short_ema - long_ema  # MACD line
    signal = macd.ewm(span=signal_window, adjust=False).mean()  # Signal line
    return macd, signal

def calculate_bollinger_bands(close_prices, window=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window=window).mean()  
    rolling_std = close_prices.rolling(window=window).std()  
    upper_band = rolling_mean + (num_std_dev * rolling_std)  # Upper band
    lower_band = rolling_mean - (num_std_dev * rolling_std)  # Lower band
    return upper_band, rolling_mean, lower_band

def add_technical_indicators(df, name):
    df['RSI'] = calculate_rsi(df[name], window=14)
    df['MACD'], df['Signal'] = calculate_macd(df[name])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df[name])
    return df

def prepare_features(df):
    # Wähle numerische Spalten
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    print(numeric_features)
    # Standardisiere die Features
    #scaler = StandardScaler()   
    scaler = MinMaxScaler()
    # Create scaled DataFrames
    df_scaled = df.copy()
    # Apply StandardScaler
    df_scaled[numeric_features]=scaler.fit_transform(df[numeric_features])
    return df_scaled, scaler

def create_sequences(data, seq_length):
    sequences = []    
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:(i + seq_length), 1:]  # Skip the first column here
        target = data.iloc[i + seq_length]['Close']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def minmax_scaler(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train_scaled, X_test_scaled

def create_pdf(subfolder, filename):
    """Create a PdfPages object and return it, ensuring the folder exists."""
    os.makedirs(subfolder, exist_ok=True)  # Ensure the folder exists
    pdf_path = os.path.join(subfolder, filename)
    return PdfPages(pdf_path), pdf_path  # Return PdfPages object and file path

def save_plot_to_pdf(pdf_pages, fig):
    """Save a matplotlib figure to the provided PdfPages object."""
    if pdf_pages:
        pdf_pages.savefig(fig)  # Save the plot
        print("Plot saved to PDF.")
    else:
        print("Error: pdf_pages is None. Cannot save plot.")

def plot_training_history(history, title="Training vs. Validation Loss LSTM"):
    """
    Plots the training and validation loss from a model's history object.

    Parameters:
    history : keras.callbacks.History
        The history object returned by model.fit().
    title : str, optional
        The title of the plot (default is "Training vs. Validation Loss").

    Returns:
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axis
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title(title)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    return fig  # Return the figure object

def get_csv(ticker, start_date, end_date):
    directory = "stock_data_csv"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{ticker}_{start_date}_to_{end_date}.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} successfully!")
        return df
    else:
        print(f"File {file_path} does not exist.")
        return None
 
