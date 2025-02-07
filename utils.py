import numpy as np
from sklearn.preprocessing import StandardScaler
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

def add_technical_indicators(df):
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
    return df

def prepare_features(df):
    # Wähle numerische Spalten
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    print(numeric_features)
    # Standardisiere die Features
    scaler = StandardScaler()   
    # Create scaled DataFrames
    df_standardized = df.copy()
    # Apply StandardScaler
    df_standardized[numeric_features]=scaler.fit_transform(df[numeric_features])
    return df_standardized, scaler

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
    os.makedirs(subfolder, exist_ok=True)  # ✅ Ensure the folder exists
    pdf_path = os.path.join(subfolder, filename)
    return PdfPages(pdf_path), pdf_path  # ✅ Return PdfPages object and file path

def save_plot_to_pdf(pdf_pages, fig):
    """Save a matplotlib figure to the provided PdfPages object."""
    if pdf_pages:
        pdf_pages.savefig(fig)  # ✅ Save the plot
        print("Plot saved to PDF.")
    else:
        print("Error: pdf_pages is None. Cannot save plot.")

def augment_data(X, y, shift_range=3):
    X_aug, y_aug = [], []
    for shift in range(-shift_range, shift_range + 1):
        if shift == 0: continue  # Skip zero shift
        X_shifted = np.roll(X, shift, axis=0)
        X_aug.append(X_shifted)
        y_aug.append(y)
    return np.vstack(X_aug), np.hstack(y_aug)
