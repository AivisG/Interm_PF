import pandas as pd
import yfinance as yf

class StockDataFetcher:
    def __init__(self, ticker, start, end, interval):
        """Initialize the StockDataFetcher class."""
        self.ticker = ticker  # Stock ticker symbol (e.g., "AAPL").
        self.start = start # Start date (YYYY-MM-DD).
        self.end = end # End date (YYYY-MM-DD).
        self.interval = interval # Data interval (default: "1d").
        self.stock_data = None
                
    def fetch_data(self):
        """Download stock data from Yahoo Finance."""
        self.stock_data = yf.download(self.ticker, start=self.start, end=self.end, interval=self.interval)
        self.clean_data()
        return self.stock_data
    
    def clean_data(self):
        """Clean the downloaded stock data."""
        if self.stock_data is None or self.stock_data.empty:
            return

        self.stock_data.reset_index(inplace=True)

        if isinstance(self.stock_data.columns, pd.MultiIndex):
            self.stock_data.columns = ['_'.join(filter(None, col)).strip() if isinstance(col, tuple) else col for col in self.stock_data.columns]

        self.stock_data.columns = [col.replace(f'_{self.ticker}', '').strip() for col in self.stock_data.columns]

        if 'Date' not in self.stock_data.columns:
            self.stock_data.rename(columns={self.stock_data.columns[0]: 'Date'}, inplace=True)

        self.stock_data.set_index('Date', inplace=True)
        self.stock_data.index = pd.to_datetime(self.stock_data.index)

    def get_data(self):
        """Return the cleaned stock data."""
        return self.stock_data