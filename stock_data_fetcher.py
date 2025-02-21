import pandas as pd
import requests

class StockDataFetcher:
    def __init__(self, ticker, start, end, interval="1d"):
        """Initialize the StockDataFetcher class."""
        self.ticker = ticker  # Stock ticker symbol (e.g., "AAPL").
        self.start = start  # Start date (YYYY-MM-DD).
        self.end = end  # End date (YYYY-MM-DD).
        self.interval = interval  # Data interval (default: "1d").
        self.stock_data = None

    def fetch_data(self):
        """Fetch stock data using Yahoo Finance CSV download (No API needed)."""
        try:
            # Convert start and end dates to UNIX timestamps
            start_timestamp = int(pd.Timestamp(self.start).timestamp())
            end_timestamp = int(pd.Timestamp(self.end).timestamp())

            url = (
                f"https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}"
                f"?period1={start_timestamp}&period2={end_timestamp}"
                f"&interval=1d&events=history&includeAdjustedClose=true"
            )

            # Download CSV file
            self.stock_data = pd.read_csv(url)
            self.clean_data()
            return self.stock_data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def clean_data(self):
        """Clean the downloaded stock data."""
        if self.stock_data is None or self.stock_data.empty:
            print("No data available to clean.")
            return

        # Convert 'Date' column to datetime and set it as index
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.stock_data.set_index('Date', inplace=True)

        print("Cleaned Data:\n", self.stock_data.head())  # Debugging output

    def get_data(self):
        """Return the cleaned stock data."""
        return self.stock_data
