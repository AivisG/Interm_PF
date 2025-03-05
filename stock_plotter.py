import pandas as pd
import matplotlib.pyplot as plt

class StockPlotter:
    def __init__(self, stock_data: pd.DataFrame, ticker: str, start, end):
        """
        Initialize the StockPlotter class.

        Parameters:
            stock_data (pd.DataFrame): Stock data to plot.
            ticker (str): The stock ticker symbol.
        """
        self.stock_data = stock_data.copy()
        self.ticker = ticker
        self.start = start
        self.end = end
        self._prepare_data()

    def _prepare_data(self):
        """
        Ensure the 'Date' column is in datetime format and reset the index if necessary.
        """
        if 'Date' not in self.stock_data.columns and self.stock_data.index.name == 'Date':
            self.stock_data.reset_index(inplace=True)
        
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])

    def plot(self):
        """
        Create subplots for stock closing price, RSI, and MACD.

        Returns:
            fig (matplotlib.figure.Figure): The created matplotlib figure.
        """
        # Adjust figure size to A4 landscape
        fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

        # Closing price plot
        axes[0].plot(self.stock_data['Date'], self.stock_data['Close'], label="Closing Price", color='blue')
        axes[0].set_title(f"Fig 1 {self.ticker} Closing Price Over Time, {self.start} - {self.end}.", fontsize=14)
        axes[0].set_ylabel("Price in USD", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True)

        # RSI plot
        axes[1].plot(self.stock_data['Date'], self.stock_data['RSI'], color='purple')
        axes[1].axhline(y=70, color='r', linestyle='--', label="Overbought (70)")
        axes[1].axhline(y=30, color='g', linestyle='--', label="Oversold (30)")
        axes[1].set_title(f'Fig 2 {self.ticker} RSI (Relative Strength Index)', fontsize=14)
        axes[1].set_ylabel('RSI', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True)

        # MACD plot
        axes[2].plot(self.stock_data['Date'], self.stock_data['MACD'], label='MACD', color='black')
        axes[2].plot(self.stock_data['Date'], self.stock_data['Signal'], label='Signal Line', color='orange')
        axes[2].legend(fontsize=11)
        axes[2].set_title(f'Fig 3 {self.ticker} MACD (Moving Average Convergence Divergence)', fontsize=14)
        axes[2].set_xlabel("Date", fontsize=12)
        axes[2].set_ylabel('Value', fontsize=12)
        axes[2].grid(True)

        # Adjust spacing between subplots
        fig.tight_layout()
        
        return fig
