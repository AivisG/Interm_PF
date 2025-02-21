import pandas as pd
import matplotlib.pyplot as plt

class StockPlotter:
    def __init__(self, stock_data: pd.DataFrame, ticker: str):
        """
        Initialize the StockPlotter class.

        Parameters:
            stock_data (pd.DataFrame): Stock data to plot.
            ticker (str): The stock ticker symbol.
        """
        self.stock_data = stock_data.copy()
        self.ticker = ticker
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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
        
        # Closing price plot
        ax1.plot(self.stock_data['Date'], self.stock_data['Close'], label="Closing Price", color='blue')
        ax1.set_title(f"{self.ticker} Closing Price Over Time")
        ax1.set_ylabel("Price in USD")
        ax1.legend()
        ax1.grid(True)
        
        # RSI plot
        ax2.plot(self.stock_data['Date'], self.stock_data['RSI'], color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', label="Overbought (70)")
        ax2.axhline(y=30, color='g', linestyle='--', label="Oversold (30)")
        ax2.set_title(f'{self.ticker} RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)
        
        # MACD plot
        ax3.plot(self.stock_data['Date'], self.stock_data['MACD'], label='MACD', color='black')
        ax3.plot(self.stock_data['Date'], self.stock_data['Signal'], label='Signal Line', color='orange')
        ax3.legend()
        ax3.set_title(f'{self.ticker} MACD (Moving Average Convergence Divergence)')
        ax3.set_xlabel("Date")
        ax3.set_ylabel('Value')
        ax3.grid(True)

         fig.text(0.1, -0.2, 
             f"This plot displays the backtest results for {model_name}.\n"
             "It shows the portfolio balance changes over time based on trade decisions.\n"
             "Peaks indicate periods of high returns, while dips represent drawdowns.\n"
             "Use this to evaluate strategy effectiveness and risk management.",
             fontsize=12, wrap=True)

        
        plt.tight_layout()
        return fig  