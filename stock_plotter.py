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

    def plot_closing_price(self):
        """
        Plot the closing price over time.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.stock_data['Date'], self.stock_data['Close'], label="Closing Price", color='blue')
        ax.set_title(f"{self.ticker} Closing Price Over Time", fontsize=14)
        ax.set_ylabel("Price in USD", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True)
        ax.set_xlabel("Date", fontsize=12)
        plt.figtext(0.1, 0.01,  
                    "Understanding This Chart:\n\n"
                    "- The blue line represents the closing price trend over time.\n"
                    "- Use this chart to identify price movements and trends in the stock market.\n",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))
        plt.tight_layout()
        return fig

    def plot_rsi(self):
        """
        Plot the Relative Strength Index (RSI) over time.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.stock_data['Date'], self.stock_data['RSI'], color='purple')
        ax.axhline(y=70, color='r', linestyle='--', label="Overbought (70)")
        ax.axhline(y=30, color='g', linestyle='--', label="Oversold (30)")
        ax.set_title(f'{self.ticker} RSI (Relative Strength Index)', fontsize=14)
        ax.set_ylabel('RSI', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True)
        ax.set_xlabel("Date", fontsize=12)
        plt.figtext(0.1, 0.01,  
                    "Understanding This Chart:\n\n"
                    "- The purple line represents the RSI over time.\n"
                    "- The red and green dashed lines indicate overbought and oversold levels.\n"
                    "- Use this chart to identify potential market reversals.\n",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))
        plt.tight_layout()
        return fig

    def plot_macd(self):
        """
        Plot the MACD and Signal Line over time.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.stock_data['Date'], self.stock_data['MACD'], label='MACD', color='black')
        ax.plot(self.stock_data['Date'], self.stock_data['Signal'], label='Signal Line', color='orange')
        ax.set_title(f'{self.ticker} MACD (Moving Average Convergence Divergence)', fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True)
        plt.figtext(0.1, 0.01,  
                    "Understanding This Chart:\n\n"
                    "- The black line represents the MACD indicator.\n"
                    "- The orange line is the Signal Line, helping to confirm trends.\n"
                    "- Use this chart to identify bullish and bearish signals.\n",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))
        plt.tight_layout()
        return fig

    def plot_backtest_results(self, results):
        """
        Plot backtest results with realistic portfolio calculations.
        """
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(results['Date'], results['Portfolio Value'], label="Portfolio Value", color='green')
        ax.set_title(f"{self.ticker} Backtest Results", fontsize=14)
        ax.set_ylabel("Portfolio Value", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True)
        ax.set_xlabel("Date", fontsize=12)
        plt.figtext(0.1, 0.9,  
                     "**Understanding This Chart:**\n\n"
                "- The green line represents the portfolio value over time.\n"
                "- This backtest simulates an automated trading strategy based on technical indicators.\n\n"
                "**Trading Rules Applied:**\n"
                "- Buy when RSI is rising and price is higher than 3 days ago (ensures uptrend confirmation).\n"
                "- Sell only if price has increased by at least 2% since buying (prevents premature exits).\n"
                "- MACD and Moving Averages are used for additional trade confirmation.\n\n"
                "**Key Notes:**\n"
                "- Portfolio value changes based on executed trades.\n"
                "- The system avoids unnecessary trades by ensuring multiple confirmations before buying or selling.\n"
                "- Unlike basic strategies, this approach prevents buying in a downtrend and avoids frequent stop-outs.\n\n"
                "This backtest provides a realistic view of potential trading performance, but real-world trading includes slippage, execution delays, and psychological factors.",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))
        plt.tight_layout()
        return fig
