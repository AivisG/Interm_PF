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
                    "Understanding This Chart:\n\n"
                    "- The green line represents the portfolio value over time.\n"
                    "- This chart shows how an investment strategy performed using simulated trades.\n"
                    "- The strategy follows advanced trading rules:\n"
                    "  - Buy when RSI is below 30 (oversold) and sell when RSI is above 70 (overbought).\n"
                    "  - Confirm trades with MACD: Buy when MACD crosses above the Signal Line, sell when it crosses below.\n"
                    "  - Use a moving average crossover: Buy when the short-term moving average crosses above the long-term moving average.\n"
                    "  - Apply stop-loss and take-profit levels to manage risk.\n"
                    "  - Consider volume-based confirmation: Enter trades when high volume supports the trend.\n"
                    "- Portfolio value is calculated based on these trading actions rather than simple cumulative sums of stock prices.\n"
                    "This backtest gives an optimal view of how a trader could perform if all conditions align perfectly.\n"
                    "However, real-world trading involves uncertainties, execution challenges, and external market factors.\n",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))
        plt.tight_layout()
        return fig
