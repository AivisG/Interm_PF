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
        # ✅ Adjust figure size to full A4 landscape
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.7, 8.3), sharex=True)

        # Closing price plot
        ax1.plot(self.stock_data['Date'], self.stock_data['Close'], label="Closing Price", color='blue')
        ax1.set_title(f"{self.ticker} Closing Price Over Time", fontsize=14)
        ax1.set_ylabel("Price in USD", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True)

        # RSI plot
        ax2.plot(self.stock_data['Date'], self.stock_data['RSI'], color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', label="Overbought (70)")
        ax2.axhline(y=30, color='g', linestyle='--', label="Oversold (30)")
        ax2.set_title(f'{self.ticker} RSI (Relative Strength Index)', fontsize=14)
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True)

        # MACD plot
        ax3.plot(self.stock_data['Date'], self.stock_data['MACD'], label='MACD', color='black')
        ax3.plot(self.stock_data['Date'], self.stock_data['Signal'], label='Signal Line', color='orange')
        ax3.legend(fontsize=11)
        ax3.set_title(f'{self.ticker} MACD (Moving Average Convergence Divergence)', fontsize=14)
        ax3.set_xlabel("Date", fontsize=12)
        ax3.set_ylabel('Value', fontsize=12)
        ax3.grid(True)

        # ✅ Adjust spacing between subplots to optimize height
        fig.subplots_adjust(bottom=0.35, hspace=0.3)

        # ✅ Add explanatory text **outside the figure but inside the A4 page**
        plt.figtext(0.15, 0.01,  
                    "**Understanding This Chart:**\n\n"
                    "- **Closing Price**: The daily closing price trend over time.\n"
                    "- **RSI (Relative Strength Index)**: Identifies overbought (>70) and oversold (<30) conditions.\n"
                    "- **MACD (Moving Average Convergence Divergence)**: A trend-following momentum indicator.\n\n"
                    "**Use this analysis to identify trading opportunities and assess market trends.**",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))
        
        plt.tight_layout()
        return fig   
