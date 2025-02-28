import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
from backtest import BacktestStrategy

class StockPrediction:
    def __init__(self, y_pred, price_test_start, date_test_start, ticker, portfolio_test_start, module_name):
        self.y_pred = y_pred
        self.price_test_start = price_test_start
        self.date_test_start = date_test_start
        self.ticker = ticker
        self.portfolio_test_start = portfolio_test_start
        self._calculate_predicted_prices()
        self.module_name = module_name

    def _calculate_predicted_prices(self):
        """Calculate predicted prices by applying a scaling coefficient."""
        self.koeffizient = self.price_test_start / self.y_pred[0]
        self.price_predicted = self.y_pred * self.koeffizient
        self.df_price_predicted = pd.DataFrame(self.price_predicted, columns=["Predicted Close"])
        self.df_price_predicted["Date"] = pd.date_range(start=self.date_test_start, periods=len(self.df_price_predicted), freq="D")

    def run_backtest(self):
        """Run backtest using the predicted prices."""
        df_2 = utils.add_technical_indicators(self.df_price_predicted, "Predicted Close")
        backtest_strategy = BacktestStrategy("Advanced Strategy", self.ticker, self.portfolio_test_start)
        backtest_results = backtest_strategy.run_backtest(df_2, "Predicted Close")
        return backtest_results

    def plot_predictions_and_backtest(self, backtest_results):
        """Plot both predicted prices and backtest results in a single figure."""
        fig, ax1 = plt.subplots(figsize=(20, 10))

        # Plot predicted prices
        ax1.plot(self.df_price_predicted["Date"], self.df_price_predicted["Predicted Close"], label="Predicted Price", color="blue", linewidth=2)
        ax1.set_xlabel("Time Steps", fontsize=12)
        ax1.set_ylabel("Predicted Price", fontsize=12, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        # Plot backtest portfolio value
        ax2 = ax1.twinx()
        ax2.plot(backtest_results['Date'], backtest_results['Portfolio Value'], label="Portfolio Value", color='green', linewidth=2)
        ax2.set_ylabel("Portfolio Value", fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Titles and legends
        fig.suptitle(f"Predicted Stock Price and Backtest Portfolio Value, {self.module_name}", fontsize=14)
        fig.legend(loc="upper left")
        plt.show()
        
# Example usage:
# prediction = StockPrediction(y_pred_lstm, price_test_start, "2024-06-05", ticker, portfolio_test_start)
# backtest_results = prediction.run_backtest()
# prediction.plot_predictions_and_backtest(backtest_results)
