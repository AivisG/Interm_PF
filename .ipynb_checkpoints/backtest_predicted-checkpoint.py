import pandas as pd
import matplotlib.pyplot as plt

class BacktestStrategyPredicted:
    def __init__(self, initial_balance=10000, transaction_cost=0.001):
        """
        Initializes the backtest strategy.

        Args:
        - initial_balance (float): Starting money in dollars (default = $10,000)
        - transaction_cost (float): Cost per trade (e.g., 0.1% default)
        """
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

    def backtest_pred(self, y_true, y_pred, model_name="Model"):
        """
        Runs a backtest using predicted and actual price data.

        Args:
        - y_true (array-like): Actual prices
        - y_pred (array-like): Predicted prices
        - model_name (str): Name of the model used for predictions

        Returns:
        - pd.DataFrame: Backtest results with balance history
        """
        balance = self.initial_balance  # Starting money
        position = 0  # Number of shares held
        balance_history = []
        buy_sell_signals = []  # 1 for buy, -1 for sell, 0 for hold

        for i in range(len(y_true) - 1):
            predicted_change = y_pred[i+1] - y_pred[i]
            
            if predicted_change > 0:  # Buy if prediction indicates an increase
                if position == 0:
                    position = balance / y_true[i]  # Buy as many shares as possible
                    balance = 0
                    buy_sell_signals.append(1)  # Buy signal
                else:
                    buy_sell_signals.append(0)  # Hold
            elif predicted_change < 0:  # Sell if prediction indicates a decrease
                if position > 0:
                    balance = position * y_true[i]  # Sell all shares
                    balance -= balance * self.transaction_cost  # Apply transaction cost
                    position = 0
                    buy_sell_signals.append(-1)  # Sell signal
                else:
                    buy_sell_signals.append(0)  # Hold
            else:
                buy_sell_signals.append(0)  # No trade

            # Portfolio balance calculation
            portfolio_value = balance + (position * y_true[i])
            balance_history.append(portfolio_value)

        # Final balance calculation
        final_value = balance + (position * y_true[-1])
        balance_history.append(final_value)

        # Store results in DataFrame
        backtest_df = pd.DataFrame({
            'Actual Price': y_true,
            'Predicted Price': y_pred,
            'Portfolio Balance': balance_history,
            'Trade Signal': [0] + buy_sell_signals  # Shift for alignment
        })

        print(f"Backtest for {model_name} completed. Final balance: ${final_value:.2f}")
        return backtest_df
    
    def plot_results_pred(self, df, model_name="Model"):
        """
        Plots the backtest results and returns the figure for saving to PDF.

        Args:
        - df (pd.DataFrame): Backtest results DataFrame
        - model_name (str): Name of the model
        
        Returns:
        - fig (matplotlib.figure.Figure): The figure object for saving to PDF.
        """
        fig, ax1 = plt.subplots(figsize=(22,12))

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price", color="blue")
        ax1.plot(df.index, df["Actual Price"], label="Actual Price", color="blue")
        ax1.plot(df.index, df["Predicted Price"], label="Predicted Price", color="orange", linestyle="dashed")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Balance", color="green")
        ax2.plot(df.index, df["Portfolio Balance"], label="Portfolio Balance", color="green")
        ax2.legend(loc="upper right")

        plt.title(f"Backtest Results for {model_name}")
        plt.show()  # ✅ Display the plot on screen

        return fig  # ✅ Return the figure so it can be saved in PDF
    
    def evaluate_performance_pred(self, df):
        """
        Evaluates backtest performance using key financial metrics.

        Args:
        - df (pd.DataFrame): Backtest results DataFrame

        Returns:
        - dict: Performance metrics including final balance, total return, Sharpe ratio, max drawdown.
        """
        final_balance = df["Portfolio Balance"].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        returns = df["Portfolio Balance"].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std()
        max_drawdown = (df["Portfolio Balance"].cummax() - df["Portfolio Balance"]).max() / df["Portfolio Balance"].cummax().max()

        metrics = {
            "Final Balance": final_balance,
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }

        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        return metrics
