import pandas as pd
import matplotlib.pyplot as plt

class BacktestStrategy:
    def __init__(self, strategy_name: str, ticker, initial_balance=10000, output_file="backtest_results.xlsx"):
        self.strategy_name = strategy_name
        self.initial_balance = initial_balance
        self.output_file = output_file
        self.ticker = ticker

    def run_backtest(self, df, name):
        """
        Run a backtesting strategy using df_scaled (already processed data) and save the results.
        """
        # Ensure data is sorted correctly (from oldest to newest)
        df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

        results = df.copy()
        results['Position'] = 0  
        results['Portfolio Value'] = float(self.initial_balance)
        results['Cash'] = float(self.initial_balance)
        results['Shares Held'] = float(0)

        cash = float(self.initial_balance)
        shares_held = float(0)
        entry_price = 0  

        results['Short_MA'] = results[name].rolling(window=5, min_periods=1).mean()
        results['Long_MA'] = results[name].rolling(window=20, min_periods=1).mean()
        
        start_trading_index = 20 if len(results) > 20 else 0  

        trade_history = []

        for i in range(start_trading_index, len(results)):
            rsi = results['RSI'].iloc[i]
            macd = results['MACD'].iloc[i]
            signal = results['Signal'].iloc[i]
            short_ma = results['Short_MA'].iloc[i]
            long_ma = results['Long_MA'].iloc[i]
            close_price = results[name].iloc[i]

            trade_action = "HOLD"  

            # Only Buy If RSI is Rising AND Price Has Increased in the Last 3 Days
            if rsi > results['RSI'].iloc[i-1] and close_price > results[name].iloc[i-3] and macd > signal and short_ma > long_ma:
                if cash > 0:  
                    shares_to_buy = cash / close_price
                    shares_held += shares_to_buy
                    cash = 0  
                    entry_price = close_price  
                    results.at[results.index[i], 'Position'] = 1  
                    trade_action = "BUY"

            # Only Sell If the Price Has Grown at Least 2% Since Buying
            elif shares_held > 0 and close_price > entry_price * 1.02 and macd < signal and short_ma < long_ma:
                cash = shares_held * close_price
                shares_held = 0  
                entry_price = 0  
                results.at[results.index[i], 'Position'] = -1  
                trade_action = "SELL"

            portfolio_value = cash + (shares_held * close_price)
            results.at[results.index[i], 'Portfolio Value'] = float(portfolio_value)
            results.at[results.index[i], 'Cash'] = float(cash)
            results.at[results.index[i], 'Shares Held'] = float(shares_held)

            trade_history.append({
                "Date": results['Date'].iloc[i],
                name: close_price,
                "RSI": rsi,
                "MACD": macd,
                "Signal": signal,
                "Short_MA": short_ma,
                "Long_MA": long_ma,
                "Trade Action": trade_action,
                "Shares Held": shares_held,
                "Cash": cash,
                "Portfolio Value": portfolio_value
            })

        trade_df = pd.DataFrame(trade_history)
        trade_df.to_excel(self.output_file, index=False)

        print(f"Backtest results saved to {self.output_file}")

        return results


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
        plt.tight_layout()
        return fig
