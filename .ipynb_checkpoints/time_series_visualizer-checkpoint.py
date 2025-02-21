import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesVisualizer:
    def __init__(self, y_test, predictions_dict, ticker: str, std_devs=None):        
        """
        Initializes the TimeSeriesVisualizer.
        
        :param y_test: Actual time series values (1D array)
        :param predictions_dict: Dictionary of model name -> predicted values
        :param ticker: Stock ticker or dataset name
        :param std_devs: Dictionary of model name -> standard deviation (optional, only for Gaussian Process)
        """
        self.y_test = np.array(y_test).flatten()  # Ensure y_test is a 1D array
        self.predictions_dict = {model: np.array(pred).flatten() for model, pred in predictions_dict.items()}  # Ensure predictions are 1D
        self.std_devs = std_devs if std_devs is not None else {}
        self.ticker = ticker

        # Estimate std deviation (sigma) for models without uncertainty (LSTM, XGBoost)
        for model_name, y_pred in self.predictions_dict.items():
            if model_name not in self.std_devs:  # Only estimate if sigma is not provided
                residuals = self.y_test - y_pred  # Error between actual & predicted
                self.std_devs[model_name] = np.std(residuals)  # Standard deviation of residuals

    def plot_predictions(self):
        """
        Plots actual vs. predicted values for all models in separate subplots.
        Includes confidence intervals if available.
        """
        if not self.predictions_dict:
            print("No predictions found. Please run models first.")
            return

        # Create subplots (one for each model)
        fig, axes = plt.subplots(len(self.predictions_dict), 1, figsize=(18, 10), sharex=True)

        # Ensure axes is iterable when there's only one model
        if len(self.predictions_dict) == 1:
            axes = [axes]

        for ax, (model_name, y_pred) in zip(axes, self.predictions_dict.items()):
            sigma = self.std_devs.get(model_name, None)  # Get standard deviation if available

            ax.plot(self.y_test, label='Actual', color='blue', linewidth=2)
            ax.plot(y_pred, label=f'Predicted ({model_name})', color='red', linestyle="dashed")

            # Add confidence intervals if available
            if sigma is not None:
                sigma = np.array(sigma).flatten()  # Ensure sigma is 1D
                ax.fill_between(range(len(y_pred)), 
                                y_pred - 1.96 * sigma, 
                                y_pred + 1.96 * sigma, 
                                color='pink', alpha=0.3, label='Confidence Interval')

            ax.set_title(f'{self.ticker} Time Series Forecast: {model_name}')
            ax.set_ylabel('Stock Price')
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel('Time Steps')
        plt.tight_layout()
        plt.show()

    def compare_models(self):        
        """
        Plots all model predictions and actual values in a single figure for comparison.
        """
        plt.figure(figsize=(18, 9))
        plt.plot(self.y_test, label='Actual', color='black', linewidth=2)

        for model_name, y_pred in self.predictions_dict.items():
            plt.plot(y_pred, label=f'Predicted ({model_name})', linestyle="dashed")

        plt.title(f'{self.ticker} Comparison of Time Series Forecasts')
        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        plt.show()