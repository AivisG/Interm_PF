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

        Returns:
            fig (matplotlib.figure.Figure): The created figure.
        """
        if not self.predictions_dict:
            print("No predictions found. Please run models first.")
            return None

        # Adjust figure size to A4 landscape
        fig, axes = plt.subplots(len(self.predictions_dict), 1, figsize=(11.7, 8.3), sharex=True, gridspec_kw={'height_ratios': [1] * len(self.predictions_dict)})

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

            ax.set_title(f'{self.ticker} Time Series Forecast: {model_name}', fontsize=14)
            ax.set_ylabel('Stock Price', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True)

        axes[-1].set_xlabel('Time Steps', fontsize=12)
        fig.tight_layout()
        return fig

