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
        Plots actual vs. predicted values for each model in separate figures.

        Returns:
            List of figures (list of matplotlib.figure.Figure objects)
        """
        if not self.predictions_dict:
            print("No predictions found. Please run models first.")
            return []

        figures = []
        for model_name, y_pred in self.predictions_dict.items():
            fig, ax = plt.subplots(figsize=(11.7, 5.8))
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
            ax.set_xlabel('Time Steps', fontsize=12)

            plt.figtext(0.1, -0.1,  
                        f"**Understanding This Chart ({model_name}):**\n\n"
                        "- **Blue Line (Actual)**: Represents true stock price movements.\n"
                        "- **Red Dashed Line (Predicted)**: Model’s predicted values.\n"
                        "- **Pink Shaded Area (Confidence Interval)**: Represents prediction uncertainty (if available).\n\n"
                        " **Use this visualization to evaluate model accuracy and prediction confidence levels.**",
                        fontsize=12, ha="left", 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))

            plt.tight_layout()
            figures.append(fig)
        
        return figures

    def compare_models(self):        
        """
        Plots all model predictions and actual values in a single figure for comparison.

        Returns:
            fig (matplotlib.figure.Figure): The created figure.
        """
        fig, ax = plt.subplots(figsize=(11.7, 5.8))
        ax.plot(self.y_test, label='Actual', color='black', linewidth=2)

        for model_name, y_pred in self.predictions_dict.items():
            ax.plot(y_pred, label=f'Predicted ({model_name})', linestyle="dashed")

        ax.set_title(f'{self.ticker} Comparison of Time Series Forecasts', fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Stock Price', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True)

        plt.figtext(0.1, -0.1,  
                    "**Comparing Different Forecast Models:**\n\n"
                    "- **Black Line (Actual)**: Represents the true stock price movement.\n"
                    "- **Dashed Lines (Predictions)**: Each model’s forecast.\n\n"
                    "  **Use this comparison to identify the best-performing model for forecasting stock prices.**",
                    fontsize=12, ha="left", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))

        plt.tight_layout()
        return fig
