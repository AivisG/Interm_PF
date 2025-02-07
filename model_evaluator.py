import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

class ModelEvaluator:
    def __init__(self, models, X_test, y_test, ticker):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.ticker = ticker

    def evaluate(self):
        """Evaluate models and return metrics."""
        evaluation_results = {}
        y_predictions = {}

        for model_name, model in self.models.items():
            if model is None:
                print(f"Skipping evaluation for {model_name} (no trained model).")
                continue

            # Get model predictions
            y_pred = model.predict(self.X_test)
            y_predictions[model_name] = y_pred

            # Compute metrics
            mse = np.mean((y_pred - self.y_test) ** 2)  # Mean Squared Error
            mae = np.mean(np.abs(y_pred - self.y_test))  # Mean Absolute Error
            r2 = r2_score(self.y_test, y_pred)  # R² Score
            direction = np.mean((np.sign(y_pred[1:] - y_pred[:-1]) ==
                                 np.sign(self.y_test[1:] - self.y_test[:-1])))  # Directional Accuracy

            evaluation_results[model_name] = {
                "MSE": mse,
                "MAE": mae,
                "R²": r2,
                "Direction": direction
            }

        return evaluation_results, y_predictions

    def plot_metrics(self):
        """Generate a bar plot of model evaluation metrics with grid and values displayed on the bars."""
        evaluation_results, _ = self.evaluate()  # Evaluate the models internally

        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for plotting
        model_names = []
        mse_values = []
        mae_values = []
        r2_values = []
        direction_values = []

        for model_name, metrics in evaluation_results.items():
            model_names.append(model_name)
            mse_values.append(metrics.get("MSE", 0))
            mae_values.append(metrics.get("MAE", 0))
            r2_values.append(metrics.get("R²", 0))
            direction_values.append(metrics.get("Direction", 0))

        # Plot grouped bar charts
        x = np.arange(len(model_names))  # The label locations
        width = 0.2  # The width of the bars

        bars_mse = ax.bar(x - 3*width/2, mse_values, width, label="MSE", color='blue')
        bars_mae = ax.bar(x - width/2, mae_values, width, label="MAE", color='orange')
        bars_r2 = ax.bar(x + width/2, r2_values, width, label="R²", color='green')
        bars_direction = ax.bar(x + 3*width/2, direction_values, width, label="Direction", color='purple')

        # Add grid lines
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # ✅ Add horizontal grid lines
        ax.xaxis.grid(False)  # Disable vertical grid lines

        # Add labels and title
        ax.set_xlabel("Models")
        ax.set_ylabel("Metric Values")
        ax.set_title(f"Model Evaluation Metrics for {self.ticker}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()

        # Add values on top of bars
        def add_bar_values(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.4f}',
                        ha='center', va='bottom', fontsize=9)

        add_bar_values(bars_mse)
        add_bar_values(bars_mae)
        add_bar_values(bars_r2)
        add_bar_values(bars_direction)

        plt.tight_layout()
        return fig  # ✅ Return the figure for saving
