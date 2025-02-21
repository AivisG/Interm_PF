import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    def __init__(self, models, X_test, y_test, ticker: str):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.evaluation_results = {}
        self.y_predictions = {}
        self.ticker = ticker

    def evaluate(self):
        for name, model in self.models.items():
            # Get predictions
            y_pred = model.predict(self.X_test).flatten()

            # Compute metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            direction_accuracy = np.mean(
                np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(self.y_test[1:] - self.y_test[:-1])
            )
            # Store results
            self.evaluation_results[name] = {
                'MSE, lower is better': mse,
                'MAE, lower is better': mae,
                'R², higher is better': r2,
                'Direction Accuracy, above 0.5 is better': direction_accuracy
            }
            self.y_predictions[name] = y_pred

        return self.evaluation_results, self.y_predictions

    def plot_metrics(self):
        metrics = list(self.evaluation_results[next(iter(self.evaluation_results))].keys())
        model_names = list(self.evaluation_results.keys())

        results_dict = {
            metric: [self.evaluation_results[model][metric] for model in model_names]
            for metric in metrics
        }
        bar_width = 0.2
        x_positions = range(len(model_names))
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics):
            metric_values = results_dict[metric]
            bars = ax.bar(
                [x + i * bar_width for x in x_positions],
                metric_values,
                width=bar_width,
                label=metric
            )
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # X position
                    height,  # Y position
                    f"{height:.2f}",  # Text (formatted to 2 decimal places)
                    ha='center', va='bottom', fontsize=10, color='black'
                )
        ax.set_xticks([x + (len(metrics) - 1) * bar_width / 2 for x in x_positions])
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{self.ticker} Comparison of Models Based on Evaluation Metrics")
        ax.legend(title="Metrics", loc="upper left", bbox_to_anchor=(1, 1))
        #ax.legend(title="Metrics")
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # Adds grid lines at intervals of 0.1
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.show()