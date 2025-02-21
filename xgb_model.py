import xgboost as xgb
import matplotlib.pyplot as plt

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs  
        self.history = None  # Store training history

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs  
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Trains the model and tracks training history if validation data is provided.
        """
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Flatten input if needed
        eval_set = [(X_train_reshaped, y_train)]  # ✅ Always include training data
        
        if X_val is not None and y_val is not None:
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1)  # Flatten validation input if needed
            eval_set.append((X_val_reshaped, y_val))  # ✅ Ensure validation data is included
        
        # ✅ Ensure eval_metric is set inside model parameters
        self.model.set_params(eval_metric="rmse")

        self.model.fit(
            X_train_reshaped, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        # ✅ Store the training history
        self.history = self.model.evals_result()

    def predict(self, X_test):
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)  # Flatten input if needed
        return self.model.predict(X_test_reshaped)

    def summary(self):
        """
        Prints the configuration of the XGBoost model.
        """
        print("\n📌 **XGBoost Model Summary**")
        print(f" - n_estimators: {self.n_estimators}")
        print(f" - max_depth: {self.max_depth}")
        print(f" - learning_rate: {self.learning_rate}")
        print(f" - Other Parameters: {self.kwargs}")
        print(f" - Model Parameters: {self.model.get_params()}\n")

    def plot_training_history(self, metric="rmse"):
        """
        Plots the training and validation loss from training history.
    
        Parameters:
        metric : str, optional
            Metric to plot (default is "rmse").
    
        Returns:
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        """
        if self.history is None:
            print("Error: No training history found. Fit the model with validation data first.")
            return None  # Return None to avoid errors

        fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure

        # ✅ Plot training RMSE
        if metric in self.history["validation_0"]:
            ax.plot(self.history["validation_0"][metric], label="Training " + metric, color="blue", marker="o")

        # ✅ Plot validation RMSE if available
        if len(self.history) > 1 and "validation_1" in self.history and metric in self.history["validation_1"]:
            ax.plot(self.history["validation_1"][metric], label="Validation " + metric, color="orange", linestyle="dashed", marker="o")

        ax.set_title(f"Training vs. Validation {metric.upper()} XGBoost")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)

        return fig  # Return figure for saving
