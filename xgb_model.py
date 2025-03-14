import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, **kwargs):
        self.n_estimators = n_estimators  # Number of boosting rounds (trees)
        self.max_depth = max_depth  # Maximum depth of each decision tree
        self.learning_rate = learning_rate  # Step size shrinkage to prevent overfitting
        self.kwargs = kwargs  # Allows additional hyperparameters to be passed
        self.history = None  # Store training history

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs  
        )

    def predict(self, X_test):
        """
        Makes predictions on new data using Booster.
        """
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)  # Flatten input if needed
        dtest = xgb.DMatrix(X_test_reshaped)  # Convert to DMatrix
        return self.model.predict(dtest)  # Use DMatrix for prediction
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model performance on test data.
        """
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Final Test RMSE: {rmse}")
        return rmse

    def summary(self):
        """
        Prints the configuration of the XGBoost model.
        """
        print("\n **XGBoost Model Summary**")
        print(f" - n_estimators: {self.n_estimators}")
        print(f" - max_depth: {self.max_depth}")
        print(f" - learning_rate: {self.learning_rate}")
        print(f" - Other Parameters: {self.kwargs}")
        print(f" - Model Parameters: {self.model.get_params()}\n")
    
    def fit(self, X_train, y_train, verbose=False):
        # Ensure X_train is 2D
        if X_train.ndim != 2:
            print(f"Reshaping X_train from {X_train.shape} to 2D")
            X_train = X_train.reshape(X_train.shape[0], -1)  # Convert to (samples, features)
    
        # Ensure y_train is 1D
        y_train = y_train.flatten()  
    
        dtrain = xgb.DMatrix(X_train, label=y_train) 
    
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            **self.kwargs
        }
    
        # Initialize storage for training history
        self.history = {}
    
        # Train with history tracking (even without validation)
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train")],  # Store training metrics
            evals_result=self.history,  # Store evaluation results
            verbose_eval=verbose
        )
    
    def plot_training_history(self, metric="rmse"):
        """
        Plots the training and validation loss curve.
        """
        if self.history is None or "train" not in self.history:
            print("Error: No training history found. Fit the model with validation data first.")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot training metric
        if "train" in self.history and metric in self.history["train"]:
            ax.plot(self.history["train"][metric], label="Training " + metric, color="blue", marker="o")
    
        # Plot validation metric only if available
        if "validation" in self.history and metric in self.history["validation"]:
            ax.plot(self.history["validation"][metric], label="Validation " + metric, color="orange", linestyle="dashed", marker="o")
    
        ax.set_title(f"Training vs. Validation {metric.upper()} XGBoost without validation.")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
        return fig 
