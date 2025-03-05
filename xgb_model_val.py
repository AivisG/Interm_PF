import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

class XGBoostModelVal:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, early_stopping_rounds=10, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs
        self.history = None

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs
        )

    '''def predict(self, X_test):
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test_reshaped)'''
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Final Test RMSE: {rmse}")
        return rmse

    def summary(self):
        print("\n **XGBoost Model Summary**")
        print(f" - n_estimators: {self.n_estimators}")
        print(f" - max_depth: {self.max_depth}")
        print(f" - learning_rate: {self.learning_rate}")
        print(f" - Early stopping rounds: {self.early_stopping_rounds}")
        print(f" - Other Parameters: {self.kwargs}")
        print(f" - Model Parameters: {self.model.get_params()}\n")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        # Convert only if the input is a DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.to_numpy()
    
        X_train = X_train.reshape(X_train.shape[0], -1)  # Ensure it's 2D
        if X_val is not None:
            X_val = X_val.reshape(X_val.shape[0], -1)
    
        y_train = y_train.flatten()  # Flatten target if needed
        if y_val is not None:
            y_val = y_val.flatten()
    
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = [(dtrain, "train")]
    
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            **self.kwargs
        }
    
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "validation"))
    
        self.history = {}
    
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            evals_result=self.history,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=verbose
        )

    def predict(self, X_test):
        # Convert DataFrame to NumPy array if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
    
        X_test = X_test.reshape(X_test.shape[0], -1)  # Ensure 2D
        dtest = xgb.DMatrix(X_test)  # Convert to DMatrix
        return self.model.predict(dtest)  # Use DMatrix for prediction

    
    def plot_training_history(self, metric="rmse"):
        if self.history is None or "train" not in self.history:
            print("Error: No training history found. Fit the model with validation data first.")
            return None
    
        fig, ax = plt.subplots(figsize=(10, 6))
    
        if "train" in self.history and metric in self.history["train"]:
            ax.plot(self.history["train"][metric], label="Training " + metric, color="blue", marker="o")
    
        if "validation" in self.history and metric in self.history["validation"]:
            ax.plot(self.history["validation"][metric], label="Validation " + metric, color="orange", linestyle="dashed", marker="o")
    
        ax.set_title(f"Fig 6 Training vs. Validation {metric.upper()} XGBoost")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
        return fig
