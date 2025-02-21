import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

class GaussianProcessModel:
    def __init__(self, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True):
        """
        Initializes the Gaussian Process Model with default or custom parameters.
        """
        self.kernel = C(1.0) * Matern(length_scale=5, length_scale_bounds=(1, 5000)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1))
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.normalize_y = normalize_y

        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            optimizer="fmin_l_bfgs_b"
        )
        # Feature scaler
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        """
        Trains the Gaussian Process model.
        """
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        self.model.fit(X_train_scaled, y_train)
        print("Optimized Kernel:", self.model.kernel_)

    def predict(self, X_test, return_std=False):
        """
        Makes predictions using the trained model, with optional uncertainty estimates.
        """
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_test_scaled = self.scaler.transform(X_test_flat)
        y_pred, sigma = self.model.predict(X_test_scaled, return_std=True)
        return (y_pred, sigma) if return_std else y_pred

    def validate(self, X, y):
        """
        Validates the model by computing Training vs. Validation RMSE.

        Returns:
        - fig (matplotlib.figure.Figure): The created figure.
        """
        # ✅ Train-Test Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # ✅ Train the model (NO EPOCHS)
        self.fit(X_train, y_train)

        # ✅ Predict on training and validation sets
        y_train_pred, _ = self.predict(X_train, return_std=True)
        y_val_pred, _ = self.predict(X_val, return_std=True)

        # ✅ Compute RMSE for Training & Validation
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        # ✅ Compute Cross-Validation RMSE (5-fold)
        cv_scores = -cross_val_score(self.model, self.scaler.transform(X.reshape(X.shape[0], -1)), y, cv=5, scoring="neg_root_mean_squared_error")
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # ✅ Plot Training vs Validation RMSE as **two bars**
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Training RMSE", "Validation RMSE"], [train_rmse, val_rmse], color=["blue", "orange"])

        ax.set_title("Gaussian Process: Training vs Validation RMSE")
        ax.set_ylabel("RMSE")
        ax.grid(axis="y")

        # ✅ Add Cross-Validation RMSE Info
        plt.figtext(0.5, -0.1,  
                    f"Cross-Validation Mean RMSE: {cv_mean:.4f} ± {cv_std:.4f}",
                    fontsize=12, ha="center", 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", edgecolor="black"))

        plt.tight_layout()
        return fig

    def summary(self):
        print(f"Gaussian Process Model Kernel: {self.model.kernel_}")
