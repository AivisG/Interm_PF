from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class GaussianProcessModel:
    def __init__(self, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True):
        """
        Initializes the Gaussian Process Model with default or custom parameters.
        """
        self.kernel = C(1.0) * Matern(length_scale=5, length_scale_bounds=(1, 5000)) + WhiteKernel()
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
        # Flatten and standardize input data
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        print("Optimized Kernel:", self.model.kernel_)

    def predict(self, X_test, return_std=False):
        """
        Makes predictions using the trained model, with optional uncertainty estimates.
        """
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        X_test_scaled = self.scaler.transform(X_test_flat)

        y_pred, sigma = self.model.predict(X_test_scaled, return_std=True)

        if return_std:
            return y_pred, sigma
        else:
            return y_pred

    def summary(self):
        print(f"Gaussian Process Model Kernel: {self.model.kernel_}")