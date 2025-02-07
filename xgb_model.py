import xgboost as xgb

class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs  

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs  
        )

    def fit(self, X_train, y_train):
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Flatten input if needed
        self.model.fit(X_train_reshaped, y_train)

    def predict(self, X_test):
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)  # Flatten input if needed
        return self.model.predict(X_test_reshaped)

    def get_params(self):
        return self.model.get_params()

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)

    def summary(self):
        print(f"XGBoost Model Config: {self.model.get_params()}")