import keras_tuner as kt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


class LSTM_Model:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None  # Model will be created after tuning

    def create_lstm_model(self, hp):
        """Creates an LSTM model with tunable hyperparameters (uses Keras Tuner)."""
        neurons = hp.Int('neurons', min_value=50, max_value=200, step=50)
        dropout = hp.Choice('dropout', [0.2, 0.3, 0.4])
        optimizer = tf.keras.optimizers.get(hp.Choice('optimizer', ['adam', 'rmsprop']))

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),  
            LSTM(neurons, return_sequences=True),
            Dropout(dropout),
            LSTM(neurons),
            Dropout(dropout),
            Dense(1)
        ])        
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def tune_hyperparameters(self, X_train, y_train, max_trials=20, executions_per_trial=1, epochs=10):
        """Uses Keras Tuner to find the best hyperparameters and optionally train the model."""
        tuner = kt.RandomSearch(
            lambda hp: self.create_lstm_model(hp),  # Ensure hp is passed correctly
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='lstm_tuner',
            project_name='lstm_hyperparameter_tuning'
        )

        tuner.search(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Create the LSTM model with the best parameters
        self.model = tuner.hypermodel.build(best_hps)    

        return best_hps

    '''def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
        """Fits the model (after tuning)."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Run `tune_hyperparameters` first.")
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)'''

    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1):
        """Fits the model (after tuning)."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Run `tune_hyperparameters` first.")

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

    @tf.function(reduce_retracing=True)
    def predict_once(self, X):
        """Predicts output using the trained model while reducing retracing."""
        return self.model.predict(X)

    def predict(self, X):
        """Predicts output using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Run `tune_hyperparameters` first.")
        return self.model.predict(X)

    def summary(self):
        """Prints the model summary."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Run `tune_hyperparameters` first.")
        self.model.summary()