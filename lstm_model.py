import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization

class LSTM_Model:
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self.create_lstm_model()  # Directly create the model without tuning

    def create_lstm_model(self):
        """Creates an LSTM model with fixed hyperparameters (no tuning)."""
        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),  
            LSTM(200, return_sequences=True),
            BatchNormalization(),  
            Dropout(0.3),
            LSTM(200),
            Dropout(0.3),
            Dense(1)
        ])        
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)  
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())  
        return model

    def fit(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.1):
        """Fits the model with provided data."""
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X, scaler_target=None):
        """Predicts output using the trained model and optionally inverse transforms predictions."""
        predictions = self.model.predict(X)

        if scaler_target is not None:
            predictions = scaler_target.inverse_transform(predictions)
        
        return predictions

    def summary(self):
        """Prints the model summary."""
        self.model.summary()
