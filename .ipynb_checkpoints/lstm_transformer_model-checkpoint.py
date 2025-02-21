import tensorflow as tf
import numpy as np
import os

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMTransformer(tf.keras.Model):
    def __init__(self, lstm_units=64, num_layers=2, d_model=128, num_heads=4, dff=512, input_seq_len=60):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        self.projection = tf.keras.layers.Dense(d_model)  # ✅ Projects LSTM output to d_model
        self.transformer = TransformerEncoder(num_layers, d_model, num_heads, dff, input_seq_len)
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.lstm(inputs)  # Shape: (batch, seq_len, 64)
        x = self.projection(x)  # ✅ Convert to (batch, seq_len, 128)
        x = self.transformer(x, training=training)  # Now matches d_model=128
        return self.final_layer(x[:, -1, :])  # Output: (batch, 1)
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_seq_len, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Positional encoding to maintain sequence information
        self.pos_encoding = PositionalEncoding(input_seq_len, d_model)

        # Transformer layers
        self.enc_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
            for _ in range(num_layers)
        ]
        self.ffn_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ])
            for _ in range(num_layers)
        ]
        self.layernorms1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.layernorms2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.dropouts1 = [tf.keras.layers.Dropout(dropout_rate) for _ in range(num_layers)]
        self.dropouts2 = [tf.keras.layers.Dropout(dropout_rate) for _ in range(num_layers)]

    def call(self, inputs, training):
        x = self.pos_encoding(inputs)

        for i in range(self.num_layers):
            attn_output = self.enc_layers[i](x, x)  # Self-attention
            attn_output = self.dropouts1[i](attn_output, training=training)
            x = self.layernorms1[i](x + attn_output)

            ffn_output = self.ffn_layers[i](x)
            ffn_output = self.dropouts2[i](ffn_output, training=training)
            x = self.layernorms2[i](x + ffn_output)

        return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sine to even indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cosine to odd indices
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        return pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# If run as a script, test with dummy data
if __name__ == "__main__":
    transformer = LSTMTransformer(  # ✅ Remove 'target_seq_len'
        num_layers=4,
        d_model=128,
        num_heads=8,
        dff=512,
        input_seq_len=60
    )

    # Example Input
    dummy_input = tf.random.normal((32, 60, 128))  # (Batch size, Sequence length, Features)
    output = transformer(dummy_input)

    print("Model Output Shape:", output.shape)  # Expected: (32, 1)