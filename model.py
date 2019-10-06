import tensorflow as tf

def make_model():
    dense_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(468, activation='tanh', input_dim=47),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(234, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(127, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='tanh'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    dense_model.summary()
    return dense_model
