
import tensorflow as tf


class ParamsPredictor(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.fc1 = tf.keras.layers.Dense(1, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.fc3 = tf.keras.layers.Dense(16, activation='sigmoid')

        self.reshape = tf.keras.layers.Reshape(target_shape=(50,))

    def call(self, inputs, **kwargs):

        # outputs = self.dropout1(inputs)
        outputs = self.fc1(inputs)
        outputs = self.reshape(outputs)
        outputs = self.dropout2(self.fc2(outputs))
        outputs_params = self.fc3(outputs)

        return outputs_params
