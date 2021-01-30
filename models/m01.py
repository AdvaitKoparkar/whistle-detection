import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

class M01(object):
    def __init__(self):
        self.conv01 = tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(5,5),
                                   strides=(2,2),
                                   padding="same",
                                   activation="relu",
                                   name="Conv01")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense01 = tf.keras.layers.Dense(1, activation='sigmoid', name="Dense01")

    def get_model(self, input_shape, batch_size):
        input = tf.keras.layers.Input(shape=input_shape,
                                      batch_size=batch_size,
                                      name="input0")
        X = self.conv01(input)
        X = self.flatten(X)
        output = self.dense01(X)
        m = tf.keras.Model(inputs=input, outputs=output)
        return m
