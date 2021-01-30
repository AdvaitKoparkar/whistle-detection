import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers

class M02(object):
    def __init__(self):
        self.conv01 = tf.keras.layers.Conv2D(filters=8,
                                   kernel_size=(5,5),
                                   strides=(2,2),
                                   padding='same',
                                   activation=None,
                                   name="Conv01")
        self.bn01 = tf.keras.layers.BatchNormalization(name='bn01')
        self.act01 = tf.keras.layers.ReLU(name='relu01')

        self.conv02 = tf.keras.layers.Conv2D(filters=16,
                                             kernel_size=(3,3),
                                             strides=(2,2),
                                             padding='same',
                                             activation=None,
                                             name="Conv02")
        self.bn02 = tf.keras.layers.BatchNormalization(name='bn02')
        self.act02 = tf.keras.layers.ReLU(name='relu02')

        self.conv03 = tf.keras.layers.Conv2D(filters=16,
                                             kernel_size=(3,3),
                                             strides=(2,2),
                                             padding='same',
                                             activation=None,
                                             name='Conv03')
        self.bn03 = tf.keras.layers.BatchNormalization(name='bn03')
        self.act03 = tf.keras.layers.ReLU(name='relu03')

        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        self.fc04 = tf.keras.layers.Conv2D(filters=1,
                                           kernel_size=(8,8),
                                           strides=(1,1),
                                           padding='valid',
                                           activation='sigmoid',
                                           name='fc04')

        self.global_pool = tf.keras.layers.GlobalMaxPool2D(name='global_pool')

    def get_model(self, input_shape, batch_size):
        input = tf.keras.layers.Input(shape=input_shape,
                                      batch_size=batch_size,
                                      name="input0")
        X = self.conv01(input)
        X = self.bn01(X)
        X = self.act01(X)

        X = self.conv02(X)
        X = self.bn02(X)
        X = self.act02(X)

        X = self.conv03(X)
        X = self.bn03(X)
        X = self.act03(X)

        X = self.dropout(X)

        X = self.fc04(X)
        output = self.global_pool(X)

        m = tf.keras.Model(inputs=input, outputs=output)
        return m
