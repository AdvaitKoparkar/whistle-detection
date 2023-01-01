import sys
sys.path.append('../')
import unittest
import numpy as np
import tensorflow as tf
from utils.inference.inference_lib import Conv2D

# class TestConv2D(unittest.TestCase):
#     def test_Conv2D_3x3(self):
#         weights = np.zeros((1,5,5,1))
#         config = {'padding': 'same', 'stride': 1}
#         conv2D_instance = Conv2D(weights, config)
#         y = conv2D_instance(np.random.randn(1,64,64,1))

#         model = tf.keras.models.Sequential()
#         model.add(tf.keras.layers.Conv2D(filters=1,
#                                          kernel_size=(5,5),
#                                          strides=(1,1),
#                                          padding='same',
#                                          use_bias=False))

if __name__ == '__main__':
    # unittest.main()

    weights = np.random.randn(1,5,5,1)
    # bn_weights = []
    config = {'padding': 'same', 'stride': 1}
    conv2D_instance = Conv2D(weights, config)
    y = conv2D_instance(np.random.randn(1,64,64,1))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=1,
                                     kernel_size=(5,5),
                                     strides=(1,1),
                                     padding='same',
                                     use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    import pdb; pdb.set_trace()
