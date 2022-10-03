import numpy as np
import tensorflow as tf
from inference_lib import Conv2D

def convert_model(model):
    conv2d = Conv2D(model.layers[1].weights, {'padding': model.layers[1].padding, 'strides': model.layers[1].strides})
    m = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)    
    X = np.random.randn(1,64,64,1)
    y_tf = m.predict(X)
    
    import pdb; pdb.set_trace()
    conv2d(X)

if __name__ == '__main__':
    model_path = r"D:/Advait/Handouts_and_assignments/Raspberry_Pi/whistle_activated_camera/logs/M02_20220606_1/model.h5"
    model = tf.keras.models.load_model(model_path)
    convert_model(model)