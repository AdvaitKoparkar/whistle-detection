import scipy.signal
import numpy as np

class Conv2D(object):
    def __init__(self, weights, config={}):
        self.weights = weights
        self.padding  = config.get('padding', 'same')
        self.strides   = config.get('strides', (1,1))

    def __call__(self, X):
        filters = self.weights[0].numpy()
        bias    = self.weights[1].numpy()
        num_filters = filters.shape[3]
        kernel_size = filters.shape[0:2]
        num_channels = X.shape[3]
        num_samples = X.shape[0]

        if self.padding == 'same':
            pad_y = ((kernel_size[0]-1)//2, (kernel_size[0]-1)//2)
            pad_x = ((kernel_size[1]-1)//2, (kernel_size[1]-1)//2)
            X = np.pad(X, pad_width=((0,0),(pad_y), (pad_x), (0,0)))

        output = np.zeros((num_samples,
                          (X.shape[1] - kernel_size[0] + 1) // self.strides[0],
                          (X.shape[2] - kernel_size[1] + 1) // self.strides[1],
                          num_filters))

        for sample_idx in range(num_samples):
            for filter_idx in range(num_filters):
                output[sample_idx, :, :, filter_idx] += bias[filter_idx]
                for channel_idx in range(num_channels):
                    output[sample_idx, :, :, filter_idx] += scipy.signal.correlate2d(X[sample_idx, :, :, channel_idx],
                                                                           filters[:,:,channel_idx, filter_idx],
                                                                           mode='valid')[1::self.strides[0], 1::self.strides[1]]
        return output

class BatchNormalization(object):
    def __init__(self, weights, config={}):
        self.weights = weights
        self.epsilon = config.get('epsilon', 1e-24)

    def __call__(self, X):
        gamma = self.weights[0].numpy()
        beta  = self.weights[1].numpy()
        mu    = self.weights[2].numpy()
        sigma = self.weights[3].numpy()
        output = gamma * (X-mu) / (np.sqrt(sigma + self.epsilon)) + beta
        return output

class ReLU(object):
    def __init__(self, config):
        pass

    def __call__(self, X):
        output = np.maximum(0, X)
        return output
