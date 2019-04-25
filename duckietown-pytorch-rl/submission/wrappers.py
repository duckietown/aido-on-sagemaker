import numpy as np

class DTPytorchWrapper():
    def __init__(self, shape=(120, 160, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])

    def preprocess(self, obs):
        from scipy.misc import imresize
        return imresize(obs, self.shape).transpose(2, 0, 1)