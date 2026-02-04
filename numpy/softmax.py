import numpy as np


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)
