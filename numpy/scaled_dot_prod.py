import numpy as np
from numpy.softmax import softmax


def attention(Q, K, V, d_k):
    QK = np.matmul(Q, K.T)
    QK /= np.sqrt(d_k)
    return np.matmul(softmax(QK), V)
