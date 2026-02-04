from numpy.scaled_dot_prod import attention
import numpy as np


def multihead_attention(h, W_O, W_Q_channels, W_K_channels, W_V_channels, Q, K, V):
    outputs = []
    d_k = Q.shape[-1] // h
    for i in range(h):
        Q_i = np.matmul(Q, W_Q_channels[i])
        K_i = np.matmul(K, W_K_channels[i])
        V_i = np.matmul(V, W_V_channels[i])
        outputs.append(attention(Q_i, K_i, V_i, d_k))

    concat = np.concatenate(outputs, axis=-1)
    return np.matmul(concat, W_O)
