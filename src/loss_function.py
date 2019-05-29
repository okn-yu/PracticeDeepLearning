import numpy as np


def mean_square_error(x, t):
    return 0.5 * np.sum((x - t) ** 2)


def cross_entropy_error(x, t):

    if t.size == x.size:
        t = t.argmax(axis=1)

    batch_size = x.shape[0]
    return -np.sum(np.log(x[np.arange(batch_size), t] + 1e-7)) / batch_size