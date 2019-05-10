import numpy as np


def mean_square_error(x, t):
    return 0.5 * np.sum((x - t) ** 2)


def cross_entropy_error(x, t):
    loss = -np.sum(t.T * np.log(x.T + 1e-7))
    return loss
