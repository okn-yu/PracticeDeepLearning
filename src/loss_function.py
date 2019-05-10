import numpy as np

def mean_square_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    loss =  -np.sum(t * np.log(y + delta))
    return loss
