import numpy as np


# y: output
# t: train_data
# y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# t = np.array([0, 0, 1, 0, 0])

def mean_square_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))