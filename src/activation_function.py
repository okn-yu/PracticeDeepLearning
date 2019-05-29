import numpy as np


def identify_function(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # if x  == matrix
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # if x == vector
    else:
        x = x - np.max(x)
        y = np.exp(x) / np.sum(np.exp(x))
        return y


def step_function(x):
    boolean_value = (x > 0)
    # y return 1 if bool_value = True, else y returns 0.
    array_y = np.array(boolean_value, dtype=np.int)
    return array_y


def relu(x):
    return np.maximum(0, x)