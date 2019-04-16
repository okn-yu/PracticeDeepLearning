import numpy as np
import matplotlib.pyplot as plt


def identify_function(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    else:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def relu(x):
    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = identify_function(x)
    y2 = sigmoid(x)
    y3 = softmax(x)
    y4 = step_function(x)
    y5 = relu(x)

    #plt.plot(x, y1)
    #plt.plot(x, y2)
    plt.plot(x, y3)
    #plt.plot(x, y4)
    #plt.plot(x, y5)
    plt.show()
