import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int)
  
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def plot_activation_functions():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = relu(x)
    y2 = sigmoid(x)
    y3 = step_function(x)

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.show()
