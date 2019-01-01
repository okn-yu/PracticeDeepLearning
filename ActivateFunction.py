import numpy as np
import matplotlib.pyplot as plt

# def step_function(x):
#     return np.array(x > 0, dtype = np.int)
#　
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def relu(x):
#     return np.maximum(0, x)
#
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a -c)
#     sum_exp_a = np.sum(exp_a)
#
#     return exp_a / sum_exp_a
#
# a = np.array([0.3, 2.9, 4.0])
# y = softmax(a)
# print(y)

# x = np.arange(-5.0, 5.0, 0.1)
# y1 = relu(x)
# y2 = sigmoid(x)
# y3 = step_function(x)
#
#
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.show()
