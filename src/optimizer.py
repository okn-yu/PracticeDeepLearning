import numpy as np

class SGD:
    # Stochastic Gradient Descent

    def __init__(self, lr):
        self.lr = lr

    def update(self, param, grad_param):
        param -= self.lr * grad_param

        #self.W -= LEARNING_RATE * self.dW
        #self.b -= LEARNING_RATE * self.db

class Momentum_SGD:
    # Momentum SDG

    def __init__(self, momentum):
        self.momentum = momentum
        #self.v_param ...各パラメータごとにクラス変数化する必要がある

    def update(self, param, grad_param):
        v_param = np.zeros(param.shape)
        v_param = self.momentum * v_param - 0.01 * grad_param
        param += v_param

class AdaGard:

    def __init__(self):
        pass