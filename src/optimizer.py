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

    def __init__(self):
        pass

class AdaGard:

    def __init__(self):
        pass