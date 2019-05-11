import numpy as np
from src.activation_function import softmax, sigmoid
from src.loss_function import cross_entropy_error

WEIGHT_INIT_STD = 0.01
LEARNING_RATE = 0.1


class Layer:
    def __init__(self, output_size, input_size, activate_function):
        self.affine_layer = AffineLayer(output_size, input_size)
        self.activation_layer = self.set_activation_layer(activate_function)

    def set_activation_layer(self, activate_function):
        if activate_function == 'relu':
            return ReluLayer()
        elif activate_function == 'sigmoid':
            return SigmoidLayer()
        elif activate_function == 'softmax':
            return SoftmaxWithLossLayer()
        else:
            raise Exception

    def forward(self, x):
        u = self.affine_layer.forward(x)
        z = self.activation_layer.forward(u)

        return z

    def backward(self, z):
        u = self.activation_layer.backward(z)
        x = self.affine_layer.backward(u)

        return x

    def train(self):
        self.affine_layer.train()

    def loss(self, t):
        return self.activation_layer.loss(t)


class AffineLayer:
    def __init__(self, output_size, input_size):
        self.W = WEIGHT_INIT_STD * np.random.randn(output_size, input_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.W, self.x) + self.b.reshape(self.W.shape[0], 1)

        return out

    def backward(self, dout):
        self.dW = np.dot(dout, self.x.T)
        self.db = np.sum(dout, axis=1)

        dx = np.dot(self.W.T, dout)
        return dx

    def train(self):
        self.W -= LEARNING_RATE * self.dW
        self.b -= LEARNING_RATE * self.db


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0

        return x

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class SigmoidLayer:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = sigmoid(x)
        self.y = y

        return y

    def backward(self, dout):
        dx = dout * (1.0 - self.y) * self.y

        return dx


class SoftmaxWithLossLayer:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x):
        self.y = softmax(x)
        return self.y

    def loss(self, t):
        self.t = t
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[1]
        dx = (self.y - self.t) / batch_size
        return dx
