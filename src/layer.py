import numpy as np
from src.activation_function import softmax, sigmoid
from src.loss_function import cross_entropy_error
from src.optimizer import SGD, Momentum_SGD

WEIGHT_INIT_STD = 0.01
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class Layer:
    def __init__(self, output_dim, input_dim, activate_function):
        self.affine_layer = AffineLayer(output_dim, input_dim)
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
    def __init__(self, output_dim, input_dim):
        self.W = WEIGHT_INIT_STD * np.random.randn(output_dim, input_dim)
        self.b = np.zeros(output_dim)
        self.x = None
        self.dW = None
        self.db = None

        # TODO:パラメータとパラメータの傾配は学習で必須なので辞書型で書き直すこと（やっぱり必要だった）

        #""""
        # Momentum
        self.momentum = 0.9
        self.v_W = np.zeros((output_dim, input_dim))
        #self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(output_dim)
        #self.v_b = np.zeros(self.b.shape)
        #"""

        """
        # AdaGrad
        self.h_W = np.zeros((output_dim, input_dim))
        self.h_b = np.zeros(output_dim)
        self.lr = 0.1
        """


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

        self.param = {}
        self.grad_param = {}

        self.param['W'] = self.W
        self.param['b'] = self.b
        self.grad_param['dW'] = self.dW
        self.grad_param['dB'] = self.db

        """
        sgd = SGD(LEARNING_RATE)
        sgd.update(self.W, self.dW)
        sgd.update(self.b, self.db)
        """

        #"""
        mom_sgd = Momentum_SGD(MOMENTUM)
        mom_sgd.update(self.W, self.dW)
        mom_sgd.update(self.b, self.db)
        #"""


        # Momentum_SGD
        #self.v_W = self.momentum * self.v_W - 0.01 * self.dW
        #self.W += self.v_W

        #self.v_b = self.momentum * self.v_b - 0.01 * self.db
        #self.b += self.v_b


        """
        # AdaGard
        self.h_W += self.dW * self.dW
        self.W -= self.lr * self.dW / np.sqrt(self.h_W + 1e-7)

        self.h_b += self.db * self.db
        self.b -= self.lr * self.db / np.sqrt(self.h_b + 1e-7)
        """




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