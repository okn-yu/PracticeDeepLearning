import numpy as np
from src.activation_function import softmax, sigmoid
from src.loss_function import cross_entropy_error
from src.optimizer import SGD, Momentum_SGD
from src.util import col2im, im2col

WEIGHT_INIT_STD = 0.01
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class AffineLayer:
    def __init__(self, output_dim, input_dim):
        self.W = WEIGHT_INIT_STD * np.random.randn(output_dim, input_dim)
        self.b = np.zeros(output_dim)
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

"""
class ConvLayer:
    def __init__(self, fil_num, fil_chan, fil_hight, fil_width, pad, stride):
        self.fil_num = fil_num
        self.fil_chan = fil_chan
        self.fil_hight = fil_hight
        self.fil_width = fil_width
        self.pad = pad
        self.stride = stride
        self.W = WEIGHT_INIT_STD * np.random.randn(self.fil_num, self.fil_chan, self.fil_hight, self.fil_width)
        self.b = np.zeros(self.fil_num)

        self.x = None
        self.col = None
        self.col_w = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # batch_size, chan, hight, width = x.shape
        width, hight, chan, batch_size = x.shape

        output_hight = int(((hight + 2 * self.pad - self.fil_hight) / self.stride) + 1)
        output_width = int(((width + 2 * self.pad - self.fil_width) / self.stride) + 1)

        # two_dim_x.shape: (batch_size * output_hight * output_width, self.fil_chan * self.fil_size)
        # two_dim_W.shape: (self.fil_chan * self.fil_size, -1)

        two_dim_x = im2col(x.T, self.fil_hight, self.fil_width, self.stride, self.pad)
        # two_dim_x.shape: (57600, 25)
        print("two_dim_x.shape " + str(two_dim_x.shape))

        two_dim_W = self.W.reshape(self.fil_num, -1)
        print("two_dim_W.shape " + str(two_dim_W.shape))
        # two_dim_W.shape: (25, 30)

        out = np.dot(two_dim_W, two_dim_x.T) + self.b.reshape(two_dim_W.shape[0], 1)
        # out = out.reshape(batch_size, output_hight, output_width, -1).transpose(0, 3, 1, 2)

        out = out.reshape(output_width, output_hight, -1, batch_size).transpose(0, 3, 1, 2)

        print("out.shape " + str(out.shape))
        # out.shape: (100, 30, 24, 24)

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.fil_num)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.fil_num, self.fil_chan, self.fil_hight, self.fil_width)

        dcol = np.dot(dout, self.col_w.T)
        dx = col2im(dcol, self.x.shape, self.fil_hight, self.fil_width, self.stride, self.pad)

        return dx


class PoolingLayer():
    def __init__(self, pool_hight, pool_width, pad, stride):
        self.pool_hight = pool_hight
        self.pool_width = pool_width
        self.pad = pad
        self.stride = stride

    def forward(self, x):
        print("x.shape " + str(x.shape))
        batch_size, chan, hight, width = x.shape

        output_hight = int(((hight - self.pool_hight) / self.stride) + 1)
        output_width = int(((width - self.pool_width) / self.stride) + 1)

        col = im2col(x, self.pool_hight, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_hight * self.pool_width)

        out = np.max(col, axis=1)
        out = out.reshape(batch_size, output_hight, output_width, chan).transpose(0, 3, 1, 2)

        # out.shape (100, 30, 12, 12)
        print("pool:out.shape " + str(out.shape))

        return out
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

    def train(self):
        pass


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

    def train(self):
        pass


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

    def train(self):
        pass
