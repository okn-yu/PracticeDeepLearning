import numpy as np
from src.activation_function import softmax, sigmoid
from src.loss_function import cross_entropy_error
from src.optimizer import SGD, Momentum_SGD
from src.util import col2im, im2col

WEIGHT_INIT_STD = 0.01
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class AffineLayer:
    def __init__(self, input_dim, output_dim):
        self.W = WEIGHT_INIT_STD * np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = np.dot(dout, self.W.T)
        dx = dx.reshape(*self.original_x_shape)

        return dx

    def train(self):
        self.W -= LEARNING_RATE * self.dW
        self.b -= LEARNING_RATE * self.db


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
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        batch_size, chan, hight, width = x.shape
        #print(x.shape)

        output_hight = int(((hight + 2 * self.pad - self.fil_hight) / self.stride) + 1)
        output_width = int(((width + 2 * self.pad - self.fil_width) / self.stride) + 1)

        col = im2col(x, self.fil_hight, self.fil_width, self.stride, self.pad)
        col_W = self.W.reshape(self.fil_num, -1).T
        #print(col.shape, col_W.shape)

        out = np.dot(col, col_W) + self.b
        out = out.reshape(batch_size, output_hight, output_width, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        #print(out.shape)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.fil_num)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.fil_num, self.fil_chan, self.fil_hight, self.fil_width)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, self.fil_hight, self.fil_width, self.stride, self.pad)

        return dx

    def train(self):
        self.W -= LEARNING_RATE * self.dW
        self.b -= LEARNING_RATE * self.db


class PoolingLayer():
    def __init__(self, pool_hight, pool_width, pad, stride):
        self.pool_hight = pool_hight
        self.pool_width = pool_width
        self.pad = pad
        self.stride = stride

    def forward(self, x):
        batch_size, chan, hight, width = x.shape

        output_hight = int(((hight - self.pool_hight) / self.stride) + 1)
        output_width = int(((width - self.pool_width) / self.stride) + 1)

        col = im2col(x, self.pool_hight, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_hight * self.pool_width)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(batch_size, output_hight, output_width, chan).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_hight * self.pool_width
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_hight, self.pool_width, self.stride, self.pad)

        return dx

    def train(self):
        pass


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
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

    def train(self):
        pass
