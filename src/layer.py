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

        # TODO:パラメータとパラメータの傾配は学習で必須なので辞書型で書き直すこと（やっぱり必要だった）

        """"
        # Momentum
        self.momentum = 0.9
        self.v_W = np.zeros((output_dim, input_dim))
        #self.v_W = np.zeros(self.W.shape)
        self.v_b = np.zeros(output_dim)
        #self.v_b = np.zeros(self.b.shape)
        """

        """
        # AdaGrad
        self.h_W = np.zeros((output_dim, input_dim))
        self.h_b = np.zeros(output_dim)
        self.lr = 0.1
        """

    def forward(self, x):
        self.x = x

        # テンソル対応
        self.x = x.reshape(x.shape[0], -1)

        # self.x: (100, 4320)
        # self.W: (100, 4320)

        out = np.dot(self.W, self.x) + self.b.reshape(self.W.shape[0], 1)
        #out = np.dot(self.W, self.x) + self.b
        #out = np.dot(self.x, self.W)
        # out.shape: (4320, 4320)

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

        # """
        sgd = SGD(LEARNING_RATE)
        sgd.update(self.W, self.dW)
        sgd.update(self.b, self.db)
        # """

        """
        mom_sgd = Momentum_SGD(MOMENTUM)
        mom_sgd.update(self.W, self.dW)
        mom_sgd.update(self.b, self.db)
        """

        # Momentum_SGD
        # self.v_W = self.momentum * self.v_W - 0.01 * self.dW
        # self.W += self.v_W

        # self.v_b = self.momentum * self.v_b - 0.01 * self.db
        # self.b += self.v_b

        """
        # AdaGard
        self.h_W += self.dW * self.dW
        self.W -= self.lr * self.dW / np.sqrt(self.h_W + 1e-7)

        self.h_b += self.db * self.db
        self.b -= self.lr * self.db / np.sqrt(self.h_b + 1e-7)
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
        batch_size, chan, hight, width = x.shape

        output_hight = int(((hight + 2 * self.pad - self.fil_hight) / self.stride) + 1)
        output_width = int(((width + 2 * self.pad - self.fil_width) / self.stride) + 1)

        # two_dim_x.shape: (batch_size * output_hight * output_width, self.fil_chan * self.fil_size)
        # two_dim_W.shape: (self.fil_chan * self.fil_size, -1)

        two_dim_x = im2col(x, self.fil_hight, self.fil_width, self.stride, self.pad)
        # two_dim_x.shape: (57600, 25)
        print(two_dim_x.shape)

        two_dim_W = self.W.reshape(self.fil_num, -1).T
        print(two_dim_W.shape)
        # two_dim_W.shape: (25, 30)

        out = np.dot(two_dim_x, two_dim_W) + self.b
        out = out.reshape(batch_size, output_hight, output_width, -1).transpose(0, 3, 1, 2)

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
