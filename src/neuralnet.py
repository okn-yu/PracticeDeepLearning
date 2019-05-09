import numpy as np
from collections import OrderedDict
from src.activation_function import sigmoid, softmax
from src.loss_function import cross_entropy_error
from src.gradient import numerical_gradient, sigmoid_grad
from src.layer import AffineLayer, SoftmaxWithLossLayer, ReluLayer

from debug.debug import WEIGHT_MATRIX_1, WEIGHT_MATRIX_2

WEIGHT_INIT_STD = 0.01
np.random.seed(seed=100)

class TwoLayerNet():

    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = WEIGHT_INIT_STD * np.random.randn(input_size, hidden_size)
        self.params['W1'] = WEIGHT_MATRIX_1
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = WEIGHT_INIT_STD * np.random.randn(hidden_size, output_size)
        self.params['W2'] = WEIGHT_MATRIX_2
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = AffineLayer(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLossLayer()

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        # return cross_entropy_error(y, t)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # t = np.argmax(t, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def neural_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}

        print('w1')
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        print('b1')
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        print('w2')
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        print('b2')
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):

        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db

        print("Affine2-dW:%s" % self.layers['Affine2'].dW[0][0])

        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
