import numpy as np
from collections import OrderedDict
from src.activation_function import sigmoid, softmax
from src.loss_function import cross_entropy_error
from src.layer import AffineLayer, SoftmaxWithLossLayer, ReluLayer


WEIGHT_INIT_STD = 0.01
LEARNING_RATE = 0.1

class TwoLayerNet():

    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = WEIGHT_INIT_STD * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = WEIGHT_INIT_STD * np.random.randn(output_size, hidden_size)
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

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=0)
        t = np.argmax(t, axis=0)

        accuracy = (np.sum(y == t) / float(x.shape[1]))
        return accuracy


    def train(self, x, t):

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
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        for key in ('W1', 'b1', 'W2', 'b2'):
            self.params[key] -= LEARNING_RATE * grads[key]

        return grads
