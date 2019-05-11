import numpy as np
from collections import OrderedDict
from src.layer import AffineLayer, SoftmaxWithLossLayer, ReluLayer


class NeuralNet():

    def __init__(self, input_size, hidden_size, output_size):

        self.layers = []
        self.layers.append(AffineLayer(hidden_size, input_size))
        self.layers.append(ReluLayer())
        self.layers.append(AffineLayer(output_size, hidden_size))
        self.layers.append(SoftmaxWithLossLayer())

    def predict(self, x):

        for layer in self.layers:
            x = layer.forward(x)

        return x


    def loss(self, x, t):
        y = self.predict(x)

        return self.layers[3].loss(t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=0)
        t = np.argmax(t, axis=0)

        accuracy = (np.sum(y == t) / float(x.shape[1]))
        return accuracy


    def train(self, x, t):

        self.loss(x, t)

        dout = 1
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        reversed(self.layers)
        self.layers[0].update()
        self.layers[2].update()

