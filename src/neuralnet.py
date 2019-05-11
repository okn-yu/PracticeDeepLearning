import numpy as np


class NeuralNet():
    def __init__(self, layers):
        self.layers = []
        self._add_layers(layers)

    def _add_layers(self, layers):
        self.layers = [layer for layer in layers]

    def train(self, x, t):
        self._forward(x)
        self._loss(t)
        self._backward(dout=1)
        self._train()

    def _forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def _loss(self, t):
        return self.layers[-1].loss(t)

    def _backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def _train(self):
        for layer in self.layers:
            layer.train()

    def test(self, x, t):
        y = self._forward(x)
        y = np.argmax(y, axis=0)
        t = np.argmax(t, axis=0)

        accuracy = (np.sum(y == t) / float(x.shape[1]))
        return accuracy
