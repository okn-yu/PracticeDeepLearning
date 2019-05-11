import numpy as np
from src.layer import AffineLayer, ReluLayer, SoftmaxWithLossLayer, SigmoidLayer, Layer


class NeuralNet():
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = []
        #self.last_layer = self.layers[-1]

        self.layers.append(Layer(hidden_size, input_size, 'relu'))
        self.layers.append(Layer(output_size, hidden_size, 'softmax'))

        # self.layers.append(AffineLayer(hidden_size, input_size))
        # self.layers.append(ReluLayer())
        # #self.layers.append((SigmoidLayer()))
        #
        #
        # #self.layers.append(AffineLayer(hidden_size, hidden_size))
        # #self.layers.append(ReluLayer())
        # #self.layers.append((SigmoidLayer()))
        #
        # self.layers.append(AffineLayer(hidden_size, hidden_size))
        # self.layers.append(ReluLayer())
        #
        # self.layers.append(AffineLayer(hidden_size, hidden_size))
        # self.layers.append(ReluLayer())
        #
        # self.layers.append(AffineLayer(output_size, hidden_size))
        # self.layers.append(SoftmaxWithLossLayer())

    def add_layer(self, layer):
        self.layers.append(layer)

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
