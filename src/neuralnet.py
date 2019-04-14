import numpy as np
from src.activation_function import sigmoid, softmax
from src.loss_function import cross_entropy_error
from src.gradient import numerical_gradient, sigmoid_grad

WEIGHT_INIT_STD = 0.01


class TwoLayerNet():

    def __init__(self, input_size, hidden_size, output_size):
        self.params = {}
        self.params['W1'] = WEIGHT_INIT_STD * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = WEIGHT_INIT_STD * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
