import numpy as np
import matplotlib.pyplot as plt
import dataset.mnist as mn
from src.neuralnet import TwoLayerNet

from debug.debug import BATCH_MASK

(x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)

# HyperParameters
ITERS_NUM = 10000
TRAIN_SIZE = x_train.shape[0]
BATCH_SIZE = 100
ITER_PER_EPOC = max(TRAIN_SIZE / BATCH_SIZE, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

nnet = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(ITERS_NUM):
    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    x_batch = x_train[batch_mask].T
    t_batch = t_train[batch_mask].T

    grad = nnet.train(x_batch, t_batch)

    if i % ITER_PER_EPOC == 0:
        train_acc = nnet.accuracy(x_train.T, t_train.T)
        test_acc = nnet.accuracy(x_test.T, t_test.T)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc...%s, test_acc...%s" % (train_acc, test_acc))
