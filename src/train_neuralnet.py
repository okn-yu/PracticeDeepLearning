import numpy as np
import matplotlib.pyplot as plt
import dataset.mnist as mn
from src.neuralnet import TwoLayerNet

from debug.debug import BATCH_MASK

(x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)

# HyperParameters
ITERS_NUM = 1
TRAIN_SIZE = x_train.shape[0]
BATCH_SIZE = 100
LEARNING_RATE = 0.1
ITER_PER_EPOC = max(TRAIN_SIZE / BATCH_SIZE, 1)

train_loss_list = []
train_acc_list = []
test_acc_list = []

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(ITERS_NUM):
    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    batch_mask = BATCH_MASK
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= LEARNING_RATE * grad[key]

    #loss = network.loss(x_batch, t_batch)
    #train_loss_list.append(loss)

    if i % ITER_PER_EPOC == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc...%s, test_acc...%s" % (train_acc, test_acc))

markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
#plt.show()