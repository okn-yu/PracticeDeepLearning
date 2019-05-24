import numpy as np
import dataset.mnist as mn
from src.neuralnet import NeuralNet
from src.layer import AffineLayer, ReluLayer, SigmoidLayer, SoftmaxWithLossLayer

(x_train, t_train), (x_test, t_test) = mn.load_mnist(normalize=True, one_hot_label=True)

# HyperParameters
ITERS_NUM = 10000
TRAIN_SIZE = x_train.shape[0]
BATCH_SIZE = 100
ITER_PER_EPOC = max(TRAIN_SIZE / BATCH_SIZE, 1)

layer1 = AffineLayer(output_dim=50, input_dim=784)
layer2 = ReluLayer()
layer3 = AffineLayer(output_dim=10, input_dim=50)
layer4 = SoftmaxWithLossLayer()

nnet = NeuralNet([layer1, layer2, layer3, layer4])

for i in range(ITERS_NUM):
    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    x_batch = x_train[batch_mask].T
    t_batch = t_train[batch_mask].T
    nnet.train(x_batch, t_batch)

    if i % ITER_PER_EPOC == 0:
        train_acc = nnet.test(x_train.T, t_train.T)
        test_acc = nnet.test(x_test.T, t_test.T)
        print("train_acc...%s, test_acc...%s" % (train_acc, test_acc))