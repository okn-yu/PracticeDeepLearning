import numpy as np
import dataset.mnist as mn
from src.neuralnet import NeuralNet
from src.layer import AffineLayer, ConvLayer, ReluLayer, PoolingLayer, SoftmaxWithLossLayer

# train_data.shape: (60000, 1, 28, 28)
# train_label.shape: (60000,)
(train_data, train_label), (test_data, test_label) = mn.load_mnist(flatten=False)

test_data = test_data[:3000]
test_label = test_label[:3000]

# HyperParameters
ITERS_NUM = 10000
TRAIN_SIZE = train_data.shape[0]
BATCH_SIZE = 200
ITER_PER_EPOC = max(TRAIN_SIZE / BATCH_SIZE, 1)

# for debug
batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
x_batch = train_data[batch_mask]
t_batch = train_label[batch_mask]

input_size = 28
filter_size = 5
filter_pad = 0
filter_stride = 1
filter_num = 30

conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
hidden_size = 100

layer1 = ConvLayer(fil_num=30, fil_chan=1, fil_hight=5, fil_width=5, pad=0, stride=1)
layer2 = ReluLayer()
layer3 = PoolingLayer(pool_hight=2, pool_width=2, pad=0, stride=2)

layer8 = ConvLayer(fil_num=90, fil_chan=30, fil_hight=5, fil_width=5, pad=0, stride=1)
layer9 = ReluLayer()
layer10 = PoolingLayer(pool_hight=2, pool_width=2, pad=0, stride=2)

# TODO: caluculate Affine W shape automatically!
pool_output_size = 1440
layer4 = AffineLayer(pool_output_size, hidden_size)
layer5 = ReluLayer()
layer6 = AffineLayer(hidden_size, 10)
layer7 = SoftmaxWithLossLayer()

#nnet = NeuralNet([layer1, layer2, layer8, layer2, layer3, layer4, layer5, layer6, layer7])
nnet = NeuralNet([layer1, layer2, layer3, layer8, layer9, layer10, layer4, layer5, layer6, layer7])
#nnet = NeuralNet([layer1, layer2, layer3, layer4, layer5, layer6, layer7])


for i in range(ITERS_NUM):

    if i % 100 == 0:
        print(i)

    batch_mask = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
    x_batch = train_data[batch_mask]
    t_batch = train_label[batch_mask]
    nnet.train(x_batch, t_batch)

    #if i % ITER_PER_EPOC == 0:
    if i % 300 == 0:
        train_acc = nnet.test(train_data[batch_mask], train_label[batch_mask])
        #print("train_acc...%s" % train_acc)
        test_acc = nnet.test(test_data, test_label)
        print("train_acc...%s, test_acc...%s" % (train_acc, test_acc))
