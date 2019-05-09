import numpy as np


# y: output
# t: train_data
# y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# t = np.array([0, 0, 1, 0, 0])

def mean_square_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    loss =  -np.sum(t * np.log(y + delta))
    print("loss...%s" % loss)
    return loss


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
#     if t.size == y.size:
#         t = t.argmax(axis=1)
#
#     batch_size = y.shape[0]
#
#
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size