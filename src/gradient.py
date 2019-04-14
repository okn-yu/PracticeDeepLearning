import numpy as np

LR = 0.01
STEP_NUM = 100

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x):
    x = init_x

    for i in range(STEP_NUM):
        grad = numerical_gradient(f, x)
        x -= LR * grad

    return x

# def sample_function_1(x):
#     return 0.01 * x ** 2 + 0.1 * x
#
# def sample_function_2(x):
#     return x[0] ** 2 + x[1] ** 2
#
# print(numerical_diff(sample_function_1, 5))
# print(numerical_gradient(sample_function_2,np.array([3.0, 4.0])))
# print(gradient_descent(sample_function_2, np.array([-3.0, 4.0])))