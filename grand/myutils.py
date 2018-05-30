import numpy as np


def sigmod(x, reverse=False):
    if reverse:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def relu(x, reverse=False):
    if reverse:
        x[x > 0] = 1
        x[x < 0] = 0
    else:
        x[x<0] = 0
    return x


def softmax_and_log(target, x=None, s=None, reverse=False):
    if reverse:
        batch_size = target.shape[1]
        return (s - target) / batch_size  # Loss的关于x的一阶偏导就是(s - t) / batch_size
    else:
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=0, keepdims=True)
        s_matrix = exp_x / sum_exp_x
        ee = s_matrix * target
        ee[ee < 1e-3] = 1
        y_matrix = -np.log(ee)
        return s_matrix, y_matrix


def to_one(x, has_param=False, param_x_mean=None, param_x_max=None):
    if has_param:
        x = x - param_x_mean
        return x / param_x_max
    else:
        x_mean = np.mean(x, axis=1, keepdims=True)
        x = x - x_mean
        x_max = np.max(x, axis=1, keepdims=True)
        return x / x_max, x_mean, x_max