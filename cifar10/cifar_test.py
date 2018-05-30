import numpy as np
import matplotlib.pyplot as plt

def sigmod(matrix, reverse=False):
    if reverse:  # go back
        res = matrix * (1 - matrix)
        return res
    else:  # forward
        res = 1 / (1 + np.exp(-matrix))
        return res

def softmax_and_log(target_eye_matrix, x_matrix=None, s_matrix=None, reverse=False):
    """
    该方法是softmax + log整体，注意输入矩阵的shape都是m * num
    :param target_eye_matrix: shape(m, num)
    :param x_matrix: shape(m, num) 前向传播必填
    :param s_matrix: shape(m, num) 反向传播必填
    :param reverse: False表示前向传播，True表示反向传播
    :return: 当前向传播时，返回exp的得分概率与Loss值；当为反向传播时，返回Loss对输入向量X的误差
    """
    if reverse:  # go back
        s_matrix_shape = s_matrix.shape
        bitch = s_matrix_shape[1]
        return (s_matrix - target_eye_matrix) / bitch  # 一阶偏导就是(s - targer) / bitch
    else:  # forward
        exp_x = np.exp(x_matrix)
        exp_x_sum = np.sum(exp_x, axis=0, keepdims=True)
        s_matrix = exp_x / exp_x_sum
        ee = s_matrix * target_eye_matrix
        ee[ee < 0.0001] = 1
        y_matrix = -np.log(ee)
        return s_matrix, y_matrix


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def handle_input():
    mydict = unpickle('data_batch_1')
    mydict['hello'] = 5
    # print(mydict)
    data = np.array(mydict[b'data']).T
    labels_list = mydict[b'labels']
    labels = np.zeros(data.shape)
    for i in range(len(labels_list)):
        item = labels_list[i]
        labels[item, i] = 1
    return data[:, 500:600], labels[:, 500:600]


def to_one(a):
    one_size = a.shape[0]
    a_mean = np.mean(a, axis=1).reshape((one_size, 1))
    a_min = np.min(a, axis=1).reshape((one_size, 1))
    a_max = np.max(a, axis=1).reshape((one_size, 1))
    a = (a - a_mean) * 2 / (a_max - a_min)
    return a


def test(x0, w0, b0, w1, b1):
    # forward
    x1 = np.dot(w0, x0) + b0
    y1 = sigmod(x1)
    x2 = np.dot(w1, y1) + b1
    y2 = x2
    mmax = np.max(y2, axis=0, keepdims=True)
    y2 = y2 - mmax
    y2[y2 == 0] = 1
    y2[y2 < 0] = 0
    return y2


data_test = {}
data_test['x0'], data_test['target'] = handle_input()
data_test['x0'] = to_one(data_test['x0'])
param = np.load('cifar_paramters_result.npy').tolist()
result_test = test(data_test['x0'], param['best_w0'], param['best_b0'], param['best_w1'], param['best_b1'])
test_num = len(result_test.T)
target = []
test = []
for i in range(test_num):
    item_test = result_test[:, i].tolist()
    item_target = data_test['target'][:, i].tolist()
    try:
        test_type = item_test.index(1)
        test.append(test_type)
    except:
        test.append(-1)
    try:
        test_type = item_target.index(1)
        target.append(test_type)
    except:
        target.append(-1)

print(target)
print(test)



error_num = np.count_nonzero(np.array(target) - np.array(test))
totle_num = len(target)
error_rate = error_num / totle_num
print('error rate: ', error_rate)

plt.subplot(2, 1, 1)
plt.plot(range(totle_num), target, 'r', range(totle_num), test, 'g')


print('best loss = ', param['min_loss1'], param['min_loss2'])
plt.subplot(2, 1, 2)
plt.title('min Loss, epoch = '+str(param['epo']))
plt.xlabel('epoch')
plt.ylabel('loss')
# draw loss line, loss+reg line
plt.plot(param['loss_list'][:, 0], param['loss_list'][:, 1], 'r.', param['loss_list'][:, 0], param['loss_list'][:, 2], 'g.')
plt.show()