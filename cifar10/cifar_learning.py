import numpy as np
import time
import matplotlib.pyplot as plt

start = time.clock()

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
    return data, labels


def train(source_x,  source_t, bitch_size, hidden_number, learning_rate, epoch, reg_bad):
    sample_num = source_x.shape[1]  # 样本个数
    input_num = source_x.shape[0]  # 输入层神经元个数
    output_num = source_t.shape[0]
    loss_list = []  # 存放loss的list，在绘制loss曲线图使用

    # 初始化W,B
    W0 = 100 * np.random.random() * (
    np.random.random((hidden_number, input_num)) - 0.5)  # W0.shape = (hidden_number, input_num)
    B0 = np.zeros((hidden_number, 1))  # B0.shape = (hidden_number, 1)
    W1 = 100 * np.random.random() * (
    np.random.random((output_num, hidden_number)) - 0.5)  # W1.shape = (output_num, hidden_number)
    B1 = np.zeros((output_num, 1))  # B1.shape = (output_num, 1)

    bitch_num = (int)(np.ceil(sample_num / bitch_size))
    for i in range(epoch):
        for ii in range(bitch_num):
            X = source_x[:, bitch_size * ii:bitch_size * (ii + 1)]
            T = source_t[:, bitch_size * ii:bitch_size * (ii + 1)]
            real_bitch_size = X.shape[1]
            # forward
            x1 = np.dot(W0, X) + B0
            y1 = sigmod(x1)
            x2 = np.dot(W1, y1) + B1
            y2 = x2
            s_matrix, y_matrix = softmax_and_log(T, x_matrix=y2)

            # go back
            dx2 = softmax_and_log(T, s_matrix=s_matrix, reverse=True)  # shape(2, num)
            dW1 = np.dot(dx2, y1.T)  # shape(2, h_number)
            dy1 = np.dot(W1.T, dx2)  # shape(h_number, num)
            dB1 = np.dot(dx2, np.ones((real_bitch_size, 1)))  # shape(2, 1)
            dx1 = dy1 * sigmod(y1, reverse=True)  # shape(h_number, num)
            dW0 = np.dot(dx1, X.T)  # shape()
            dB0 = np.dot(dx1, np.ones((real_bitch_size, 1)))  # shape()

            loss1 = np.sum(y_matrix) / real_bitch_size
            loss2 = loss1 + reg_bad * (np.sum(W1 ** 2) / 2 + np.sum(W0 ** 2) / 2)
            if param['min_loss2']:
                if param['min_loss2'] > loss2:
                    print('find min_loss2 -------------------------> ' + str(loss2))
                    param['min_loss2'] = loss2
                    param['min_loss1'] = loss1
                    param['best_w1'] = W1
                    param['best_b1'] = B1
                    param['best_w0'] = W0
                    param['best_b0'] = B0
            else:
                param['min_loss2'] = loss2
                param['min_loss1'] = loss1
                param['best_w1'] = W1
                param['best_b1'] = B1
                param['best_w0'] = W0
                param['best_b0'] = B0
            # log loss to loss_list
            print(i, loss1, loss2)
            if i % 100 == 0:
                loss_list.append([i, loss1, loss2])
            W1 -= learning_rate * (dW1 + reg_bad*W1)
            B1 -= learning_rate * dB1
            W0 -= learning_rate * (dW0 + reg_bad*W0)
            B0 -= learning_rate * dB0
    return W0, B0, W1, B1, np.array(loss_list)


def to_one(a):
    one_size = a.shape[0]
    a_mean = np.mean(a, axis=1).reshape((one_size, 1))
    a_min = np.min(a, axis=1).reshape((one_size, 1))
    a_max = np.max(a, axis=1).reshape((one_size, 1))
    a = (a - a_mean) * 2 / (a_max - a_min)
    return a

data_source = {}
data_source['x0'], data_source['target'] = handle_input()
data_source['x0'] = to_one(data_source['x0'])
print(data_source['x0'].shape)
print(data_source['target'].shape)
param = {
    'h_n': 50,
    'epo': 20000,
    'reg': 1e-4,
    'bitch_size': 128,
    'learning_rate': 1,
    'best_w1': None,
    'best_b1': None,
    'best_w0': None,
    'best_b0': None,
    'min_loss1': None,
    'min_loss2': None
}
param['w0'], param['b0'], param['w1'], param['b1'], param['loss_list'] = train(source_x=data_source['x0'], source_t=data_source['target'], bitch_size=param['bitch_size'], hidden_number=param['h_n'], learning_rate=param['learning_rate'], epoch=param['epo'], reg_bad=param['reg'])
print('best loss = ', param['min_loss1'], param['min_loss2'])

np.save('cifar_paramters_result2.npy', param)
end = time.clock()
print('Running time: %s Seconds'%(end-start))
plt.title('min Loss, epoch = '+str(param['epo']))
plt.xlabel('epoch')
plt.ylabel('loss')
# draw loss line, loss+reg line
plt.plot(param['loss_list'][:, 0], param['loss_list'][:, 1], 'r.', param['loss_list'][:, 0], param['loss_list'][:, 2], 'g.')
plt.show()