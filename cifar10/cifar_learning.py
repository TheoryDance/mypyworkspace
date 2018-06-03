import numpy as np
import time
import matplotlib.pyplot as plt
import grand.myutils as my

start = time.clock()


def unpickle(file):  # use cifar extract data
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def handle_input():
    mydict = unpickle('data_batch_1')
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
    W0 = 5 * (np.random.random((hidden_number, input_num)) - 0.5)  # W0.shape = (hidden_number, input_num)
    B0 = np.zeros((hidden_number, 1))  # B0.shape = (hidden_number, 1)
    W1 = 5 * (np.random.random((output_num, hidden_number)) - 0.5)  # W1.shape = (output_num, hidden_number)
    B1 = np.zeros((output_num, 1))  # B1.shape = (output_num, 1)

    bitch_num = (int)(np.ceil(sample_num / bitch_size))
    for i in range(epoch):
        for ii in range(bitch_num):
            X = source_x[:, bitch_size * ii:bitch_size * (ii + 1)]
            T = source_t[:, bitch_size * ii:bitch_size * (ii + 1)]
            real_bitch_size = X.shape[1]
            # forward
            x1 = np.dot(W0, X) + B0
            y1 = my.sigmod(x1)
            x2 = np.dot(W1, y1) + B1
            y2 = x2
            s_matrix, y_matrix = my.softmax_and_log(T, x=y2)

            # go back
            dx2 = my.softmax_and_log(T, s=s_matrix, reverse=True)  # shape(2, num)
            dW1 = np.dot(dx2, y1.T)  # shape(2, h_number)
            dy1 = np.dot(W1.T, dx2)  # shape(h_number, num)
            dB1 = np.dot(dx2, np.ones((real_bitch_size, 1)))  # shape(2, 1)
            dx1 = dy1 * my.sigmod(y1, reverse=True)  # shape(h_number, num)
            dW0 = np.dot(dx1, X.T)  # shape()
            dB0 = np.dot(dx1, np.ones((real_bitch_size, 1)))  # shape()

            loss1 = np.sum(y_matrix) / real_bitch_size
            loss2 = loss1 + reg_bad * (np.sum(W1 ** 2) + np.sum(W0 ** 2)) / 2
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
            if i % 10 == 0:
                loss_list.append([i, loss1, loss2])
            W1 -= learning_rate * (dW1 + reg_bad * W1)
            B1 -= learning_rate * dB1
            W0 -= learning_rate * (dW0 + reg_bad * W0)
            B0 -= learning_rate * dB0
    return W0, B0, W1, B1, np.array(loss_list)


def to_one(x, has_param=False, param_x_mean=None, param_x_max=None):
    if has_param:
        x = x - param_x_mean
        return x / param_x_max
    else:
        x_mean = np.mean(x, axis=1, keepdims=True)
        x = x - x_mean
        x_max = np.max(x, axis=1, keepdims=True)
        return x / x_max, x_mean, x_max

data_source = {}
data_source['x0'], data_source['target'] = handle_input()
data_source['x0'], x_mean, x_max = to_one(data_source['x0'])
np.save('x_mean.npy', x_mean)
np.save('x_max.npy', x_max)
print(data_source['x0'].shape)
print(data_source['target'].shape)
param = {
    'h_n': 50,
    'epo': 300,
    'reg': 1e-3,
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
print('Running time: %s Seconds' % (end-start))
plt.title('min Loss, epoch = '+str(param['epo']))
plt.xlabel('epoch')
plt.ylabel('loss')
# draw loss line, loss+reg line
plt.plot(param['loss_list'][:, 0], param['loss_list'][:, 1], 'r.', param['loss_list'][:, 0], param['loss_list'][:, 2], 'g.')
plt.show()