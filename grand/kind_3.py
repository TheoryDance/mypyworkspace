import os,sys
sys.path.append(r'C:\Users\Administrator\git\mypyworkspace')

import numpy as np
import grand.myutils as my
import matplotlib.pyplot as plt

# 从文件中加载样本数据
data = np.load('data_volution_k3.npy')
list_two = []
list_one = []
list_zero = []
target_two = []
target_one = []
target_zero = []
for item in data:
    if item[2] == 2:
        list_two.append([item[0], item[1]])
        target_two.append([0, 0, 1])
    elif item[2] == 1:
        list_one.append([item[0], item[1]])
        target_one.append([0, 1, 0])
    else:
        list_zero.append([item[0], item[1]])
        target_zero.append([1, 0, 0])
list_two = np.array(list_two)
list_one = np.array(list_one)
list_zero = np.array(list_zero)
sample_data = np.vstack((list_two, list_one, list_zero))
target_data = np.vstack((target_two, target_one, target_zero))
sample_data = sample_data.T
target_data = target_data.T

# 对样本数据进行归一化处理
sample_data, x_mean, x_max = my.to_one(sample_data)
output_size = target_data.shape[0]
input_size = sample_data.shape[0]
epoch = 20000
hidden_size = 30
alpha = 0.2
reg = 1e-4
# myactive_func = my.relu
# a_value = 2
myactive_func = my.sigmod
a_value = 10
batch_size = 256
batch_num = (int)(np.round(sample_data.shape[1] / batch_size))

w1 = a_value * np.random.random() * (np.random.random((output_size, hidden_size)) - 0.5)
w0 = a_value * np.random.random() * (np.random.random((hidden_size, input_size)) - 0.5)
b1 = np.zeros((output_size, 1))
b0 = np.zeros((hidden_size, 1))
best_w1, best_b1, best_w0, best_b0, best_loss1, best_loss2 = None, None, None, None, None, None

list_loss = []
for i in range(epoch):
    for j in range(batch_num):
        batch_sample = sample_data[:, j*batch_size:(j+1)*batch_size]
        batch_target = target_data[:, j*batch_size:(j+1)*batch_size]
        # 前向传播
        x1 = np.dot(w0, batch_sample) + b0
        y1 = myactive_func(x1)  # shape(hidden_size, num)
        x2 = np.dot(w1, y1) + b1
        y2 = x2
        s, yout = my.softmax_and_log(target=batch_target, x=y2)
        loss1 = np.sum(yout) / yout.shape[1]
        loss2 = loss1 + (np.sum(w0 ** 2) + np.sum(w0 ** 2)) * reg / 2
        if i % 100 == 0:
            list_loss.append([i, loss2])
        # 反向传播
        d_y2 = my.softmax_and_log(batch_target, s=s, reverse=True)
        d_x2 = d_y2  # shape(output_size, num)
        d_y1 = np.dot(w1.T, d_x2) # shape(hidden_size, num)
        d_b1 = np.sum(d_x2, axis=1, keepdims=True)
        d_w1 = np.dot(d_x2, y1.T)   # shape(output_size, hidden_size)

        d_x1 = d_y1 * myactive_func(y1, reverse=True)
        d_w0 = np.dot(d_x1, batch_sample.T)
        d_b0 = np.sum(d_x1, axis=1, keepdims=True)

        w0 -= alpha * (d_w0 + reg*w0)
        w1 -= alpha * (d_w1 + reg*w1)
        b0 -= alpha * d_b0
        b1 -= alpha * d_b1
        if best_loss2:
            if best_loss2 > loss2:
                best_w1, best_b1, best_w0, best_b0, best_loss1, best_loss2 = w1, b1, w0, b0, loss1, loss2
        else:
            best_w1, best_b1, best_w0, best_b0, best_loss1, best_loss2 = w1, b1, w0, b0, loss1, loss2
        print(i, loss1, loss2)

# make test data
x_test = np.linspace(-30, 30, 120)
y_test = np.linspace(-30, 30, 120)
x_test_data, y_test_data = np.meshgrid(x_test, y_test)
point_num = x_test_data.shape[0] * x_test_data.shape[1]
x_test_data = x_test_data.reshape((1, point_num))
y_test_data = y_test_data.reshape((1, point_num))
test_data = np.vstack((x_test_data, y_test_data))
print('best loss1: ', best_loss1, ', loss2: ', best_loss2)
# test
test_data_toone = my.to_one(test_data, has_param=True, param_x_mean=x_mean, param_x_max=x_max)
x1 = np.dot(best_w0, test_data_toone) + best_b0
y1 = myactive_func(x1)  # shape(hidden_size, num)
x2 = np.dot(best_w1, y1) + best_b1
y2 = x2
y2_max = np.max(y2, axis=0, keepdims=True)
y2 = y2 - y2_max
y2[y2 >= 0] = 1
y2[y2 < 0] = 0


test_two = []
test_one = []
test_zero = []
test_num = y2.shape[1]
print(y2)
for i in range(test_num):
    item = y2[:, i]
    pos = test_data[:, i]
    if np.sum((item - np.array([1, 0, 0]))**2) < 1e-3:
        test_zero.append(pos)
    elif np.sum((item - np.array([0, 1, 0]))**2) < 1e-3:
        test_one.append(pos)
    else:
        test_two.append(pos)

test_two = np.array(test_two)
test_one = np.array(test_one)
test_zero = np.array(test_zero)

list_loss = np.array(list_loss)
plt.subplot(2, 1, 1)
plt.title('Loss / epoch')
plt.ylabel('loss')
plt.plot(list_loss[:, 0], list_loss[:, 1], 'r.')

plt.subplot(2, 1, 2)
plt.title('test point')
print(test_two.shape)
print(test_one.shape)
print(test_zero.shape)
plt.plot(test_two[:, 0], test_two[:, 1], 'w.', test_one[:, 0], test_one[:, 1], 'b.', test_zero[:, 0], test_zero[:, 1], 'k.')
plt.plot(list_two[:, 0], list_two[:, 1], 'r*', list_one[:, 0], list_one[:, 1], 'g*', list_zero[:, 0], list_zero[:, 1], 'b*')
plt.show()