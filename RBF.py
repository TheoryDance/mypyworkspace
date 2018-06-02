#!/usr/bin/python3
import numpy as np
import grand.myutils as my
import matplotlib.pyplot as plt
"""
使用自定义神经网络拟合
y = tan(x) x取值范围为[-1,+1]
y = cos(x) x取值范围[0,3]
"""
sample_num, hidden_num = 30, 30
input_size = 1
output_size = input_size
x = np.linspace(-1, 1, sample_num).reshape(input_size, sample_num)
target = np.cos(5*x)

W1 = 0.1 * (np.random.random((hidden_num, input_size)) - 0.5)
B1 = np.zeros((hidden_num, 1))
W2 = 0.1 * (np.random.random((output_size, hidden_num)) - 0.5)
B2 = np.zeros((output_size, 1))
alpha = 1e-3
error_up_limit = 1e4
# training
for i in range(50000 * 200):
    try:
        input1 = np.dot(W1, x) + B1         # x = (input_size, num), W1 = (hidden_num, input_size
        output1 = my.gauss(input1)
        input2 = np.dot(W2, output1)
        output2 = input2 + B2

        error = (target - output2)**2 / 2 / sample_num   # (1,n)
        error_output2 = (output2 - target) / sample_num    # (1,n)
        error_input2 = error_output2        # (1,n)
        error_output1 = np.dot(W2.T, error_output2)  # (hidden,1) * (1,n) = (hidden,n)
        error_w2 = np.dot(error_output2, output1.T)  # (1,n) * (n, hidden) = (1, hidden)
        error_b2 = np.sum(error_input2, axis=1, keepdims=True)
        error_input1 = error_output1 * my.gauss(input1, output1, True)  # (hidden,n) .*  (hidden,n)
        error_w1 = np.dot(error_input1, x.T)
        error_b1 = np.sum(error_input1, axis=1, keepdims=True)

        error = np.sum(error)
        W1 -= alpha * error_w1
        W2 -= alpha * error_w2
        B1 -= alpha * error_b1
        B1 -= alpha * error_b2
        print(i, error)
        if error < 1e-4 or error > error_up_limit:
            print("训练次数： i = " + str(i))
            plt.plot(x.reshape(sample_num), target.reshape(sample_num), 'r*')
            plt.plot(x.reshape(sample_num), output2.reshape(sample_num))
            plt.show()
            break
    except:
        print('failed : i = ' + str(i))
        break
print("训练次数： over.i = " + str(i))
plt.plot(x.reshape(sample_num), target.reshape(sample_num), 'r*')

# testing
sample_num = 200
x = np.linspace(-1, 1, sample_num).reshape(input_size, sample_num)
input1 = np.dot(W1, x) + B1         # x = (input_size, num), W1 = (hidden_num, input_size
output1 = my.gauss(input1)
input2 = np.dot(W2, output1)
output2 = input2 + B2
plt.plot(x.reshape(sample_num), output2.reshape(sample_num), 'g.')
plt.show()