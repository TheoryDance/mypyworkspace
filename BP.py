#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
"""
使用自定义神经网络拟合y = cos(x) x取值范围[0,3]
y = tan(x) x取值范围为[-1,+1]
"""

"""
    先拟合y = tan(x) x取值范围为[-1,+1]
"""
sample_num = 30
input_size = 1
output_size = 1
hidden_num = 30
x = np.linspace(-1, 1, sample_num)
target = np.cos(5*x)
# print(x)
print(x.shape)          # (200,)
print(target.shape)     # (200,)

# plt.plot(x, target)
# plt.show()

x = x.reshape(input_size, sample_num)
target = target.reshape(output_size, sample_num)
print(x.shape)


def sigmod(x, model=False):
    # return np.exp(x)
    if model:  # back
        return x * (1 - x)
    else:  # forward
        return 1/(1 + np.exp(-x))


def gauss(x, y, model=False):
    if model:  # back
        return -2 * x * y
    else:  # forward
        return np.exp(-x ** 2)



W1 = 0.01 * (2 * np.random.random((hidden_num, input_size)) - 1)
B1 = np.zeros((hidden_num, 1))
W2 = 0.01 * (2 * np.random.random((output_size, hidden_num)) - 1)
B2 = np.zeros((output_size, 1))
delta = 0.001
lastoutput = 10000
for i in range(50000 * 50):
    try:
        input1 = W1.dot(x) + B1.dot(np.ones((1, sample_num)))         # x = (input_size, num), W1 = (hidden_num, input_size
        output1 = gauss(input1, 0)
        input2 = W2.dot(output1)
        output2 = input2 + B2.dot(np.ones((1, sample_num)))

        error = (target - output2)**2 / 2   # (1,n)
        error_output2 = output2 - target  # (1,n)
        error_input2 = error_output2  # (1,n)
        error_output1 = W2.T.dot(error_output2)  # (hidden,1) * (1,n) = (hidden,n)
        error_w2 = error_output2.dot(output1.T)  # (1,n) * (n, hidden) = (1, hidden)
        error_b2 = error_input2.dot(np.ones((sample_num, 1)))  # (1,1)
        error_input1 = error_output1 * gauss(input1, output1, True)  # (hidden,n) .*  (hidden,n)
        error_w1 = error_input1.dot(x.T)
        error_b1 = error_input1.dot(np.ones((sample_num, 1)))

        error = np.sum(error)
        W1 -= delta * error_w1  # * np.min((1, error))
        W2 -= delta * error_w2  # * np.min((1, error))
        B1 -= delta * error_b1  # * np.min((1, error))
        B1 -= delta * error_b2  # * np.min((1, error))
        print(error)
        if error < 0.0001 or lastoutput < error:
            print("训练次数： i = " + str(i))
            plt.plot(x.reshape(sample_num), target.reshape(sample_num), 'r*')
            plt.plot(x.reshape(sample_num), output2.reshape(sample_num))
            plt.show()
            break
        else:
            lastoutput = error
        if i % 5000 == 0:
            print("训练次数： i = " + str(i))
    except:
        print('failed : i = ' + str(i))
        break
print("训练次数： over.i = " + str(i))
plt.plot(x.reshape(sample_num), target.reshape(sample_num), 'r*')
plt.plot(x.reshape(sample_num), output2.reshape(sample_num))
# print('W1', W1)
# print('W2', W2)
# print('B1', B1)
# print('B2', B2)
plt.show()