#!/usr/bin/python3
import numpy as np
import grand.myutils as my
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
print(x.shape)          # (200,)
print(target.shape)     # (200,)

x = x.reshape(input_size, sample_num)
target = target.reshape(output_size, sample_num)
print(x.shape)

W1 = 0.01 * (2 * np.random.random((hidden_num, input_size)) - 1)
B1 = np.zeros((hidden_num, 1))
W2 = 0.01 * (2 * np.random.random((output_size, hidden_num)) - 1)
B2 = np.zeros((output_size, 1))
alpha = 1e-3
lastoutput = 1e4
for i in range(50000*10):
    try:
        input1 = W1.dot(x) + B1.dot(np.ones((1, sample_num)))         # x = (input_size, num), W1 = (hidden_num, input_size
        output1 = my.gauss(input1, 0)
        input2 = W2.dot(output1)
        output2 = input2 + B2.dot(np.ones((1, sample_num)))

        error = (target - output2)**2 / 2   # (1,n)
        error_output2 = output2 - target    # (1,n)
        error_input2 = error_output2        # (1,n)
        error_output1 = W2.T.dot(error_output2)  # (hidden,1) * (1,n) = (hidden,n)
        error_w2 = error_output2.dot(output1.T)  # (1,n) * (n, hidden) = (1, hidden)
        error_b2 = np.sum(error_input2, axis=1, keepdims=True)
        error_input1 = error_output1 * my.gauss(input1, output1, True)  # (hidden,n) .*  (hidden,n)
        error_w1 = error_input1.dot(x.T)
        error_b1 = np.sum(error_input1, axis=1, keepdims=True)

        error = np.sum(error)
        W1 -= alpha * error_w1
        W2 -= alpha * error_w2
        B1 -= alpha * error_b1
        B1 -= alpha * error_b2
        print(i, error)
        if error < 1e-3 or lastoutput < error:
            print("训练次数： i = " + str(i))
            plt.plot(x.reshape(sample_num), target.reshape(sample_num), 'r*')
            plt.plot(x.reshape(sample_num), output2.reshape(sample_num))
            plt.show()
            break
        else:
            lastoutput = error
    except:
        print('failed : i = ' + str(i))
        break
print("训练次数： over.i = " + str(i))
plt.plot(x.reshape(sample_num), target.reshape(sample_num), 'r*')
plt.plot(x.reshape(sample_num), output2.reshape(sample_num))
plt.show()