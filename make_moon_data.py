#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import codecs
import random
"""
    双半月样本数据生成并存数据到文件，样本1000个，测试数据2000个，两个类型各占一半
"""
# 定义半月的参数：半径，距离，宽度
radius, distance, width = 10, -5, 6
data_sample1 = []
data_sample2 = []
data_test1 = []
data_test2 = []
for i in range(500):
    r = radius + (random.random() - 0.5) * width
    alpha = random.random() * np.pi
    data_sample1.append([r*np.cos(alpha), r*np.sin(alpha)])

for i in range(500):
    r = radius + (random.random() - 0.5) * width
    alpha = (random.random() + 1) * np.pi
    data_sample2.append([r*np.cos(alpha), r*np.sin(alpha)])

for i in range(1000):
    r = radius + (random.random() - 0.5) * width
    alpha = random.random() * np.pi
    data_test1.append([r*np.cos(alpha), r*np.sin(alpha)])

for i in range(1000):
    r = radius + (random.random() - 0.5) * width
    alpha = (random.random() + 1) * np.pi
    data_test2.append([r*np.cos(alpha), r*np.sin(alpha)])

data_sample1 = np.array(data_sample1)
data_sample2 = np.array(data_sample2)
data_test1 = np.array(data_test1)
data_test2 = np.array(data_test2)

# 下半月进行偏移
data_sample2[:, 0] = data_sample2[:, 0] + radius    # 水平偏移radius
data_sample2[:, 1] = data_sample2[:, 1] - distance    # 水平偏移distance
data_test2[:, 0] = data_test2[:, 0] + radius    # 水平偏移radius
data_test2[:, 1] = data_test2[:, 1] - distance    # 水平偏移distance

print('data_sample1.shape = ', data_sample1.shape)
print('data_sample2.shape = ', data_sample2.shape)
print('data_test1.shape = ', data_test1.shape)
print('data_test2.shape = ', data_test2.shape)

# 将样本数据存入到文件中
simple_file = codecs.open('data_simple1.txt', 'w', 'utf-8')
simple_file.write(str(data_sample1))
simple_file.close()
simple_file = codecs.open('data_simple2.txt', 'w', 'utf-8')
simple_file.write(str(data_sample2))
simple_file.close()
# 将测试数据存入到文件中
simple_file = codecs.open('data_test1.txt', 'w', 'utf-8')
simple_file.write(str(data_test1))
simple_file.close()
simple_file = codecs.open('data_test2.txt', 'w', 'utf-8')
simple_file.write(str(data_test2))
simple_file.close()

plt.plot(data_sample1[:, 0], data_sample1[:, 1], 'r*', data_sample2[:, 0], data_sample2[:, 1], 'b*')
plt.plot(data_test1[:, 0], data_test1[:, 1], 'b.', data_test2[:, 0], data_test2[:, 1], 'r.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sample and test data')
plt.show()