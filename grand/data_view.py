# 数据可视化展现

import numpy as np
import matplotlib.pyplot as plt

path = 'data_xor.npy'
data = np.load(path)
print(type(data))

list_one = []
list_zero = []
for item in data:
    if item[2] == 1:
        list_one.append(item)
    else:
        list_zero.append(item)

list_one = np.array(list_one)
list_zero = np.array(list_zero)
plt.title(path)
plt.plot(list_one[:, 0], list_one[:, 1], 'r*', list_zero[:, 0], list_zero[:, 1], 'g*')
plt.show()