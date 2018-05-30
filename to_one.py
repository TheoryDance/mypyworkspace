#coding=utf-8
import numpy as np


a = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 8]])
a_mean = np.mean(a, axis=1).reshape((3,1))
a_min = np.min(a, axis=1).reshape((3,1))
a_max = np.max(a, axis=1).reshape((3,1))
a = (a - a_mean)*2 / (a_max - a_min)
print(a)