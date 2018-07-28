# 10000 examples
# t10k-images.idx3-ubyte  test set images 1648877 bytes
# t10k-labels.idx1-ubyte  test set labels 4542 bytes

# 60000 examples    shape=(60000, 784)
# train-images.idx3-ubyte  training set images 9912422 bytes
# train-labels.idx1-ubyte  training set labels 28881 bytes


import numpy as np
# train_labels = np.load('train-labels.idx1-ubyte')
# print(train_labels.shape())

f = open('train-labels.idx1-ubyte', 'r')
for i in f:
    print(i, type(i))
# content = f.readlines()
# print(len(content))
f.close()
