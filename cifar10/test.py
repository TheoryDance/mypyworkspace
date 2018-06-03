import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


mydict = unpickle('data_batch_2')
data = np.array(mydict[b'data'])
labels_list = mydict[b'labels']
aa = 10
for i in range(aa**2):
    item = data[i]
    item = item.reshape((3, 32, 32))
    item[0] = item[0].T
    item[1] = item[1].T
    item[2] = item[2].T
    plt.subplot(aa, aa, i+1)
    plt.imshow(item.T)
    plt.axis('off')
plt.show()



# lend = mpimg.imread('../pic/6.jpg')
# plt.imshow(lend)
# plt.title('lend')
# plt.axis('off')
# plt.show()
