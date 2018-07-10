import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def unpickle(file):  # use cifar extract data
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def handle_input():
    """
    将指定的一张图片显示出来
    """
    mydict = unpickle('data_batch_1')
    data = np.array(mydict[b'data'])  # shape(10000, 3072)
    pic = data[6]
    pic = pic.reshape((3,1024))
    data = []
    print(type(pic[0]), pic[0])
    data.append(pic[0].reshape((32,32)).T)
    data.append(pic[1].reshape((32,32)).T)
    data.append(pic[2].reshape((32,32)).T)
    data = np.array(data);
    print(data.shape)
    plt.imshow(data.T)
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def handle_input1():
    """"
    显示前100张图片
    """
    mydict = unpickle('data_batch_1')
    dataSource = np.array(mydict[b'data'])  # shape(10000, 3072)
    for i in range(100):
        pic = dataSource[i]
        pic = pic.reshape((3,1024))
        data = []
        print(type(pic[0]), pic[0])
        data.append(pic[0].reshape((32,32)).T)
        data.append(pic[1].reshape((32,32)).T)
        data.append(pic[2].reshape((32,32)).T)
        data = np.array(data);
        print(data.shape)
        plt.subplot(10, 10, i+1)
        plt.imshow(data.T)
        plt.axis('off')  # 不显示坐标轴
    plt.show()

handle_input1()
