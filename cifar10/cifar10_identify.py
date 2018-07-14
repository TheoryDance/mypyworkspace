import numpy as np
import matplotlib.pyplot as plt
import grand.myutils as my


def unpickle(file):
    """
    按照cifar10官网上的解析数据文件
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class MyNet:
    def __init__(self, learning_rate=0.02, hidden_num=30, hidden_add=5, my_active=my.sigmod):
        """指定网络的超级参数：学习率、网络结构、激活函数等"""
        self.learning_rate = learning_rate
        self.hidden_num = hidden_num
        self.hidden_add = hidden_add
        self.my_active = my_active

    def parse_data(self, file_name):
        """按照cifar10官网上的解析数据文件"""
        data_source = unpickle(file_name)
        data = data_source[b'data']  # shape(sample_num, row_data)
        labels = data_source[b'labels']  # is list
        output_size = 10
        sample_num = data.shape[1]

        labels_eye = np.zeros((output_size, sample_num))
        data = data.T  # shape(col_data, sample_num)

        for i in range(sample_num):
            sample_i_label = labels[i]
            labels_eye[sample_i_label, i] = 1

        self.sample_data = data
        self.labels_eye = labels_eye
        self.sample_num = sample_num
        self.input_size = data.shape[0]
        self.output_size = output_size
        self.labels_source = labels

        # 解析label的含义
        data_meta = unpickle('batches.meta')
        labels_name = data_meta[b'label_names']
        self.labels_name = labels_name
        return data, labels_eye

    def show(self, start_index, end_index, w, h):
        cursor = start_index
        index = 0
        while cursor < end_index:
            index += 1
            # 将数据转为图片显示需要的格式
            item = self.sample_data[:, cursor].T  # row data
            item = item.reshape((3, 32, 32))
            plt.subplot(w, h, index)
            lend = []
            lend.append(item[0].T)
            lend.append(item[1].T)
            lend.append(item[2].T)
            lend = np.array(lend)
            lend = lend.T
            # 获取该张图片对应的标签
            plt.title(self.labels_name[self.labels_source[cursor]])
            plt.imshow(lend)
            plt.axis('off')
            cursor += 1
        plt.show()

    def train(self):
        # 对训练数据进行标准化与归一化处理
        sample_data_to_one, x_mean, x_max = my.to_one(self.sample_data)
        self.sample_data_to_one = sample_data_to_one
        self.x_mean = x_mean
        self.x_max = x_max

        # 初始化W与B，使用softmax作为损失函数
        pass

    def test(self, test_data_path):
        # 对测试数据进行标准化与归一化处理
        pass

mynet = MyNet()
mynet.parse_data('data_batch_1')
mynet.show(0, 25, 5, 5)
