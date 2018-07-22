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
    def __init__(self, learning_rate=0.02, hidden_num=30, hidden_add=5, my_active=my.sigmod, batch_size=2**6):
        """
        指定网络的超级参数：学习率、网络结构、激活函数等
        :param learning_rate: 学习率
        :param hidden_num: 隐藏层神经元初始个数
        :param hidden_add: 隐藏层神经元增加个数步长
        :param my_active: 使用的激活函数
        :param batch_size: 批处理大小设置
        """
        self.learning_rate = learning_rate
        self.hidden_num = hidden_num
        self.hidden_add = hidden_add
        self.my_active = my_active
        self.batch_size = batch_size

    def parse_data(self, file_name):
        """
        按照cifar10官网说明进行数据解析，并将其转化为需要的格式
        :param file_name: 需要解析的文件路径
        :return:
        """
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
        """
        将解析的数据，通过图片展示的方式展示出来，会将图片对应的分类标签一起显示，更加直观的显示数据
        :param start_index: 展示图的开始序号
        :param end_index: 展示图的结束序号
        :param w: 显示图片的figure的图横向个数
        :param h: 显示图片的figure的图纵向个数
        :return:
        """
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
        """
        开始对数据进行训练，会自动训练处W与B的权值
        :return:
        """
        # 对训练数据进行标准化与归一化处理
        sample_data_to_one, x_mean, x_max = my.to_one(self.sample_data)
        self.sample_data_to_one = sample_data_to_one
        self.x_mean = x_mean
        self.x_max = x_max
        # 初始化W与B，使用softmax作为损失函数
        self.W0 = np.random.random((self.hidden_num, self.input_size))
        self.B0 = np.zeros((self.hidden_num, 1))
        self.W1 = np.random.random((self.output_size, self.hidden_num))
        self.B1 = np.zeros((self.output_size, 1))

        # 计算样本的批个数，循环使用每批数据进行权值更新与迭代
        pass

    def test_correct_rate(self, test_data_path):
        """
        测试样本正确率
        :param test_data_path:
        :return:
        """
        # 对测试数据进行标准化与归一化处理
        correct_rate = 0
        return correct_rate

mynet = MyNet()
mynet.parse_data('data_batch_1')
mynet.show(0, 25, 5, 5)
