# 圆圈数据分类

import numpy as np


def make(center=(3, 4), radius=5, sample_num=1000, rate=0.4):
    """
    :param center: 分界圆中心坐标
    :param radius: 分界圆半径
    :param sample_num: 样本个数
    :param rate: 圆内样本个数占总样本数的比例
    :return:
    """
    sample_list = []
    # 圆内的样本
    for i in range((int)(sample_num * rate)):
        # 随机半径
        temp_radius = np.random.random() * (radius - 1) + 1
        # 随机弧度
        temp_alpha = np.random.random() * 2 * np.pi
        temp_x = center[0] + temp_radius * np.cos(temp_alpha)
        temp_y = center[1] + temp_radius * np.sin(temp_alpha)
        sample_list.append([temp_x, temp_y, 1])
    for i in range(sample_num - (int)(sample_num * rate)):
        # 随机半径
        temp_radius = (1 + np.random.random()) * radius
        # 随机弧度
        temp_alpha = np.random.random() * 2 * np.pi
        temp_x = center[0] + temp_radius * np.cos(temp_alpha)
        temp_y = center[1] + temp_radius * np.sin(temp_alpha)
        sample_list.append([temp_x, temp_y, 0])
    return sample_list


data = make(rate=0.35)
np.save('data_circle.npy', data)











