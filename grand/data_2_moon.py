# 2个半圆环数据
import numpy as np


def make(radius=5, d=2, offset_x=5, offset_y=1, sample_num=800):
    sample_list = []
    for i in range(sample_num // 2):
        temp_radius = radius - d / 2 + d * np.random.random()
        temp_alpha = np.random.random() * np.pi
        x = temp_radius * np.cos(temp_alpha)
        y = temp_radius * np.sin(temp_alpha)
        sample_list.append([x, y, 1])

        temp_radius = radius - d / 2 + d * np.random.random()
        temp_alpha = np.random.random() * np.pi
        x = 2 * offset_x + temp_radius * np.cos(temp_alpha)
        y = temp_radius * np.sin(temp_alpha)
        sample_list.append([x, y, 1])

    for i in range(sample_num // 2):
        temp_radius = radius - d / 2 + d * np.random.random()
        temp_alpha = (1 + np.random.random()) * np.pi
        x = offset_x + temp_radius * np.cos(temp_alpha)
        y = offset_y + temp_radius * np.sin(temp_alpha)
        sample_list.append([x, y, 0])
    return sample_list


data = make(offset_y=3)
np.save('data_3_moon.npy', data)