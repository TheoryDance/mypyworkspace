# 螺旋数据

import numpy as np


def make(k=5, alpha=2*np.pi, sample_num=800):
    sample_list = []
    matrix_alhpa = np.linspace(0, alpha, sample_num // 2)
    for temp_alpha in matrix_alhpa:
        temp_radius = k * temp_alpha
        x = temp_radius * np.cos(temp_alpha)
        y = temp_radius * np.sin(temp_alpha)
        sample_list.append([x, y, 0])

        x = temp_radius * np.cos(temp_alpha + np.pi*2/3)
        y = temp_radius * np.sin(temp_alpha + np.pi*2/3)
        sample_list.append([x, y, 1])

        x = temp_radius * np.cos(temp_alpha + np.pi * 4 / 3)
        y = temp_radius * np.sin(temp_alpha + np.pi * 4 / 3)
        sample_list.append([x, y, 2])

    return sample_list


data = make()
print(data)
np.save('data_volution_k3.npy', data)