"""
kmeans聚类算法：将半月形的数据读取出来，然后进行kmeans进行聚类
再使用RBF神经网络，将上半月的值设置为1，下半月的值设置为0，最后线性偏执为-1或其他，进行拟合曲线
然后遍历整个空间，将向量带入网络，如果网络的输出为0.5的点信息存储到集合中（即是值为0.5的等势线），绘制成图
"""
import numpy as np
import matplotlib.pyplot as plt

data_sample1 = np.load("data_sample1.npy")
data_sample2 = np.load("data_sample2.npy")


def kmeans_calc(k, data):
    """
    将给出的集合data进行集合，聚合最终点个数为K个
    :param k: 需要聚合的点个数
    :param data: 点集合,shape = (n, 2)
    :return:
    """
    center_list = []
    # init center
    for i in range(k):
        center_list.append(data[i].tolist())
    point_list1 = digui(center_list, k, data)
    return center_list, point_list1


def digui(center_list, k, data):
    point_list = []
    for i in range(k):
        point_list.append([])
    # foreach list
    for temp in data:
        index = 0
        distance = -1
        # 计算那个中心点距离当前点最近，得到中心点的下标
        for i in range(k):
            distance1 = (temp[0] - center_list[i][0]) ** 2 + (temp[1] - center_list[i][1]) ** 2
            if distance1 < distance or distance < 0:
                distance = distance1
                index = i
        # 将该点添加到对应中心点的数组中
        point_list[index].append(temp.tolist())

    center_change_num = 0
    error = 0.01
    # 完成一轮，重新计算中心点，当中心点不发生变化为止
    for i in range(point_list.__len__()):
        center_new = np.mean(point_list[i], axis=0).tolist()
        center_old = center_list[i]
        if (center_new[0] - center_old[0]) ** 2 + (center_new[1] - center_old[1]) ** 2 <= error:
            pass
        else:
            center_list[i] = center_new
            center_change_num += 1

    if center_change_num > 0:
        return digui(center_list, k, data)
    else:
        return point_list


plt.title('moon data show')
plt.xlabel('x')
plt.ylabel('y')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
centers1, points = kmeans_calc(7, data_sample1)
for i in range(len(centers1)):
    points_np = np.array(points[i])
    plt.plot(points_np[:, 0], points_np[:, 1], colors[i % 8])
    print(i, points_np.shape)
    plt.plot(centers1[i][0], centers1[i][1], 'b*')

centers2, points = kmeans_calc(7, data_sample2)
for i in range(len(centers2)):
    points_np = np.array(points[i])
    plt.plot(points_np[:, 0], points_np[:, 1], colors[i % 8])
    print(i, points_np.shape)
    plt.plot(centers2[i][0], centers2[i][1], 'b*')


centers1 = np.array(centers1)
centers1 = np.hstack((centers1, np.ones((centers1.shape[0], 1))))
centers2 = np.array(centers2)
centers2 = np.hstack((centers2, np.zeros((centers2.shape[0], 1))))
kmeans_result_data = np.vstack((centers1, centers2))
np.save('kmeans_result_data.npy', kmeans_result_data)
plt.show()