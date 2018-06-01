import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

lend = mpimg.imread('6.jpg')
# lend = mpimg.imread('2.jpg')
print(type(lend), lend.shape)
plt.imshow(lend)
plt.axis('off')  # 不显示坐标轴
plt.show()



