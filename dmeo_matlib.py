#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0,2*np.pi,100)

ax1 = plt.subplot(1,2,1, projection='polar')
ax2 = plt.subplot(1,2,2)
ax1.bar(theta,theta/6)
ax2.plot(theta,theta/6,'--',lw=2)
plt.show()