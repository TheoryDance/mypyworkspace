import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)
plt.plot((2,4,6,8),'k--')

ax3 = fig.add_subplot(2,2,3)
from numpy.random import randn
plt.plot(randn(50).cumsum(), 'k--')


ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30) + 3*randn(30))
plt.show()