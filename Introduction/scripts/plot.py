"""
Author: Zhou Chen
Date: 2019/10/28
Desc: About
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# regression data
x = np.random.random([10, 1]) * 10  # 生成[0, 10)的随机数
y = 2 * x + np.random.random([10, 1])
plt.figure(figsize=(12, 6))
# regression
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='r')
plt.plot(x, 2 * x, color='b')
plt.title('Regression')
# classification data
x = np.random.random([10, 1]) * 10
y = -x + 1 + np.random.random([10, 1])*5 - 2.5
# classification
labels = np.zeros([10, 1])
labels[y > (-x+1)] = 1
labels = labels.astype(np.int)
cmap = cm.rainbow(np.linspace(0.0, 1.0, 2))
colors = cmap[labels.reshape(-1)]
plt.subplot(1, 2, 2)
plt.scatter(x, y, c=colors)
plt.plot(x, -x+1, color='b')
plt.title('classification')

# save png
plt.savefig('../assets/plot.png')
plt.show()

plt.figure(figsize=(12, 8))
x = np.random.random(20) * 10
y = x + (np.random.random(20) - 0.5) * 10
labels = np.zeros([20])
labels[x > 5.0] = 1
labels = labels.astype(np.int)
cmap = cm.rainbow(np.linspace(0.0, 1.0, 2))
colors = cmap[labels.reshape(-1)]
plt.scatter(x, y, c=colors)
plt.savefig('../assets/cluster.png')
plt.show()